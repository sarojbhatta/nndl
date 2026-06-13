import sys, os, torch, wandb
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

sys.path.extend(["./"])

from utils.figure_plot import plot_training_testing_curve
from utils.eval import evaluate, get_metrix
from utils.misc import save_csv_file
from utils.training_utils import get_threshold
from settings.config import Config


from models.Distillation.ts_residual_block import TS_ResidualNN
from models.Distillation.ts_SE_Residual import TS_SE_ResidualNN
from models.Residual_factory.residual_block import ResidualNN
from settings.config import Config


def distillation_loss(y_student, y_teacher, T=1, alpha=0.5):
    # y_student: student's predictions
    # y_teacher: teacher's predictions
    # T: distill_T = Temperature parameter (higher values make the output softer) 
    # alpha: distill_alpha = weight between standard loss and distillation loss
    
    # Traditional BCE Loss
    bce_loss = nn.BCELoss()(y_student, y_teacher)
    
    # Distillation Loss (KL Divergence)
    soft_teacher = torch.softmax(y_teacher / T, dim=1)  # Apply temperature scaling to teacher
    soft_student = torch.softmax(y_student / T, dim=1)  # Apply temperature scaling to student
    distillation_loss = nn.KLDivLoss()(torch.log(soft_student), soft_teacher) * (T * T)
    
    # Combined loss
    return alpha * bce_loss + (1 - alpha) * distillation_loss


def distill_loss(s_logits, t_logits, true_y, T, alpha):
    # standard BCE-with-logits on true labels
    bce = F.binary_cross_entropy_with_logits(s_logits, true_y)
    # softened targets 
    p_s = F.log_softmax(s_logits / T, dim=1)
    p_t = F.softmax(    t_logits / T, dim=1)
    kd  = F.kl_div(p_s, p_t, reduction='batchmean') * (T*T)
    return alpha * bce + (1 - alpha) * kd


'''def distillation_loss(student_logits, teacher_logits, T, alpha):
    # student_logits, teacher_logits: raw outputs BEFORE sigmoid
    p_student = torch.log_softmax(student_logits / T, dim=1)
    p_teacher = torch.softmax(teacher_logits / T, dim=1)
    kd_loss = F.kl_div(p_student, p_teacher, reduction='batchmean') * (T * T)
    bce_loss = F.binary_cross_entropy_with_logits(student_logits, targets)
    return alpha * kd_loss + (1 - alpha) * bce_loss
'''


def train_TeacherStudent(config: Config, model, epoch: int, optimizer, scheduler, loss_func, 
          train_loader, val_loader, test_loader, save_dir, seed, cat_list, cot_length):
    
    os.environ["WANDB_DIR"] = config.save_dir
    wandb.init(
        # set the wandb project where this run will be logged
        project="SMM-{}".format(config.dataset[0]),
        group="{}".format('deep learning-based methods'),
        name='seed' + str(seed) + '-' + config.Sampling_Strategy[0],
        # track hyperparameters and run metadata
        config=config.__dict__
    )

    if config.teacher_model == "residual_MLP":
        teacher_model = TS_ResidualNN(cat_list, cot_length, stack_layers=config.NN_model[1], 
                           act=config.NN_model[2], p=config.dropout)
    else:
        teacher_model = TS_SE_ResidualNN(cat_list, cot_length, stack_layers=config.NN_model[1], 
                              act=config.NN_model[2], p=config.dropout)
    
    #teacher_model.load(config.teacher_ckpt)
    teacher_model.load_state_dict(torch.load(config.teacher_ckpt))

    teacher_model.to(config.device)
    teacher_model.eval()



    training_loss = []
    validate_loss = []

    best_model_loss = 1e9
    best_epoch = 0

    training_loss_step = []
    training_loop = tqdm(range(epoch), desc='epoch')
    for _epoch in training_loop:
        model.train()

        training_SMM_count = 0
        total_training_samples = 0
        cur_train_loss = 0
        for i, (inputs_cat, inputs_cot, targets) in enumerate(train_loader):
            optimizer.zero_grad()

            #teacher model section
            with torch.no_grad():
                t_logits = teacher_model.logits(inputs_cat, inputs_cot)
            
            #student model section for logits
            s_logits = model.logits(inputs_cat, inputs_cot)

            # combined loss for teacher-student model
            #loss = distillation_loss(s_logits, t_logits, config.distill_T, config.distill_alpha)
            loss = distill_loss(s_logits, t_logits, targets, config.distill_T, config.distill_alpha)
            
            loss.backward()
            optimizer.step()

            cur_train_loss += loss.item() * len(targets)
            training_SMM_count += torch.sum(targets).item()
            total_training_samples += len(targets)
            training_loss_step.append(loss.item())

        cur_train_loss /= total_training_samples
        training_loss.append(cur_train_loss)
        wandb.log({"Training loss": cur_train_loss}, step=_epoch)

        # save checkpoint
        if _epoch % 10 == 0:
            cur_save_path = os.path.join(save_dir, 'ckpt')
            if not os.path.exists(cur_save_path):
                os.makedirs(cur_save_path)
            model.save(cur_save_path + '/model-{}.pth'.format(_epoch))

            cur_save_path = os.path.join(save_dir, 'training_samples_count')
            if not os.path.exists(cur_save_path):
                os.makedirs(cur_save_path)
            with open(cur_save_path + '/{}.txt'.format(str(_epoch)), 'w+') as f:
                f.write("{}/{}\n".format(training_SMM_count, total_training_samples))

        # validation
        cur_val_loss = evaluate(model, val_loader, loss_func)
        validate_loss.append(cur_val_loss)
        wandb.log({"Validate loss": cur_val_loss}, step=_epoch)

        if scheduler:
            wandb.log({"lr": scheduler.get_last_lr()[0]}, step=_epoch)
        else:
            wandb.log({"lr": optimizer.state_dict()['param_groups'][0]['lr']}, step=_epoch)
        # save model_best
        if best_model_loss > cur_val_loss:
            best_model_loss = cur_val_loss
            best_epoch = _epoch
            model.save(os.path.join(save_dir, 'model_best.pth'))
        
        # save model_final
        model.save(os.path.join(save_dir, 'model_final.pth'))

        if scheduler:
            scheduler.step()

        # test
        #threshold = get_threshold(model, train_loader) if type(model) == AutoEncoder else None
        metrics = get_metrix(model, test_loader, threshold=None)
        for key in metrics.keys():
            wandb.log({key: metrics[key]}, step=_epoch)

        if scheduler:
            cur_lr = scheduler.get_last_lr()[0]
        else:
            cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
        # print("{}/{}: lr={}, training_loss={}, val_loss={}\n".format(_epoch, epoch,
                                                                #    cur_lr, cur_train_loss, cur_val_loss))
        training_loop.set_postfix(lr=cur_lr, train_loss=cur_train_loss, test_loss=cur_val_loss,
                                  AUC=metrics['AUC'], Precision=metrics['Precision'], Recall=metrics['Recall'], 
                                  Acc=metrics['Accuracy'], F1_score=metrics['F1-score'])

    
    with open(os.path.join(save_dir, 'model_loss.txt'), 'w+') as f:
        f.write('model_best[{}]:{}\n'.format(best_epoch, best_model_loss))
        f.write('model_final[{}]:{}\n'.format(epoch, cur_val_loss))

    save_csv_file(training_loss_step, os.path.join(save_dir, 'training_loss_step.csv'))
    plot_training_testing_curve(training_loss_step, 'step', 'loss', 'training step-loss', 'training loss[step]', config.save_dir + '/training step-loss.png')
    
    return training_loss, validate_loss


'''Use Soft Targets from the Teacher Model
Instead of training the student model with the hard labels (0 or 1) directly, we will use the soft targets generated by the teacher model. The teacher model will output probabilities for each class, and the student will be trained to mimic this output.

The teachers output (soft_targets) is obtained by passing the inputs through the trained teacher model and using the softmax function to get the probability distribution.

Now, you need to define a distillation loss that combines the standard classification loss (like BCE loss) and the knowledge distillation loss, which is the difference between the soft targets predicted by the teacher and the student.

The distillation loss is a weighted sum of:

Traditional loss (e.g., binary cross-entropy for binary classification).

Distillation loss (KL-divergence between teachers soft targets and students predictions).

'''