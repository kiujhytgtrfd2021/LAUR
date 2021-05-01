import sys, time, os
import numpy as np
import random
import torch
from copy import deepcopy
import utils
from utils import *
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import *
import math

sys.path.append('..')
from arguments import get_args

args = get_args()

from bayes_layer import BayesianLinear,  _calculate_fan_in_and_fan_out, custom_regularization
from transformers import get_linear_schedule_with_warmup,AdamW

import datetime
class Appr(object):
    

    def __init__(self, model, nepochs=100, sbatch=256, lr=0.001, 
                 lr_min=2e-6, lr_factor=3, lr_patience=5, clipgrad=100, args=None, log_name=None, split=False,task_names = None):

        self.model = model
        self.model_old = deepcopy(self.model)
        file_name = log_name
        self.logger = utils.logger(file_name=file_name, resume=False, path='result_data/csvdata/', data_format='csv')

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.clipgrad = clipgrad
        self.args = args
        self.iteration = 0
        self.split = split
        
        self.tasks = task_names
        
        
        

        

        return


    

    def train(self, t, train_dataloader, test_dataloader, data,optimizer,scheduler,regular):
        best_loss = np.inf
        best_acc = 0
        best_model = utils.get_model(self.model)
        self.model_old = deepcopy(self.model)
        # utils.freeze_model(self.model_old)  # Freeze the weights


        self.optimizer = optimizer
        self.scheduler = scheduler
        # initial best_avg
        valid_acc_t = {}
        valid_acc_t_norm = {}

        # Loop epochs
        for e in range(self.nepochs):
            

            # Train
            clock0 = time.time()
            avg = 0
            # num_batch = xtrain.size(0)
            num_batch = len(train_dataloader)
            self.model.set_dataset(self.tasks[t],use_class=True)
            self.train_epoch(t, train_dataloader,regular)
            
            clock1 = time.time()
            train_loss, train_acc = self.eval(t, train_dataloader,regular)
            
            clock2 = time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                e + 1, 1000 * self.sbatch * (clock1 - clock0) / num_batch,
                1000 * self.sbatch * (clock2 - clock1) / num_batch, train_loss, 100 * train_acc), end='')
            # Valid
            
            valid_loss, valid_acc = self.eval(t, test_dataloader,regular)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')

        
            if valid_acc >= best_acc:
                best_acc = valid_acc
                best_model = utils.get_model(self.model)
                # patience = self.lr_patience
                print(' *,best_model',end=" ")

            print()
            utils.freeze_model(self.model_old)


    
        utils.set_model_(self.model, best_model)
        self.model_old = deepcopy(self.model)
        # self.calculate_import_percent()
        best_avg = 0
        for task in range(t+1):
            self.model.set_dataset(self.tasks[task],use_class=True)
            valid_loss_t, valid_acc_t[task] = self.eval(task, data[task],regular)
            best_avg += valid_acc_t[task]
            print('{} test: loss={:.3f}, acc={:5.1f}% |'.format(task,valid_loss_t, 100 * valid_acc_t[task]), end='')
            self.logger.add(epoch=(t * self.nepochs) + e+1, task_num=task + 1, test_loss=valid_loss_t,
                            test_acc=valid_acc_t[task])
        
        print('best_avg_Valid:  acc={:5.1f}% |'.format(100 * best_avg/(t+1)), end='')
        self.logger.add(task= t, avg_acc =100 * best_avg/(t+1) )
        self.logger.save()
        torch.save(self.model,'_task_{}.pt'.format(t))

        return
    def flat_accuracy(self,preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)



    def format_time(self,elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))
        
        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def train_epoch(self,t,train_dataloader,regular):
        device = 'cuda'
        total_loss = 0
        self.model.train()
        for step, batch in enumerate(tqdm(train_dataloader,desc="train")):
        # for step,batch in enumerate(train_dataloader):
            

            b_input_ids = batch[0].to(device)
            # print(b_input_ids.size())
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            self.model.zero_grad()
            if regular:       
                outputs = self.model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels,
                            sample = True,
                            use_class = True)
            else:
                outputs = self.model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels,
                            use_class = True)
            # The call to `model` always returns a tuple, so we need to pull the 
            # loss value out of the tuple.
            loss = outputs[0]

            # regualr_loss = self.custom_regularization(self.model_old, self.model, self.sbatch, loss)
            # loss = self.custom_regularization(self.model_old, self.model, self.sbatch, loss)
            if t!=0 and regular:
                loss = custom_regularization(self.model_old, self.model, self.sbatch, loss)

            # total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0)

            # if t>0:
            #     for n,p in self.model.named_modules():
            #         if isinstance(p, BayesianLinear)==False:
            #             # # if isinstance(p,nn.Embedding):
            #             # #     print(n)
            #             # #     p.weight.grad.data.fill_(0)
            #             # # elif 'LayerNorm' in n:
            #             # #     p.weight.grad.data.fill_(0)
            #             # #     p.bias.grad.data.fill_(0)
            #             # if isinstance(p,nn.Linear):
            #             #     # print(n)
            #             #     if p.weight.grad is not None:
            #             #         p.weight.grad.data.fill_(0)
            #             #         p.bias.grad.data.fill_(0)
            #             #     continue
            #             # else:
            #             #     # print("continue"+n)
            #             #     continue
            #             continue
            #         rho = p.weight_rho
            #         weight = p.weight_mu
            #         # print(rho.shape)
            #         sigma = torch.log1p(torch.exp(rho))

            #         fan_in, fan_out = _calculate_fan_in_and_fan_out(weight)

            #         if isinstance(p, BayesianLinear):
            #             std_init = 0.1* math.sqrt((2 / fan_in) * args.ratio)
                    
            #         update = (sigma/std_init)**2
            #         # print(update.shape)
            #         # print(p.bias.grad.data.shape)
            #         # p.bias.grad.data *= update.squeeze(1)
            #         # out_features,in_features = weight.shape
            #         # grad_s = update.expand(out_features,in_features)
            #         # p.weight_mu.grad.data *= grad_s
                    
            #         p.weight_rho.grad.data *= update 

            # if t!=0 and regular:
            #     loss2 = self.custom_regularization(self.model_old, self.model, 16)#, loss)
            #     loss2.backward()

            self.optimizer.step()  # the update of rho lr is in AdamW_bayes optimizer.step()
            self.scheduler.step()


            # if step % 1000 == 0 :
            #     train_loss, train_acc = self.eval(t, train_dataloader,regular)
            #     self.model.train()

            #     print('|step: {}|Train: loss={:.3f}, acc={:5.1f}% |'.format(
            #         step,train_loss, 100 * train_acc))

        avg_train_loss = total_loss / len(train_dataloader)            
            

        return


    def eval(self,t,test_dataloader,regular):

        t0 = time.time()
        device = 'cuda'

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        self.model.eval()

        # Tracking variables 
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for idx,batch in enumerate(test_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():    
                if regular:    
                    outputs = self.model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels,
                                    sample = False,
                                    use_class = True)
                else:
                    outputs = self.model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels,
                                    use_class = True)
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            loss = outputs[0]
            logits = outputs[1]
            eval_loss += loss.item()
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = self.flat_accuracy(logits, label_ids)
            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy

            # Track the number of batches
            nb_eval_steps += 1
        # print(outputs[0].size())
        # Report the final accuracy for this validation run.
        avg_eval_loss = eval_loss/nb_eval_steps

        return avg_eval_loss, eval_accuracy/nb_eval_steps


    

    def calculate_import_percent(self):

        for layer_idx, module in enumerate(self.model.modules()):
            if isinstance(module, BayesianLinear)==False:
                continue
            fan_in, fan_out = _calculate_fan_in_and_fan_out(module.weight_mu)
            
            # print(module.weight_mu)
            weight_sigma = torch.log1p(torch.exp(module.weight_rho))
            std_init = 0.1 * math.sqrt((2 / fan_in) * args.ratio)
            weight_strength = (std_init / weight_sigma)
            import_mu = torch.where((weight_strength)>1.10,torch.ones_like(weight_sigma),torch.zeros_like(weight_sigma))
            weight = import_mu.data
            num_params = weight.numel()
            num_one = weight.view(-1).eq(1).sum().item()
            print('Layer #%d: importance_percent %d/%d (%.2f%%)' % (layer_idx, num_one, num_params, 100 * num_one / num_params))
