import sys
import torch
import numpy as np
import random
from arguments import get_args
args = get_args()
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

from approaches import LAUR
from LAUR_model.model import LAUR_model,BertClassifier_base
from dataset import bert_train_loader

from transformers import get_linear_schedule_with_warmup,AdamW
import torch.nn as nn

from optimizer import AdamW_bayes,get_params




epochs = 1
lr = 3e-5
batch_size = 32
regular = False
valid_size = 0
num_warmup_steps_percent = 0
if args.approach == 'BERT':
    model = BertClassifier_base()
elif args.approach == 'LAUR':
    model = LAUR_model()
    regular = True
print("trainning :"+args.approach)

model.cuda()
# model = nn.DataParallel(model , device_ids=[0,1])


test_loader=[]

if args.tasks_order == 1:
    tasks = ['yelp', 'agnews', 'dbpedia', 'amazon', 'yahoo']
if args.tasks_order == 2:
    tasks = ['dbpedia', 'yahoo', 'agnews', 'amazon', 'yelp']
if args.tasks_order == 3:
    tasks= ['yelp', 'yahoo', 'amazon', 'dbpedia', 'agnews']
if args.tasks_order == 4:
    tasks = ['agnews', 'yelp', 'amazon', 'yahoo', 'dbpedia']

TC_NUM_CLASSES = {
    'yelp': 5,
    'yahoo': 10,
    'amazon': 5,
    'agnews': 4,
    'dbpedia': 14
}
print(tasks)
LAUR_train = LAUR.Appr(model,epochs,batch_size,args = args,log_name=args.logname,task_names = tasks)


for t in range(len(tasks)):
    train_dataloader,valid_dataloader,test_dataloader = bert_train_loader(tasks[t],valid_size,batch_size)
    test_loader.append(test_dataloader)
    

    model.add_dataset(tasks[t],TC_NUM_CLASSES[tasks[t]],use_class=True)
    model.set_dataset(dataset = tasks[t],use_class=True)



    model.cuda()

    

    rho_id,mu_id = [],[]
    for n,param in model.named_parameters():
        if 'rho' in n:
            rho_id.append(id(param))
        if 'weight_mu' in n:
            mu_id.append(id(param))
    params = get_params(model,lr)


    optimizer = AdamW_bayes(params,
                eps = 1e-8, # args.adam_epsilon  - default is 1e-8.
                rho_id=rho_id,
                mu_id=mu_id,
                min_import = 1.05,
                )


    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs
    print(len(train_dataloader))
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = num_warmup_steps_percent * total_steps, # Default value in run_glue.py
                                                num_training_steps = total_steps)


    print('training {}'.format(tasks[t]))
    LAUR_train.train(t,train_dataloader,valid_dataloader,test_loader,optimizer,scheduler,regular)
