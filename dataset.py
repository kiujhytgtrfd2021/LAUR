import collections
import glob
import os
import random
import numpy as np
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

from transformers import BertTokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import pandas as pd
import swifter
import re

TC_NUM_CLASSES = {
    'yelp': 5,
    'yahoo': 10,
    'amazon': 5,
    'agnews': 4,
    'dbpedia': 14
}

# from networks import bert_net

def read_files(file_class):
    path = "data/mtl-dataset/"
    train_label,test_label=[],[]
    train_text,test_text = [],[]
    with open(path+file_class+'.train',encoding = 'utf8',errors="ignore") as file_input:
        for line in file_input.readlines():
            train_label.append(int(line.split('\t')[0]))
            train_text.append(line.split('\t')[1])
    with open(path+file_class+'.test',encoding='utf8',errors="ignore") as file_input:
        for line in file_input.readlines():
            test_label.append(int(line.split('\t')[0]))
            test_text.append(line.split('\t')[1])
    return train_label,test_label,train_text,test_text



def bert_train_loader(data_name,validation_size,batch_size):

    if data_name in ['yelp','amazon','yahoo','dbpedia','agnews']:
        y_train,train_text = create_tc_data(data_name,mode = "train")
        y_test,test_text = create_tc_data(data_name,mode = 'test')
        y_train = list(np.array(y_train)-1) # change to right class
        y_test = list(np.array(y_test)-1)
        MAX_LEN=128

    elif data_name == "text_mutitask":
        y_train,train_text,y_test,test_text = [],[],[],[]
        num_class = 0
        for name in ['yelp','amazon','yahoo','dbpedia','agnews']:
            y_train_s,train_text_s = create_tc_data(name,mode = "train")
            y_test_s,test_text_s = create_tc_data(name,mode = 'test')
            y_train_s = list(np.array(y_train_s)+num_class-1) # change to right class
            y_test_s = list(np.array(y_test_s)+num_class-1)
            num_class += TC_NUM_CLASSES[name]

            y_train.extend(y_train_s)
            train_text.extend(train_text_s)
            y_test.extend(y_test_s)
            test_text.extend(test_text_s)
        MAX_LEN=128

    elif data_name == "sentiment_mutitask":
        y_train,train_text,y_test,test_text = [],[],[],[]
        num_class = 0
        for name in ['magazines.task','apparel.task','health_personal_care.task','camera_photo.task','toys_games.task','software.task','baby.task','kitchen_housewares.task','sports_outdoors.task',
                    'electronics.task','books.task','video.task','imdb.task','dvd.task','music.task','MR.task']:
            y_train_s,y_test_s,train_text_s,test_text_s=read_files(name)

            y_train.extend(y_train_s)
            train_text.extend(train_text_s)
            y_test.extend(y_test_s)
            test_text.extend(test_text_s)
        MAX_LEN=256

    else:
        y_train,y_test,train_text,test_text=read_files(data_name)
        MAX_LEN=256

    # Load the BERT tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    sentences = train_text
    labels = y_train
    test_sentences=test_text
    test_labels=y_test



    input_ids = [tokenizer.encode(sent,add_special_tokens=True,max_length=MAX_LEN,truncation=True) for sent in sentences]
    test_input_ids=[tokenizer.encode(sent,add_special_tokens=True,max_length=MAX_LEN,truncation=True) for sent in test_sentences]


    print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", 
                            value=0, truncating="post", padding="post")

    test_input_ids = pad_sequences(test_input_ids, maxlen=MAX_LEN, dtype="long", 
                            value=0, truncating="post", padding="post")


    # Create attention masks
    attention_masks = []

    # For each sentence...
    for sent in input_ids:
        
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]
        
        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)

    test_attention_masks = []

    # For each sentence...
    for sent in test_input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        test_attention_masks.append(att_mask)



    # # Use 90% for training and 10% for validation.
    if validation_size!=0:
        train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, 
                                                                    random_state=2020, test_size=validation_size)
        # Do the same for the masks.
        train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,
                                                    random_state=2020, test_size=validation_size)
    else:
        train_inputs,train_labels = input_ids,labels
        train_masks = attention_masks
        validation_inputs,validation_labels = test_input_ids,test_labels
        validation_masks = test_attention_masks

    train_inputs = torch.LongTensor(train_inputs)
    validation_inputs = torch.LongTensor(validation_inputs)
    test_inputs=torch.LongTensor(test_input_ids)

    train_labels = torch.LongTensor(train_labels)
    validation_labels = torch.LongTensor(validation_labels)
    test_labels=torch.LongTensor(test_labels)

    train_masks = torch.LongTensor(train_masks)
    validation_masks = torch.LongTensor(validation_masks)
    test_masks=torch.LongTensor(test_attention_masks)

    print(train_inputs.size())

    # The DataLoader needs to know our batch size for training, so we specify it 
    # here.
    # For fine-tuning BERT on a specific task, the authors recommend a batch size of
    # 16 or 32.


    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size,num_workers=2,pin_memory=True)

    # # Create the DataLoader for our validation set.
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size,num_workers=2,pin_memory=True)

    # Create the DataLoader for our test set.
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size,num_workers=2,pin_memory=True)
    if data_name == 'sentiment_mutitask':
        test_dataloader = []
        for i in range(16):
            test_data = TensorDataset(test_inputs[i*200:(i+1)*200], test_masks[i*200:(i+1)*200], test_labels[i*200:(i+1)*200])
            test_sampler = SequentialSampler(test_data)
            test_dataloader.append(DataLoader(test_data, sampler=test_sampler, batch_size=batch_size,num_workers=2,pin_memory=True))
    if data_name == 'text_mutitask':
        test_dataloader = []
        for i in range(5):
            test_data = TensorDataset(test_inputs[i*7600:(i+1)*7600], test_masks[i*7600:(i+1)*7600], test_labels[i*7600:(i+1)*7600])
            test_sampler = SequentialSampler(test_data)
            test_dataloader.append(DataLoader(test_data, sampler=test_sampler, batch_size=batch_size,num_workers=2,pin_memory=True))

    return train_dataloader,validation_dataloader, test_dataloader


def preprocess(text):
    """
    Preprocesses the text
    """
    text = text.lower()
    # removes '\n' present explicitly
    text = re.sub(r"(\\n)+", " ", text)
    # removes '\\'
    text = re.sub(r"(\\\\)+", "", text)
    # removes unnecessary space
    text = re.sub(r"(\s){2,}", u" ", text)
    # replaces repeated punctuation marks with single punctuation followed by a space
    # e.g, what???? -> what?
    text = re.sub(r"([.?!]){2,}", r"\1", text)
    # appends space to $ which will help during tokenization
    text = text.replace(u"$", u"$ ")
    # replace decimal of the type x.y with x since decimal digits after '.' do not affect, e.g, 1.25 -> 1
    text = re.sub(r"(\d+)\.(\d+)", r"\1", text)
    # removes hyperlinks
    text = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "", text)
    # Truncating the content after 1280 characters
    # 1280 = 128 (seq length) * 10((assumed avg. word size) 8 + (spaces on both sides) 2 = 10))
    # Note: our model uses sequences of length 128
    text = text[:1280]
    return str(text)

def create_tc_data(data_name, base_location='data',mode = 'train'):
    """
    creates ordered dataset for text classification with a maximum of 115,000 sequences
    and 7,600 sequences from each individual dataset for train and test data respectively
    i.e.,the size of the smallest training and test sets
    """

    dataset = {'labels': [], 'content': []}
    max_samples = 115000 if mode == 'train' else 7600
    label_to_class = dict()

    if data_name == 'yelp':
        df = pd.read_csv(os.path.join(base_location, 'yelp_review_full_csv', mode+'.csv'), 
                            header=None, names=['labels', 'content'])
        
        df.dropna(subset=['content'], inplace=True)
        df.loc[:, 'content'] = df.content.swifter.apply(preprocess)
        # filter rows with length greater than 20 (2 words including spaces on average)
        df.drop(df[df['content'].map(len) < 20].index, inplace=True)
        # shuffle and sample 
        df = df.sample(n = max_samples)
        
        
        dataset['labels'].extend(list(df.labels[:max_samples]))
        dataset['content'].extend(list(df.content[:max_samples]))

    elif data_name == 'amazon':
        df = pd.read_csv(os.path.join(base_location, 'amazon_review_full_csv', mode+'.csv'), 
                            header=None, names=['labels','title','content'])
        df.dropna(subset=['content'], inplace=True)
        # df.dropna(subset=['title'], inplace=True)
        df.loc[:, 'content'] = df.content.swifter.apply(preprocess)
        # filter rows with length greater than 20 (2 words including spaces on average)
        df.drop(df[df['content'].map(len) < 20].index, inplace=True)
        # shuffle and sample 
        df = df.sample(n = max_samples)

        dataset['labels'].extend(list(df.labels[:max_samples]))
        # dataset['content'].extend( [title + "[SEP]"+ content for title,content in zip(list(df.title[:max_samples]),list(df.content[:max_samples]))])
        dataset['content'].extend(list(df.content[:max_samples]))

    elif data_name == 'yahoo':
        df = pd.read_csv(os.path.join(base_location, 'yahoo_answers_csv', mode+'.csv'), 
                            header=None, names=['labels', 'title', 'content', 'answer'])
        df.dropna(subset=['content'], inplace=True)
        # df.dropna(subset=['title'], inplace=True)
        df.dropna(subset=['answer'], inplace=True)
        df.loc[:, 'content'] = df.content.swifter.apply(preprocess)
        # filter rows with length greater than 20 (2 words including spaces on average)
        df.drop(df[df['content'].map(len) < 20].index, inplace=True)
        # shuffle and sample 
        df = df.sample(n = max_samples)
        dataset['labels'].extend(list(df.labels[:max_samples]))
        # dataset['content'].extend( [title + "[SEP]"+ content + "[SEP]" +answer for title,content,answer in zip(list(df.title[:max_samples]),list(df.content[:max_samples]),list(df.answer[:max_samples]))])
        dataset['content'].extend( [content + "[SEP]" +answer for content,answer in zip(list(df.content[:max_samples]),list(df.answer[:max_samples]))])

    elif data_name == 'dbpedia':
        df = pd.read_csv(os.path.join(base_location, 'dbpedia_csv', mode+'.csv'), 
                            header=None, names=['labels','title','content'])

        df.dropna(subset=['content'], inplace=True)
        # df.dropna(subset=['title'], inplace=True)
        df.loc[:, 'content'] = df.content.swifter.apply(preprocess)
        # filter rows with length greater than 20 (2 words including spaces on average)
        df.drop(df[df['content'].map(len) < 20].index, inplace=True)
        # shuffle and sample 
        df = df.sample(n = max_samples)

        dataset['labels'].extend(list(df.labels[:max_samples]))
        # dataset['content'].extend( [title + "[SEP]"+ content for title,content in zip(list(df.title[:max_samples]),list(df.content[:max_samples]))])
        dataset['content'].extend(list(df.content[:max_samples]))
    
    else:
        df = pd.read_csv(os.path.join(base_location, 'ag_news_csv', mode+'.csv'), 
                            header=None, names=['labels','title','content'])
        df.dropna(subset=['content'], inplace=True)
        # df.dropna(subset=['title'], inplace=True)
        df.loc[:, 'content'] = df.content.swifter.apply(preprocess)
        # filter rows with length greater than 20 (2 words including spaces on average)
        df.drop(df[df['content'].map(len) < 20].index, inplace=True)
        # shuffle and sample 
        df = df.sample(n = max_samples)
        dataset['labels'].extend(list(df.labels[:max_samples]))
        # dataset['content'].extend( [title + "[SEP]"+ content for title,content in zip(list(df.title[:max_samples]),list(df.content[:max_samples]))])
        dataset['content'].extend(list(df.content[:max_samples]))

    return dataset['labels'],dataset['content']
