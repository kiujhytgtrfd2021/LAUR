from __future__ import division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models


from torch.nn import CrossEntropyLoss, MSELoss
import sys
from LAUR_model.networks import BertModel
from LAUR_model.config_bert import BertConfig
from transformers import BertModel as pretrainmodel
sys.path.append('../')
from WikiSQL.sqlova.sqlova.model.nl2sql.wikisql_models import FT_s2s_1

class LAUR_model(nn.Module):
    def __init__(self, make_model=True):
        super(BertClassifier_LAUR, self).__init__()
        if make_model:
            self.make_model()


    def make_model(self):
        """Creates the model."""
        # Get the model from LAUR
        # self.bert = BertModel.from_pretrained(
        #     "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        #     num_labels = 5, # The number of output labels--2 for binary classification.
        #                     # You can increase this for multi-class tasks.   
        #     output_attentions = False, # Whether the model returns attentions weights.
        #     output_hidden_states = True, # Whether the model returns all hidden-states.
        #     bayes = True,
        #     mult = True,
        #     )
        self.bert = BertModel(BertConfig(output_hidden_states=True,bayes = True,mult = True,))
        # Get the transformers BERT pretrain model
        bert_pretrain = pretrainmodel.from_pretrained(
            "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = 5, # The number of output labels--2 for binary classification.
                            # You can increase this for multi-class tasks.   
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = True, # Whether the model returns all hidden-states.
            )

        # Import the linear parameter in bet to baylinear of LAUR
        model_dict = self.bert.state_dict()
        pretrain_dict = bert_pretrain.state_dict()
        state_dict = {}
        for n,p in pretrain_dict.items():
            if 'embed' in n and 'position_ids' not in n or 'LayerNorm' in n or 'bias' in n:
                state_dict[n] = p
            elif 'weight' in n:
                state_dict[n+'_mu'] = p
            else :
                print(n)
        model_dict.update(state_dict)
        self.bert.load_state_dict(model_dict)

        # freeze Embedding and LayerNorm
        for n,param in enumerate(self.bert.named_parameters()):
            if n<3:
                param[1].requires_grad = False
            if 'LayerNorm' in param[0]:   #'bias' in param[0] or 
                param[1].requires_grad = False
        

        self.datasets, self.classifiers, self.num_class = [], nn.ModuleList(),[]
        self.classifiers_data_index = []
        self.dropout = nn.Dropout(0.1)
        self.classifier = None
        self.num_outputs = None

    def add_dataset(self, dataset, num_outputs=0,use_class = False):
        """Adds a new dataset to the classifier."""
        if len(self.num_class) == 0:
            self.classifiers_data_index = []
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            
            self.bert.encoder.add_dataset(dataset)
            if use_class:
                self.num_class.append(num_outputs)
                self.classifiers_data_index.append(dataset)
                self.classifiers.append(nn.Linear(768, num_outputs))
            if dataset == "SQuAD":
                self.qa_outputs = nn.Linear(768, 2 )
            if dataset == "SRL":
                self.qa_SRL_outputs = nn.Linear(768, 2 )
            if dataset == "WikiSQL":
                # some constants
                agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
                cond_ops = ['=', '>', '<', 'OP']  # do not know why 'OP' required. Hence,
                self.SQL_outputs = FT_s2s_1(768,100,2,0.3,270,len(cond_ops),len(agg_ops))
                

    def set_dataset(self, dataset,use_class = False):
        """Change the active classifier."""
        assert dataset in self.datasets
        self.bert.encoder.set_dataset(dataset)
        if use_class:
            self.classifier = self.classifiers[self.classifiers_data_index.index(dataset)]
            self.num_outputs = self.num_class[self.classifiers_data_index.index(dataset)]
        if dataset == "SQuAD":
            self.qa_SQuAD = True
            self.qa_SRL = False
        if dataset == "SRL":
            self.qa_SQuAD = False
            self.qa_SRL = True


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        start_positions=None,
        end_positions=None,
        sample = None,
        use_class = False,
        QA = False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            sample = sample,
        )

        if use_class:
            pooled_output = outputs[1]

            pooled_output = self.dropout(pooled_output)
        
            logits = self.classifier(pooled_output)

            outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_outputs), labels.view(-1))
                outputs = (loss,) + outputs
        
        if QA:
            sequence_output = outputs[0]

            if self.qa_SQuAD:
                logits = self.qa_outputs(sequence_output)
            if self.qa_SRL:
                logits = self.qa_SRL_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            outputs = (start_logits, end_logits,) + outputs[2:]
            if start_positions is not None and end_positions is not None:
                # If we are on multi-GPU, split add a dimension
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                if len(end_positions.size()) > 1:
                    end_positions = end_positions.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1)
                start_positions.clamp_(0, ignored_index)
                end_positions.clamp_(0, ignored_index)

                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2
                outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class LAUR_wo_PRAM(nn.Module):
    def __init__(self, make_model=True):
        super(BertClassifier_UCL, self).__init__()
        if make_model:
            self.make_model()


    def make_model(self):
        """Creates the model."""
        # Get the model from LAUR
        # self.bert = BertModel.from_pretrained(
        #     "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        #     num_labels = 5, # The number of output labels--2 for binary classification.
        #                     # You can increase this for multi-class tasks.   
        #     output_attentions = False, # Whether the model returns attentions weights.
        #     output_hidden_states = False, # Whether the model returns all hidden-states.
        #     bayes = True,
        #     mult = False
        #     )
        self.bert = BertModel(BertConfig(output_hidden_states=True,bayes = True,mult = False,))
        # Get the transformers BERT pretrain model
        bert_pretrain = pretrainmodel.from_pretrained(
            "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = 5, # The number of output labels--2 for binary classification.
                            # You can increase this for multi-class tasks.   
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
            )

        # Import the linear parameter in bet to baylinear of LAUR
        model_dict = self.bert.state_dict()
        pretrain_dict = bert_pretrain.state_dict()
        state_dict = {}
        for n,p in pretrain_dict.items():
            if 'embed' in n and 'position_ids' not in n or 'LayerNorm' in n or 'bias' in n:
                state_dict[n] = p
            elif 'weight' in n:
                state_dict[n+'_mu'] = p
            else :
                print(n)
        model_dict.update(state_dict)
        self.bert.load_state_dict(model_dict)

        # freeze Embedding and LayerNorm
        for n,param in enumerate(self.bert.named_parameters()):
            if n<3:
                param[1].requires_grad = False
            if 'LayerNorm' in param[0]:   #'bias' in param[0] or 
                param[1].requires_grad = False
        

        self.datasets, self.classifiers, self.num_class = [], nn.ModuleList(),[]
        self.dropout = nn.Dropout(0.1)
        self.classifier = None
        self.num_outputs = None

    def add_dataset(self, dataset, num_outputs=0,use_class = False):
        """Adds a new dataset to the classifier."""
        if len(self.num_class) == 0:
            self.classifiers_data_index = []
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            
            # self.bert.encoder.add_dataset(dataset)
            if use_class:
                self.num_class.append(num_outputs)
                self.classifiers_data_index.append(dataset)
                self.classifiers.append(nn.Linear(768, num_outputs))
            if dataset == "SQuAD":
                self.qa_outputs = nn.Linear(768, 2 )
            if dataset == "SRL":
                self.qa_SRL_outputs = nn.Linear(768, 2 )
            if dataset == "WikiSQL":
                # some constants
                agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
                cond_ops = ['=', '>', '<', 'OP']  # do not know why 'OP' required. Hence,
                self.SQL_outputs = FT_s2s_1(768,100,2,0.3,270,len(cond_ops),len(agg_ops))
                

    def set_dataset(self, dataset,use_class = False):
        """Change the active classifier."""
        assert dataset in self.datasets
        # self.bert.encoder.set_dataset(dataset)
        if use_class:
            self.classifier = self.classifiers[self.classifiers_data_index.index(dataset)]
            self.num_outputs = self.num_class[self.classifiers_data_index.index(dataset)]
        if dataset == "SQuAD":
            self.qa_SQuAD = True
            self.qa_SRL = False
        if dataset == "SRL":
            self.qa_SQuAD = False
            self.qa_SRL = True


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        start_positions=None,
        end_positions=None,
        sample = None,
        use_class = False,
        QA = False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            sample = sample,
        )

        if use_class:
            pooled_output = outputs[1]

            pooled_output = self.dropout(pooled_output)
        
            logits = self.classifier(pooled_output)

            outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_outputs), labels.view(-1))
                outputs = (loss,) + outputs
        
        if QA:
            sequence_output = outputs[0]

            if self.qa_SQuAD:
                logits = self.qa_outputs(sequence_output)
            if self.qa_SRL:
                logits = self.qa_SRL_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            outputs = (start_logits, end_logits,) + outputs[2:]
            if start_positions is not None and end_positions is not None:
                # If we are on multi-GPU, split add a dimension
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                if len(end_positions.size()) > 1:
                    end_positions = end_positions.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1)
                start_positions.clamp_(0, ignored_index)
                end_positions.clamp_(0, ignored_index)

                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2
                outputs = (total_loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)





class BertClassifier_base(nn.Module):
    def __init__(self, make_model=True):
        super(BertClassifier_base, self).__init__()

        if make_model:
            self.make_model()


    def make_model(self):
        """Creates the model."""
        # Get the pretrained model.
        self.bert = pretrainmodel.from_pretrained(
            "data/bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = 2, # The number of output labels--2 for binary classification.
                            # You can increase this for multi-class tasks.   
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = True, # Whether the model returns all hidden-states.
            )


        # self.datasets, self.classifiers = [], nn.ModuleList()

        
        # self.dropout = nn.Dropout(0.1)

        # self.classifier = None
        self.datasets, self.classifiers, self.num_class = [], nn.ModuleList(),[]
        self.dropout = nn.Dropout(0.1)
        self.classifier = None
        self.num_outputs = None

    def add_dataset(self, dataset, num_outputs=0,use_class = False):
        """Adds a new dataset to the classifier."""
        if len(self.num_class) == 0:
            self.classifiers_data_index = []
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            
            # self.bert.encoder.add_dataset(dataset)
            if use_class:
                self.num_class.append(num_outputs)
                self.classifiers_data_index.append(dataset)
                self.classifiers.append(nn.Linear(768, num_outputs))
            if dataset == "SQuAD":
                self.qa_outputs = nn.Linear(768, 2 )
            if dataset == "SRL":
                self.qa_SRL_outputs = nn.Linear(768, 2 )
            if dataset == "WikiSQL":
                # some constants
                agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
                cond_ops = ['=', '>', '<', 'OP']  # do not know why 'OP' required. Hence,
                self.SQL_outputs = FT_s2s_1(768,100,2,0.3,270,len(cond_ops),len(agg_ops))
                

    def set_dataset(self, dataset,use_class = False):
        """Change the active classifier."""
        assert dataset in self.datasets
        # self.bert.encoder.set_dataset(dataset)
        if use_class:
            self.classifier = self.classifiers[self.classifiers_data_index.index(dataset)]
            self.num_outputs = self.num_class[self.classifiers_data_index.index(dataset)]
        if dataset == "SQuAD":
            self.qa_SQuAD = True
            self.qa_SRL = False
        if dataset == "SRL":
            self.qa_SQuAD = False
            self.qa_SRL = True


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        start_positions=None,
        end_positions=None,
        sample = None,
        use_class = False,
        QA = False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if use_class:
            pooled_output = outputs[1]

            pooled_output = self.dropout(pooled_output)
        
            logits = self.classifier(pooled_output)

            outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_outputs), labels.view(-1))
                outputs = (loss,) + outputs
        
        if QA:
            sequence_output = outputs[0]

            if self.qa_SQuAD:
                logits = self.qa_outputs(sequence_output)
            if self.qa_SRL:
                logits = self.qa_SRL_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            outputs = (start_logits, end_logits,) + outputs[2:]
            if start_positions is not None and end_positions is not None:
                # If we are on multi-GPU, split add a dimension
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                if len(end_positions.size()) > 1:
                    end_positions = end_positions.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1)
                start_positions.clamp_(0, ignored_index)
                end_positions.clamp_(0, ignored_index)

                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2
                outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class Bert_PRAM_freezeBERT(nn.Module):
    def __init__(self, make_model=True):
        super(Bert_PRAM_freezeBERT, self).__init__()

        if make_model:
            self.make_model()


    def make_model(self):
        """Creates the model."""
        # Get the pretrained model.
        # self.bert = pretrainmodel.from_pretrained(
        #     "data/bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        #     num_labels = 2, # The number of output labels--2 for binary classification.
        #                     # You can increase this for multi-class tasks.   
        #     output_attentions = False, # Whether the model returns attentions weights.
        #     output_hidden_states = True, # Whether the model returns all hidden-states.
        #     )
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = 5, # The number of output labels--2 for binary classification.
                            # You can increase this for multi-class tasks.   
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = True, # Whether the model returns all hidden-states.
            bayes = False,
            mult = True
            )


        # self.datasets, self.classifiers = [], nn.ModuleList()

        for n,param in self.bert.named_parameters():
            if 'mult' in n:
                param.requires_grad = True
            else:
                param.requires_grad = False
        # self.dropout = nn.Dropout(0.1)

        # self.classifier = None
        self.datasets, self.classifiers, self.num_class = [], nn.ModuleList(),[]
        self.dropout = nn.Dropout(0.1)
        self.classifier = None
        self.num_outputs = None

    def add_dataset(self, dataset, num_outputs=0,use_class = False):
        """Adds a new dataset to the classifier."""
        if len(self.num_class) == 0:
            self.classifiers_data_index = []
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            
            self.bert.encoder.add_dataset(dataset)
            if use_class:
                self.num_class.append(num_outputs)
                self.classifiers_data_index.append(dataset)
                self.classifiers.append(nn.Linear(768, num_outputs))
            if dataset == "SQuAD":
                self.qa_outputs = nn.Linear(768, 2 )
            if dataset == "SRL":
                self.qa_SRL_outputs = nn.Linear(768, 2 )
            if dataset == "WikiSQL":
                # some constants
                agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
                cond_ops = ['=', '>', '<', 'OP']  # do not know why 'OP' required. Hence,
                self.SQL_outputs = FT_s2s_1(768,100,2,0.3,270,len(cond_ops),len(agg_ops))
                

    def set_dataset(self, dataset,use_class = False):
        """Change the active classifier."""
        assert dataset in self.datasets
        self.bert.encoder.set_dataset(dataset)
        if use_class:
            self.classifier = self.classifiers[self.classifiers_data_index.index(dataset)]
            self.num_outputs = self.num_class[self.classifiers_data_index.index(dataset)]
        if dataset == "SQuAD":
            self.qa_SQuAD = True
            self.qa_SRL = False
        if dataset == "SRL":
            self.qa_SQuAD = False
            self.qa_SRL = True


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        start_positions=None,
        end_positions=None,
        sample = None,
        use_class = False,
        QA = False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if use_class:
            pooled_output = outputs[1]

            pooled_output = self.dropout(pooled_output)
        
            logits = self.classifier(pooled_output)

            outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_outputs), labels.view(-1))
                outputs = (loss,) + outputs
        
        if QA:
            sequence_output = outputs[0]

            if self.qa_SQuAD:
                logits = self.qa_outputs(sequence_output)
            if self.qa_SRL:
                logits = self.qa_SRL_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            outputs = (start_logits, end_logits,) + outputs[2:]
            if start_positions is not None and end_positions is not None:
                # If we are on multi-GPU, split add a dimension
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                if len(end_positions.size()) > 1:
                    end_positions = end_positions.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1)
                start_positions.clamp_(0, ignored_index)
                end_positions.clamp_(0, ignored_index)

                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2
                outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

