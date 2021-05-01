from datasets import load_dataset, load_metric
from transformers import AutoTokenizer,AutoConfig, AutoModelForQuestionAnswering
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
from LAUR_model.model import BertClassifier_base
from transformers import AdamW,get_linear_schedule_with_warmup,BertForQuestionAnswering
from tqdm import tqdm
import numpy as np
from WikiSQL.sqlova.sqlova.utils.utils_wikisql import *
from WikiSQL.sqlova.sqlova.model.nl2sql.wikisql_models import *
tokenizer = AutoTokenizer.from_pretrained(
        'bert-base-uncased',
        use_fast=True,
    )
model = BertClassifier_base()
model.add_dataset("SRL")
model.add_dataset("SQuAD")
model.add_dataset("SST",num_outputs=2,use_class=True)
model.add_dataset("WikiSQL")
model.cuda()


def preprocess_function(examples):
    # Tokenize the texts
    args = (
        (examples["sentence"],)
    )
    result = tokenizer(*args, padding="max_length", max_length=128, truncation=True)


    return result
def prepare_train_features(examples):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples[answer_column_name][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

def prepare_inputs(inputs):
    """
    Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
    handling potential state.
    """
    for v in inputs:
        # if isinstance(v, torch.Tensor):
        print(v)
        inputs[k] = v.cuda()


    return inputs



datasets = load_dataset("./SRL/SRL_preprocessing.py", None)
column_names = datasets["train"].column_names
question_column_name = "question" if "question" in column_names else column_names[0]
context_column_name = "context" if "context" in column_names else column_names[1]
answer_column_name = "answers" if "answers" in column_names else column_names[2]

pad_on_right = tokenizer.padding_side == "right"

train_dataset = datasets["train"].map(
        prepare_train_features,
        batched=True,
        num_proc=4,
        remove_columns=column_names,
    )
# print(torch.Tensor(train_dataset["attention_mask"][0]))
attention_mask = torch.LongTensor(train_dataset["attention_mask"])
input_ids = torch.LongTensor(train_dataset['input_ids'])
token_type_ids = torch.LongTensor(train_dataset['token_type_ids'])
end_positions = torch.LongTensor(train_dataset['end_positions'])
start_positions = torch.LongTensor(train_dataset["start_positions"])
# print(attention_mask.size())
SRL_train_data = TensorDataset(input_ids,attention_mask,token_type_ids,start_positions,end_positions)
SRL_train_sampler = RandomSampler(SRL_train_data)
SRL_train_dataloader = DataLoader(
        SRL_train_data,
        batch_size=16,
        sampler=SRL_train_sampler,
    )

datasets = load_dataset("./SQuAD/squad_preprocessing.py", None)
column_names = datasets["train"].column_names
question_column_name = "question" if "question" in column_names else column_names[0]
context_column_name = "context" if "context" in column_names else column_names[1]
answer_column_name = "answers" if "answers" in column_names else column_names[2]

pad_on_right = tokenizer.padding_side == "right"

train_dataset = datasets["train"].map(
        prepare_train_features,
        batched=True,
        num_proc=4,
        remove_columns=column_names,
    )
# print(torch.Tensor(train_dataset["attention_mask"][0]))
attention_mask = torch.LongTensor(train_dataset["attention_mask"])
input_ids = torch.LongTensor(train_dataset['input_ids'])
token_type_ids = torch.LongTensor(train_dataset['token_type_ids'])
end_positions = torch.LongTensor(train_dataset['end_positions'])
start_positions = torch.LongTensor(train_dataset["start_positions"])
# print(attention_mask.size())
SQuAD_train_data = TensorDataset(input_ids,attention_mask,token_type_ids,start_positions,end_positions)
SQuAD_train_sampler = RandomSampler(SQuAD_train_data)
SQuAD_train_dataloader = DataLoader(
        SQuAD_train_data,
        batch_size=16,
        sampler=SQuAD_train_sampler,
    )

datasets = load_dataset(
            "./SST/huggingface_csv_preprocessing.py", data_files={"train": "data/SST/train.csv", "validation": "data/SST/test.csv"}
        )
# print(datasets["train"])
datasets = datasets.map(preprocess_function, batched=True)
train_dataset = datasets["train"]
attention_mask = torch.LongTensor(train_dataset["attention_mask"])
input_ids = torch.LongTensor(train_dataset['input_ids'])
token_type_ids = torch.LongTensor(train_dataset['token_type_ids'])
label = torch.LongTensor(train_dataset['label'])
SST_train_data = TensorDataset(input_ids,attention_mask,token_type_ids,label)
SST_train_sampler = RandomSampler(SST_train_data)
SST_train_dataloader = DataLoader(
        SST_train_data,
        batch_size=16,
        sampler=SST_train_sampler,
    )


def get_data(path_wikisql):
    train_data, train_table, dev_data, dev_table, _, _ = load_wikisql(path_wikisql, False, 32,
                                                                      no_w2i=True, no_hs_tok=True)
    train_loader, dev_loader = get_loader_wikisql(train_data, dev_data, 16, shuffle_train=True)

    return train_data, train_table, dev_data, dev_table, train_loader, dev_loader

train_data, train_table, dev_data, dev_table, WikiSQL_data, dev_loader = get_data("./data/WikiSQL")
sql_vocab = (
        "none", "max", "min", "count", "sum", "average",
        "select", "where", "and",
        "equal", "greater than", "less than",
        "start", "end"
        )
# pals_p_id = []
# rho_id = []
# mu_id = []
no_decay = []
lr=3e-5
for n,param in model.named_parameters():
    print(n)
    # if 'mult' in n:
    #     pals_p_id.append(id(param))
    # if 'rho' in n:
    #     rho_id.append(id(param))
    # if 'weight_mu' in n:
    #     mu_id.append(id(param))
    if 'bias' in n or 'LayerNorm.weight' in n: #BERT not use weight_decay in bias and LN
        no_decay.append(id(param))
bert_param = filter(lambda p: p.requires_grad and id(p) not in no_decay,model.parameters())
bert_param_nodecay = filter(lambda p: p.requires_grad and id(p) in no_decay,model.parameters())
# pals_param = filter(lambda p: p.requires_grad and id(p) in pals_p_id and id(p) not in no_decay ,model.parameters())
# pals_param_nodecay = filter(lambda p: p.requires_grad and id(p) in pals_p_id and id(p) in no_decay ,model.parameters())
# rho_param = filter(lambda p: p.requires_grad and id(p) in rho_id ,model.parameters())
params = [
    {"params":bert_param,"lr":lr,"weight_decay":1e-2},
    {"params":bert_param_nodecay,"lr":lr,"weight_decay":0},
    # {"params":pals_param,"lr":lr*10,"weight_decay":1e-8}, #1e-3
    # {"params":pals_param_nodecay,"lr":lr*10,"weight_decay":0},
    # {"params":rho_param,"lr":lr*100,"weight_decay":1e-8}   #lr*100
]

optimizer = AdamW(params,
            eps = 1e-8, # args.adam_epsilon  - default is 1e-8.
            )
opt_WikiSQL = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.SQL_outputs.parameters()),
                            lr=1e-3, weight_decay=0)

# total_steps = len(train_dataloader) * 2
# # Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = 50000)
# inputs= next(iter(train_dataloader))
probs = [87599,6414,6920,56355]
alpha = 0.4
probs = [p**alpha for p in probs]
tot = sum(probs)
probs = [p/tot for p in probs]

for epochs in range(5):
    for i in tqdm(range(10000)):
        optimizer.zero_grad()
        task_id = np.random.choice(4, p=probs)
        if task_id == 0:
            inputs= next(iter(SQuAD_train_dataloader))
            # inputs = prepare_inputs(inputs)
            inputs = tuple(t.cuda() for t in inputs)
            input_ids,attention_mask,token_type_ids,start_positions,end_positions = inputs
        # print(inputs['attention_mask'])
            model.set_dataset("SQuAD")
            outputs = model(input_ids = input_ids,attention_mask = attention_mask,token_type_ids = token_type_ids,end_positions = end_positions,start_positions = start_positions,QA=True)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            scheduler.step()
        elif task_id == 1:
            inputs= next(iter(SRL_train_dataloader))
            # inputs = prepare_inputs(inputs)
            inputs = tuple(t.cuda() for t in inputs)
            input_ids,attention_mask,token_type_ids,start_positions,end_positions = inputs
        # print(inputs['attention_mask'])
            model.set_dataset("SRL")
            outputs = model(input_ids = input_ids,attention_mask = attention_mask,token_type_ids = token_type_ids,end_positions = end_positions,start_positions = start_positions,QA=True)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            scheduler.step()
        elif task_id == 2:
            inputs= next(iter(SST_train_dataloader))
            # if i == 0:
                # print(len(SST_train_dataloader))
                # print(inputs[0])
            inputs = tuple(t.cuda() for t in inputs)
            input_ids,attention_mask,token_type_ids,label = inputs
            model.set_dataset("SST",use_class=True)
            outputs = model(input_ids = input_ids,attention_mask = attention_mask,token_type_ids = token_type_ids,labels = label,use_class=True)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            scheduler.step()
            # print(inputs)
        elif task_id == 3:
            t= next(iter(WikiSQL_data))
            model.set_dataset("WikiSQL")
            # model_bert = model.bert
            nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, train_table, no_hs_t=True, no_sql_t=True)
            g_sc, g_sa, g_wn, g_wc, g_wo, g_wv = get_g(sql_i)
            # get ground truth where-value index under CoreNLP tokenization scheme. It's done already on trainset.
            g_wvi_corenlp = get_g_wvi_corenlp(t)


            # g_wvi_corenlp = get_g_wvi_corenlp(t)
            all_encoder_layer, pooled_output, tokens, i_nlu, i_hds, i_sql_vocab, \
            l_n, l_hpu, l_hs, l_input, \
            nlu_tt, t_to_tt_idx, tt_to_t_idx \
                = get_bert_output_s2s(model, tokenizer, nlu_t, hds, sql_vocab, 270,sample=None)

            try:
                #
                g_wvi = get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)
            except:
                # Exception happens when where-condition is not found in nlu_tt.
                # In this case, that train example is not used.
                # During test, that example considered as wrongly answered.
                # e.g. train: 32.
                continue


            # Generate g_pnt_idx
            g_pnt_idxs = gen_g_pnt_idx(g_wvi, sql_i, i_hds, i_sql_vocab, col_pool_type="start_tok")
            pnt_start_tok = i_sql_vocab[0][-2][0]
            pnt_end_tok = i_sql_vocab[0][-1][0]
            # check
            # print(array(tokens[0])[g_pnt_idxs[0]])
            wenc_s2s = all_encoder_layer[-1]

            # wemb_h = [B, max_header_number, hS]
            cls_vec = pooled_output

            score = model.SQL_outputs(wenc_s2s, l_input, cls_vec, pnt_start_tok, g_pnt_idxs=g_pnt_idxs)


            # Calculate loss & step
            loss = Loss_s2s(score, g_pnt_idxs)
            loss.backward()
            optimizer.step()
            opt_WikiSQL.step()
            scheduler.step()

    torch.save(model,"MTL_"+str(epochs)+".pt")

