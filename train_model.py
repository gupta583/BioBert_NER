import torch
import pandas as pd
import torch.nn as nn
from pytorch_pretrained_bert import BertModel,BertConfig, BertForPreTraining
import tensorflow as tf
import re
import torch
import numpy as np
from tqdm import tqdm, trange
import os
import csv
from pytorch_pretrained_bert import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForTokenClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
import itertools
MAX_LEN = 75
bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ner_bio(nn.Module):
    def __init__(self,vocab_len,config,state_dict):
        super().__init__()
        self.bert = BertModel(config)
        self.bert.load_state_dict(state_dict)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size,vocab_len)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self,input_ids,attention_mask):
        encoded_layer,_ = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        enc = encoded_layer[-1]
        output = self.drop(enc)
        output = self.out(output)
        
        return output, output.argmax(-1)

def train_epoch(model,data_loader,loss_fn,optimizer,device,scheduler):
    model = model.train()
    losses = []
    correct_predictions = 0
    for step,batch in enumerate(data_loader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        outputs,y_hat = model(b_input_ids,b_input_mask)
        
        _,preds = torch.max(outputs,dim=2)
        outputs = outputs.view(-1,outputs.shape[-1])
        b_labels_shaped = b_labels.view(-1)
        loss = loss_fn(outputs,b_labels_shaped)
        correct_predictions += torch.sum(preds == b_labels)
        losses.append(loss.item())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    return correct_predictions.double()/len(data_loader) , np.mean(losses)

def model_eval(model,data_loader,loss_fn,device):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for step,batch in enumerate(data_loader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            outputs,y_hat = model(b_input_ids,b_input_mask)

            _,preds = torch.max(outputs,dim=2)
            outputs = outputs.view(-1,outputs.shape[-1])
            b_labels_shaped = b_labels.view(-1)
            loss = loss_fn(outputs,b_labels_shaped)
            correct_predictions += torch.sum(preds == b_labels)
            losses.append(loss.item())


    return correct_predictions.double()/len(data_loader) , np.mean(losses)


def sentence_retriver(path):

    with open(path) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        sentences = []
        tags = []
        sent = []
        tag = []
        for row in reader:
            if len(row) == 0:
                if len(sent) != len(tag):
                    print('Error')
                    break
                sentences.append(sent)
                tags.append(tag)
                sent = []
                tag = []
            else:
                sent.append(row[0])
                tag.append(row[1])

    return sentences, tags

def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        # Tokenizing the words
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)


        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels

rootdir = './BioNLP'
sentences = []
tags = []
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file == 'train.tsv':
#             print(os.path.join(subdir, file))
            path_ = os.path.join(subdir, file)
            sent, tag =sentence_retriver(path_)
            sentences.extend(sent)
            tags.extend(tag)

#sentences = sentences[0:20000]
#tags = tags[0:20000]

tag_values = list(set(itertools.chain.from_iterable(tags)))
tag_values.append("PAD")
tag2idx = {t: i for i,t in enumerate(tag_values) }

df_tags = pd.DataFrame({'tags':tag_values})
# df_tags.head()
df_tags.to_csv('tags_large.csv',index=False)
#df_ = pd.read_csv('tags_small.csv')
#df_.head()
vocab_len = len(tag_values)
tokenizer = BertTokenizer(vocab_file='biobert_v1.0_pubmed_pmc/vocab.txt', do_lower_case=False)
tokenized_texts_and_labels = [
    tokenize_and_preserve_labels(sent, labs)
    for sent, labs in zip(sentences, tags)
]

tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", value=0.0,
                          truncating="post", padding="post")

tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post",
                     dtype="long", truncating="post")
attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]
tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                            random_state=2018, test_size=0.1)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)

tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)

train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

config = BertConfig.from_json_file('biobert_v1.0_pubmed_pmc/bert_config.json')
tmp_d = torch.load('weights/pytorch_weight',map_location=device)
from collections import OrderedDict
state_dict = OrderedDict()

for i in list(tmp_d.keys())[:199]:
    x = i
    if i.find('bert') > -1:
        x = '.'.join(i.split('.')[1:])
    state_dict[x] = tmp_d[i]

model = ner_bio(vocab_len,config,state_dict)
model.to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=3e-5,
    eps=1e-8
)
epochs = 3
max_grad_norm = 1.0

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)

from collections import defaultdict
history = defaultdict(list)
best_accuracy = 0
normalizer = bs*MAX_LEN

for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')
    print('-'*10)
    train_acc,train_loss = train_epoch(model,train_dataloader,loss_fn,optimizer,device,scheduler)
    train_acc = train_acc/normalizer
    print(f'Train loss {train_loss} accuracy {train_acc}')
          


    val_acc,val_loss = model_eval(model,valid_dataloader,loss_fn,device)
    val_acc = val_acc/normalizer
    print(f'val loss {val_loss} accuracy {val_acc}')
    print()
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)


model_save_name = 'BIONER_classifier_large.pt'
path = F"app/{model_save_name}"
torch.save(model.state_dict(), path)



print('Training finished.')
