# -*- coding: utf-8 -*-

import json
import os, sys, ast, re
from tqdm import tqdm
import requests
import time
import argparse

import numpy as np
import pandas as pd
import collections
from collections import Counter

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix

from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers import DistilBertTokenizer, BertTokenizer
from transformers import BertModel


## ARGUMENTS AND LOADERS
def argparser():
    """Creating arparse.ArgumentParser and returning arguments
    """
    argparser = argparse.ArgumentParser(description='Use BioBert to predict Litcovid categories.', formatter_class=argparse.RawTextHelpFormatter)
    argparser.add_argument('--mode', choices=['train', 'test', 'predict'], required=True, help="How to use script.")
    argparser.add_argument('--model_path', '-m', type=str, help="Path to load from / save to the model.")
    argparser.add_argument('--data_path', '-d', type=str, required=True, help="Path to load training/predict data from.")
    argparser.add_argument('--predict_path', '-p', type=str, default="predictions.json", help="Path to save predict data to.")
    argparser.add_argument('--feature', '-f', type=str, choices=['title', 'abstract', 'both'], default="both", help="What to use as prediction feature.")
    # Bert parameters
    argparser.add_argument('--maxlen', type=int, default = 200, help="Max length of features used - tokenizer parameter.")
    argparser.add_argument('--batch_size', type=int, default = 32, help="Batch size for model training.")
    argparser.add_argument('--epochs', type=int, default = 3, help="Number of epochs for model training.")
    argparser.add_argument('--hidden_size', type=int, default = 128, help="GRU hidden size")
    argparser.add_argument('--embed_size', type=int, default = 128, help="Embedding size")
    argparser.add_argument('--lr_bio', type=float, default = 3e-06, help="Learning rate")
    argparser.add_argument('--lr_deci', type=float, default = 5e-03, help="Learning rate")
    argparser.add_argument('--freezebio', action='store_true', help="Freeze when training")
    argparser.add_argument('--pretrained_bert', type=str, default='dmis-lab/biobert-base-cased-v1.1', help="Pretrained model to load weights from for Bert.")
    argparser.add_argument('--decision_threshold', '-th', type=float, default=2e-1, help="Decision threshold for prediction.")

    args = argparser.parse_args()
    if (args.mode in ['test', 'predict']) and (args.model_path is None):
        raise AttributeError("No model path given for prediction.")
    elif (args.mode == 'train') and (args.model_path is None):
        args.model_path = 'biobert_model'

    return args

def load_model(path:str, name_pretrained, nb_classes, device):
    """Load model from previous training
    """
    bert_model = BioBertClassifier(name_pretrained, nb_classes, device)
    bert_model.load_state_dict(torch.load(path))
    bert_model.eval()
    return bert_model

def save_model(bert_model, path:str):
    """Save model after training
    """
    torch.save(bert_model.state_dict(), path)

def load_data(path:str, with_labels:bool=True, feature:str='both'):
    """Warning: expecting 

    Input:
    --------
    path: str
        path to load data from - expecting JSON data with 'title', 'abstract' and 'topics' (if not predict) columns

    with_labels: bool
        default True (train/test); rows with empty labels are removed

    feature: str
        'title', 'abstract' or 'both' (from argparse)

    Returns:
    --------
    X: array
        list of str (sentences)
    
    y: array
        list of one_hot_encoding of labels
    
    df: pd.DataFrame
        loaded data
    """
    with open(path) as f:
        json_file = json.load(f)
    df = pd.DataFrame(json_file)

    if with_labels:
        df = df[~df.topics.isna()]
        df[df.topics.apply(lambda x: 'NONE' in x)]

    if feature == 'both':
        df['abstract'] = df.abstract.fillna('')
        df['x_concat'] = (df['title'] + ' ' + df.abstract).apply(lambda x: x.strip())
        X = df.x_concat.tolist()
    else:
        df = df[~df[feature].isna()]
        X = df[feature].tolist()
    
    if with_labels:
        y = df.topics
    else:
        y = [[]*len(X)]
    return X,y, df

def save_predictions(df, path):
    """
    """
    result = df.to_json(orient="records")
    parsed = json.loads(result)
    with open(path, mode='w') as f:
        f.write(json.dumps(parsed, indent=2))


## DATA FUNCTIONS
def bert_text_to_ids(sentence, tokenizer):
    return torch.tensor(tokenizer.encode(sentence, add_special_tokens=True))

def prepare_texts(texts:list, labels:list, 
          maxlen=530, 
          tokenizer=BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1'), 
          device=torch.device('cuda')):
    """Ajout de padding pour que tous les textes aient la même longueur et indexation des tokens grace au tokenizer
    
    Retourne les données au format tensor.
    """
    X = torch.LongTensor(len(texts), maxlen).fill_(tokenizer.pad_token_id)
    for i, text in enumerate(texts):
        indexed_tokens = bert_text_to_ids(text, tokenizer)
        length = min([maxlen, len(indexed_tokens)])
        X[i,:length] = indexed_tokens[:length]
    
    Y = torch.tensor(labels).long()
    return X.to(device), Y.to(device)

def create_loader(texts, labels, batch_size:int=32, shuffle:bool=False, **kwargs):
    """
    kwargs: maxlen, tokenizer, device
    """
    X_train, Y_train = prepare_texts(texts, labels, **kwargs) 
    train_set = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    return train_loader


## MODEL FUNCTIONS
def CrossEntropyLossOneHot(y_score, y):
    """CrossEntropy pour la multi classification
    """
    log_y_score = torch.log(y_score)
    cost = -torch.sum(y * log_y_score)
    return cost

def perf(model, loader, device, nb_classes=8, seuil:float=2e-1):
    """Evaluation en multi-label, le seuil désigne le niveau a partir duquel on considère une classe comme 'prédite'
    """
    criterion = CrossEntropyLossOneHot
    model.eval()
    total_loss = num = num_comp = correct = 0
    total_pred = true_y = None # concaténation des batchs
    for x, y in loader:
        with torch.no_grad():
            y_scores = model(x)
            loss = criterion(y_scores, y)
            y_pred = multi_hot(y_scores, device, seuil = seuil)
            if total_pred == None: # concat
                total_pred = y_pred
                true_y = y
            else:
                total_pred = torch.cat((total_pred,y_pred), dim=0)
                true_y = torch.cat((true_y,y), dim=0)
            correct += torch.sum(y_pred == y).item()
            total_loss += loss.item()
            num_comp += len(y) * nb_classes
            num += len(y)

    f_score = f1_score(true_y.cpu(), total_pred.cpu(), average='micro')
    accu_score = accuracy_score(true_y.cpu(), total_pred.cpu())
    return total_loss / num, correct / num_comp, accu_score, f_score

def multi_hot(y_pred, device, seuil=2e-1, to_gpu:bool=True):
    """binarize prediction using threshold
    """
    res = torch.zeros(y_pred.shape[0], y_pred.shape[1])#.int()
    res[torch.arange(y_pred.shape[0]), torch.argmax(y_pred, dim=1)] = 1
    for i in range(y_pred.shape[0]):
        for j in range(y_pred.shape[1]):
            if y_pred[i,j] > seuil:
                res[i,j] = 1
    if to_gpu:
        return res.to(device)
    return res

## MODEL
class BioBertClassifier(nn.Module):
    def __init__(self, name_pretrained, nb_classes, device):
        super().__init__()
        self.bert = BertModel.from_pretrained(name_pretrained)
        self.decision = nn.Linear(self.bert.config.hidden_size, nb_classes)
        self.to(device)

    def forward(self, x, tokenizer_pad_token_id = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1').pad_token_id):
        output = self.bert(x, attention_mask = (x != tokenizer_pad_token_id).long())
        return torch.softmax(self.decision(torch.max(output[0], 1)[0]),1) # log => positive qty


def fit(model, epochs, train_loader, valid_loader, device,
            lrbio:float=1e-4, lrdeci:float=1e-02, freezebio:bool=True, 
            seuil:float=2e-1, nb_classes:int=8) -> dict:
    """Training BioBert Model - with/without frozen layers - for MultiClass Classification
    """
    criterion = CrossEntropyLossOneHot

    #freeze biobert layers
    if freezebio :
      for parameter in model.parameters():
          parameter.requires_grad = False
      for name, param in model.named_parameters():
          if 'decision' in name:
              param.requires_grad = True
      optimizer = optim.Adam(model.parameters(), lr=lrdeci)

    #different lr pour bio et decision
    else :
        my_list = ['decision.weight', 'decision.bias']
        params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in my_list, model.named_parameters()))))
        optimizer = optim.Adam([{'params': base_params}, {'params': params, 'lr': lrdeci}], lr=lrbio)
      
    history  = {
        'train_losses' : [],
        'val_losses' : [],
        'preci' : [],
        'all_good_preci' : [],
        'f_micro' : [],
    }

    for epoch in range(epochs):
        model.train()
        total_loss = num = 0
        for x, y in tqdm(train_loader):
            optimizer.zero_grad()
            y_scores = model(x)
            loss = criterion(y_scores, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num += len(y)

        valloss, preci, accu_score, f_mesure = perf(model, valid_loader, device, nb_classes = nb_classes, seuil = seuil)
        history['train_losses'].append(total_loss / num)
        history['val_losses'].append(valloss)
        history['preci'].append(preci)
        history['all_good_preci'].append(accu_score)
        history['f_micro'].append(f_mesure)

        print(f'\nepoch : {epoch}\t - train loss : {total_loss / num}\t - val loss : {valloss}\t - precision :{preci}\t - all good precision : {accu_score}\t - f-score micro : {f_mesure}')

    return history


def predict(model, loader, device, threshold:float=2e-1):
    total_pred = true_y = None
    for x, y in loader:
        with torch.no_grad():
            y_scores = model(x)
            y_pred = multi_hot(y_scores, device, seuil=threshold)
            if total_pred == None: # concat
                total_pred = y_pred
                true_y = y
            else:
                total_pred = torch.cat((total_pred,y_pred), dim=0)
                true_y = torch.cat((true_y,y), dim=0)
    return true_y.cpu(), total_pred.cpu()



if __name__ == '__main__':
    args = argparser()
    device = torch.device('cuda') # Necessary considering training duration

    # Binarizing classes
    classes = [ 'Case Report', 'Diagnosis', 'Epidemic Forecasting', 'General Info',
                'Mechanism', 'Prevention', 'Transmission', 'Treatment'] # Fixed for litcovid
    mb = MultiLabelBinarizer()
    mb.fit([classes])

    X,y, df = load_data(args.data_path, with_labels=(args.mode != 'predict'), feature = args.feature)
    y_bin = mb.transform(y)

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert)

    if args.mode == 'train':
        # Data
        train_texts, valid_texts, y_train, y_valid, train_labels, valid_labels = train_test_split(X, y, y_bin, test_size=0.2, random_state=42)
        train_loader = create_loader(train_texts,train_labels, batch_size=args.batch_size, shuffle=True, 
                        maxlen=args.maxlen, tokenizer=tokenizer, device=device) 
        valid_loader = create_loader(valid_texts,valid_labels, batch_size=args.batch_size,
                        maxlen=args.maxlen, tokenizer=tokenizer, device=device) 
        # Train Model
        bert_model = BioBertClassifier(args.pretrained_bert, len(classes), device)
        _ = fit(bert_model, args.epochs, train_loader, valid_loader, device,
                    lrbio=args.lr_bio, lrdeci=args.lr_deci, freezebio=args.freezebio, nb_classes=len(classes))
        save_model(bert_model, args.model_path)
        
        y_valid, bert_predict = predict(bert_model, valid_loader, device)
        print(classification_report(y_valid, bert_predict, target_names=classes))
    else:
        bert_model = load_model(args.model_path, args.pretrained_bert, len(classes), device)
        data_loader = create_loader(X, y_bin, batch_size=args.batch_size, shuffle=(args.mode == 'train'), 
                        maxlen=args.maxlen, tokenizer=tokenizer, device=device) 
        y_valid, bert_predict = predict(bert_model, data_loader, device)

        if args.mode == 'test':
            test_loss, test_preci, accu_score, f_score = perf(bert_model, data_loader, device=device, 
                                                    seuil=args.decision_threshold, nb_classes=len(classes))
            print(f'test loss : {test_loss}\t - precision :{test_preci}\t - all good precision : {accu_score}\t - f-score micro : {f_score}')
            print(classification_report(y_valid, bert_predict, target_names=classes))
        
        if args.mode == 'predict':
            df['topic_predictions'] = mb.inverse_transform(bert_predict).astype(list)
            save_predictions(df, args.predict_path)

