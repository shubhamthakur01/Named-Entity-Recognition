import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd


# if a word is not in your vocabulary use len(vocabulary) as the encoding
class NERDataset(Dataset):
    def __init__(self, df_enc):
        self.df = df_enc

    def __len__(self):
        """ Length of the dataset """
        ### BEGIN SOLUTION
        #L = len(self.df) - 4
        ### END SOLUTION
        return len(self.df) - 4

    def __getitem__(self, idx):
        """ returns x[idx], y[idx] for this dataset
        
        x[idx] should be a numpy array of shape (5,)
        """
        ### BEGIN SOLUTION
        #x = self.df.word[idx:idx+5].to_numpy()
        #print(self.df.label)
        #y = self.df.label[idx+2]
        
        ### END SOLUTION
        #return self.df.word[idx:idx+5].to_numpy(), self.df.label[idx+2]
        #return self.df[idx:idx+5,0], self.df[idx,1]
        return self.df.iloc[idx:idx+5,0].to_numpy(), self.df.iloc[idx+2,1]#.label[idx+2]

def label_encoding(cat_arr):
    
    """ Given a numpy array of strings returns a dictionary with label encodings.

    First take the array of unique values and sort them (as strings). 
    """
   ### BEGIN SOLUTION
    cat_arr[pd.notna(cat_arr)]
    sort_list = np.sort(np.unique(cat_arr[pd.notna(cat_arr)]))
    i = 0
    vocab2index = {}
    for word in sort_list:
        vocab2index[word] = i 
        i += 1
    ### END SOLUTION
    return vocab2index


def dataset_encoding(df, vocab2index, label2index):
    """Apply vocab2index to the word column and label2index to the label column

    Replace columns "word" and "label" with the corresponding encoding.
    If a word is not in the vocabulary give it the index V=(len(vocab2index))
    """
    V = len(vocab2index)
    df_enc = df.copy()
    ### BEGIN SOLUTION
    df_enc["word"] = df_enc["word"].apply(lambda x: vocab2index.get(x,V ))
    df_enc["label"] = df_enc["label"].apply(lambda x: label2index.get(x))
    ### END SOLUTION
    return df_enc


class NERModel(nn.Module):
    def __init__(self, vocab_size, n_class, emb_size=50, seed=3):
        """Initialize an embedding layer and a linear layer
        """
        super(NERModel, self).__init__()
        torch.manual_seed(seed)
        ### BEGIN SOLUTION
        self.word_emb = nn.Embedding(vocab_size, emb_size)
        self.linear = nn.Linear(emb_size*5, n_class)

        ### END SOLUTION
        
    def forward(self, x):
        """Apply the model to x
        
        1. x is a (N,5). Lookup embeddings for x
        2. reshape the embeddings (or concatenate) such that x is N, 5*emb_size 
           .flatten works
        3. Apply a linear layer
        """
        ### BEGIN SOLUTION
        x = self.word_emb(x)
        l,h,w = x.shape
        x =x.reshape(l, h*w)
        x = self.linear(x)
        ### END SOLUTION
        return x
    
def get_optimizer(model, lr = 0.01, wd = 0.0):
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    return optim

def train_model(model, optimizer, train_dl, valid_dl, epochs=10):
    for i in range(epochs):
        train_loss = []
        model.train()
        ### BEGIN SOLUTION
        for x, y in train_dl:
            y_onehot = torch.nn.functional.one_hot(y)
            y_pred = model(x)
            y_hat = torch.softmax(y_pred, dim = 1)
            #print(y_onehot.shape, y_pred.shape)
            L = nn.CrossEntropyLoss()
            loss = L(y_pred.float(), y_onehot.float())
            
            #loss = nn.CrossEntropyLoss(y_onehot,y_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        ### END SOLUTION
        valid_loss, valid_acc = valid_metrics(model, valid_dl)
        print(np.mean(train_loss), valid_loss,valid_acc)
        
        
def valid_metrics(model, valid_dl):
    model.eval()
    ### BEGIN SOLUTION
    losses = []
    y_hats = []
    ys = []
    for x, y in valid_dl:
        y_onehot = torch.nn.functional.one_hot(y) # One hot encoding so that we can calculate crossentropy
        y_pred = model(x) 
        y_hat = torch.softmax(y_pred, dim = 1) # Probability for each class
        L = nn.CrossEntropyLoss()
        loss = L(y_pred.float(), y_onehot.float())
        #loss = nn.CrossEntropyLoss(y_onehot,y_pred)  
        y_hat = np.argmax(y_hat.detach(), axis=1) # Getting back the predicted class
        y_hats.append(y_hat)
        ys.append(y.numpy())
        losses.append(loss.item())
    
    ys = np.concatenate(ys)
    y_hats = np.concatenate(y_hats)
    val_loss = np.mean(losses)#, roc_auc_score(ys, y_hats)
    val_acc = (ys == y_hats).sum()/len(ys)
    ### END SOLUTION
    return val_loss, val_acc

