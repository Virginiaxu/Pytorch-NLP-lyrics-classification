import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import re
torch.manual_seed(1)


class LSTMGenre(nn.Module):
    def __init__(self, model, embedding_dim, hidden_dim, num_cat, num_sentence, limit):
        super(LSTMGenre, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first = True)
        self.sentences = num_sentence
        self.wordLimit = limit
        self.hidden2cat = nn.Linear(hidden_dim*2, num_cat)
        self.hidden = self.init_hidden()
        #attention
        self.attn_combine = nn.Linear(self.hidden_dim*2, self.hidden_dim*2)
        
    def init_hidden(self):
        # size: num_layers, batch_size, hidden_dim
        return (torch.rand(1,self.sentences,self.hidden_dim).cuda(), 
                torch.rand(1,self.sentences,self.hidden_dim).cuda())

    def forward(self, group_sentences): 
        # embeds of size (seq_len, batch_size, input_size=50)
        sentence_in = prepSentences(group_sentences, self.wordLimit).cuda()
        model_out, self.hidden = self.lstm(sentence_in, self.hidden)
        previous_hiddens = model_out[:, :-2, :]
        #compare the final hidden with each previous hidden state to compute a
        #use dot product to compare
        product = torch.bmm(self.hidden[0].transpose(0,1), previous_hiddens.transpose(1,2))
        attn_weights = F.softmax(product, dim=2)
        context = torch.bmm(attn_weights, previous_hiddens).transpose(0,1)  
        #combined attention applied hidden source with hidden layer
        combined = torch.tanh(self.attn_combine(torch.cat((context, self.hidden[0]), 2)))
        cat_space = self.hidden2cat(combined)
        output = F.log_softmax(cat_space, dim=2)

        return output
        #return F.log_softmax(cat_space, dim=2)


class GRUGenre(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_cat, num_sentence, limit):
        super(GRUGenre, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first = True)
        self.sentences = num_sentence
        self.wordLimit = limit
        self.hidden2cat = nn.Linear(hidden_dim*2, num_cat)
        self.hidden = self.init_hidden()
        #attention
        self.attn_combine = nn.Linear(self.hidden_dim*2, self.hidden_dim*2)
        
    def init_hidden(self):
        # size: num_layers, batch_size, hidden_dim
        return torch.rand(1,self.sentences,self.hidden_dim).cuda()
               

    def forward(self, group_sentences): 
        # embeds of size (seq_len, batch_size, input_size=50)
        sentence_in = prepSentences(group_sentences, self.wordLimit).cuda()
        model_out, self.hidden = self.gru(sentence_in, self.hidden)
        previous_hiddens = model_out[:, :-2, :]
        #compare the final hidden with each previous hidden state to compute a
        #use dot product to compare
        product = torch.bmm(self.hidden.transpose(0,1), previous_hiddens.transpose(1,2))
        attn_weights = F.softmax(product, dim=2)
        context = torch.bmm(attn_weights, previous_hiddens).transpose(0,1)  
        #combined attention applied hidden source with hidden layer
        combined = torch.tanh(self.attn_combine(torch.cat((context, self.hidden), 2)))
        cat_space = self.hidden2cat(combined)
        output = F.log_softmax(cat_space, dim=2)

        return output
        #return F.log_softmax(cat_space, dim=2)