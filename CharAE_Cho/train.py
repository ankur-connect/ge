#from CharAE_Cho import * 

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import os
from loader import loader
import numpy as np 

from autoencoder import CharLevel_autoencoder
#import pickle 

num_epochs = 7
batch_size = 64
learning_rate = 1e-3
max_batch_len = 200
num_symbols = 125
use_cuda = torch.cuda.is_available()

criterion = nn.BCEWithLogitsLoss()

if use_cuda:
    model = CharLevel_autoencoder(criterion, num_symbols, use_cuda).cuda()
else:
    model = CharLevel_autoencoder(criterion, num_symbols, use_cuda)#, seq_len)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

import sys 

def _one_hot(batch, seq_len, use_cuda):
        inp = batch % num_symbols
        inp_ = torch.unsqueeze(inp, 2)
        
        batch_onehot = torch.FloatTensor(64, seq_len, num_symbols).zero_()
        batch_onehot.scatter_(2, inp_ , 1).float()
        batch_onehot = batch_onehot.transpose(1,2)
        # (batch_size, 1, num_symbols, seq_len)
        if use_cuda:
            return batch_onehot.cuda()
        
        return batch_onehot

def train(model, optimizer, num_epochs, batch_size, learning_rate):
    #model.load_state_dict(torch.load('./autoencoder.pth'))
    train_loader, valid_loader = loader()
    
    latent_representation = []
    representation_indices = []
    for epoch in range(num_epochs):
        model.train()
        for index, (data, label) in enumerate(train_loader):
            #batch_onehot = _one_hot(data, max_batch_len)
            label_onehot = _one_hot(label, max_batch_len, use_cuda)
            
            if use_cuda: 
                data = Variable(data).cuda()
            else:
                data = Variable(data)
            
            # ===================forward=====================
            if epoch != num_epochs - 1:
                encoder_outputs, encoder_hidden = model.encode(data, max_batch_len)
            #print(encoder_outputs.data.shape, encoder_hidden.data.shape) 
            else:
                encoder_outputs, encoder_hidden = model.encode(
                    data, max_batch_len)#, collect_filters = True)
                rep = encoder_outputs.data.view(64, -1).numpy()
                if index == 100:
                    print('good job, everything looks good: a batchs latent rep has shape', rep.shape)
                #latent_representation.append(rep)
                #representation_indices.append(label.numpy())
                
            decoder_hidden = encoder_hidden
            #print('deocder input', decoder_input.shape, 'decoder hidden', decoder_hidden.data.shape)
            
            #send in corrupted data to recover clean data
            loss = model.decode(
                data, label_onehot, decoder_hidden, encoder_outputs, max_batch_len)
            
            if index % 1000 == 0:
                print(epoch, index, loss.data[0])
                print( evaluate(model, valid_loader))
                model.train()
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # ===================log========================
        torch.save(model.state_dict(), './autoencoder.pth')
        print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch+1, num_epochs, loss.data[0]))

    # pickle.dump(
    #     latent_representation, open( "./data/latent_rep.p", "wb" ), protocol=4 )
    # pickle.dump(
    #     representation_indices, open( "./data/latent_rep_indices.p", "wb" ), protocol=4)

def evaluate(model, valid_loader ):
    model.eval()
    for index, (data, label) in enumerate(valid_loader):
        batch_onehot = _one_hot(label, max_batch_len, use_cuda)
        
        if use_cuda: 
            data = Variable(data).cuda()
        else:
            data = Variable(data)
            
        encoder_outputs, encoder_hidden = model.encode(data, max_batch_len)
        decoder_hidden = encoder_hidden
        
        loss = model.decode(data, batch_onehot, decoder_hidden, encoder_outputs, max_batch_len)
        return loss.data[0]

if __name__ == '__main__':
    with open('progress_update_big_data.txt','w') as f:
        sys.stdout = f
        train(model, optimizer, num_epochs, batch_size, learning_rate)


    def _trim_padding(batch):    
        # mask batch to detect non-zero elements than sum along the sequence length axis
        batch_max_len = torch.max(batch.gt(0).cumsum(dim = 1))
        return batch[:, :batch_max_len], batch_max_len 


    