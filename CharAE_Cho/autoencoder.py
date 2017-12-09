#from CharAE_Cho import *
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import os
import numpy as np 

from encoder import cnn_encoder, rnn_encoder
from decoder import AttnDecoderRNN

class CharLevel_autoencoder(nn.Module):
      def __init__(self, criterion, num_symbols, use_cuda):#, seq_len):
            super(CharLevel_autoencoder, self).__init__()
            self.char_embedding_dim = 64
            self.pooling_stride = 5
            self.seq_len = 200
            self.num_symbols = num_symbols
            self.use_cuda = use_cuda

            self.filter_widths = list(range(1, 8)) 
            self.num_filters_per_width = 125 #[100, 100, 125, 125, 150, 150, 150, 150] 

            self.encoder_embedding = nn.Embedding(num_symbols, self.char_embedding_dim)
            self.cnn_encoder = cnn_encoder(
            filter_widths = self.filter_widths,
            num_filters_per_width = self.num_filters_per_width,
            char_embedding_dim = self.char_embedding_dim)
            #seq_len = self.seq_len)
            
            self.decoder_hidden_size = len(self.filter_widths) * self.num_filters_per_width
            self.rnn_encoder = rnn_encoder( 
            hidden_size = self.decoder_hidden_size )

            # decoder embedding dim dictated by output dim of encoder
            self.decoder_embedding = nn.Embedding(num_symbols, self.decoder_hidden_size)
            self.attention_decoder = AttnDecoderRNN(
                  num_symbols = num_symbols,
                  hidden_size = self.decoder_hidden_size, 
                  output_size = self.seq_len//self.pooling_stride)
            
            # if use_cuda:
            #       self.cnn_encoder = self.cnn_encoder.cuda()
            #       self.rnn_encoder = self.rnn_encoder.cuda()
            #       self.attention_decoder = self.attention_decoder.cuda()
            
            self.criterion = criterion

      def encode(self, data, seq_len, collect_filters = False):
            encoder_embedded = self.encoder_embedding(data).unsqueeze(1).transpose(2,3) 
            encoded = self.cnn_encoder.forward(encoder_embedded, self.seq_len, collect_filters)
            encoded = encoded.squeeze(2)
      
            encoder_hidden = self.rnn_encoder.initHidden()
            encoder_outputs = Variable(torch.zeros(64, seq_len//self.pooling_stride, 2*self.decoder_hidden_size))
            if self.use_cuda:
                  encoder_outputs = encoder_outputs.cuda()
                  encoder_hidden = encoder_hidden.cuda()

            for symbol_ind in range(self.seq_len//self.pooling_stride):#self.rnn_emits_len): 
                  output, encoder_hidden = self.rnn_encoder.forward(
                        encoded[:,:,symbol_ind], encoder_hidden)
                  #print(output.data.shape) # (81, 64, 128)
                  encoder_outputs[:, symbol_ind,:] = output[0]
            return encoder_outputs, encoder_hidden

      def decode(self, noisy_data, target_data, encoder_hidden, encoder_outputs, seq_len):   
            loss = 0
            decoder_hidden = encoder_hidden
            #print(target_data.data.shape)
            for amino_acid_index in range(self.seq_len): 
                  target_amino_acid = target_data[ :, :, amino_acid_index]#.long()
                  decoder_input = noisy_data.data[:, amino_acid_index].unsqueeze(1)#.transpose(0,1)    
                  decoder_embedded = self.decoder_embedding(decoder_input)
                 
                  # # current symbol, current hidden state, outputs from encoder 
                  decoder_output, decoder_hidden, attn_weights = self.attention_decoder.forward(
                  decoder_embedded, decoder_hidden, encoder_outputs, self.seq_len//self.pooling_stride)
                  #print(decoder_output.data.shape, target_amino_acid.data.shape)   # torch.Size([64, 23])
                  
                  loss += self.criterion(
                        decoder_output,
                        Variable(target_amino_acid) ) 
            return loss 

# preliminary model
# class cnn_autoencoder(nn.Module):
#       def __init__(self):
#             super(cnn_autoencoder, self).__init__()
#             self.encoder = cnn_encoder()
#             self.decoder = cnn_decoder()
#             self.embedding = nn.Embedding(22, 4)
            
#       def encode(self, data):
#             char_embeddings = self.embedding(data).unsqueeze(1).transpose(2,3) 
#             encoded, unpool_indices = self.encoder.forward(char_embeddings)
#             return encoded, unpool_indices

#       def decode(self, data, unpool_indices):
#             reconstructed = self.decoder.forward(data, unpool_indices)
#             return reconstructed