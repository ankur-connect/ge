#!/usr/bin/env python

from __future__ import unicode_literals, print_function, division
from collections import Counter
from io import open
import unicodedata
import string
import re
import random
import time
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import numpy as np
import torch.optim.lr_scheduler as lr
from random import shuffle
import math
from torch.utils.data import Dataset

sys.stdout=open('progress_update_small_data.txt','w')

'''This module trains a word-level model similar to Luong et al 2015.
To call this module, use:
python3.5 nmt-hybrid-global-attention.py custom_schedule (T/F), uniform init (T/F), vocab_size (int),
e.g. python3.5 nmt-hybrid-global-attention.py True, True, 20000
Must call in same folder as train.txt'''

USE_CUDA = torch.cuda.is_available()

with open("train.txt", "r", encoding="utf8") as f:
    
    DE_seq = []
    EN_seq = []
    
    for i, line in enumerate(f):
        
        line = line.split('<JOIN>')
        DE_seq.append(line[0])
        EN_seq.append(line[1])

# ### Indexing words
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = s.replace(" <EOS>", "")
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s) #separates punctuation from the word
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s) #strips anything that isn't a character of punctuation
    return s


def readLangs(seq1, seq2):
    print("Reading lines...")
    
    seq1 = [normalizeString(s) for s in seq1]
    seq2 = [normalizeString(s) for s in seq2]
    
    pairs = list(map(list, zip(seq1, seq2)))

    return pairs

pairs = readLangs(DE_seq, EN_seq)

#will ultimately remove this and set MAX_LENGTH to either 50 like the paper or the longest sentence in corpus
MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH  \
        and p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

pairs = filterPairs(pairs)

def find_vocabulary(seq, vocab_size):
    counter = Counter()
    for sentence in seq:
        counter.update(sentence.split())
    vocabulary = [count[0] for count in counter.most_common(vocab_size)]
    return vocabulary

vocab_size = int(sys.argv[3])
input_vocab = find_vocabulary([pair[0] for pair in pairs], vocab_size)
output_vocab = find_vocabulary([pair[1] for pair in pairs], vocab_size)

UNK_token = 0
SOS_token = 1
EOS_token = 2
PAD_token = 3

class Lang:
    def __init__(self, name, vocab):
        self.name = name
        self.vocab = vocab
        self.word2index = {}
        self.word2count = {}
        self.index2word = dict(zip(range(4,len(vocab)+4), vocab))
        self.index2word[UNK_token] = "<UNK>"
        self.index2word[SOS_token] = "<SOS>"
        self.index2word[EOS_token] = "<EOS>"
        self.index2word[PAD_token] = "<PAD>"
        self.word2index = {v: k for k, v in self.index2word.items()}
        self.n_words = len(self.index2word)

input_lang = Lang('ger', input_vocab)
output_lang = Lang('eng', output_vocab)

def indexesFromSentence(lang, sentence, seq_len):
    #returns 0 if not found in word2index
    indexes = [Counter(lang.word2index)[word] for word in sentence.split(' ')]
    #adds EOS token at EOS
    indexes.extend([EOS_token])
    #trims to the seq len
    indexes = indexes[0:(min(seq_len, len(indexes)))]
    #pads if needed
    indexes.extend([3] * (seq_len - len(indexes)))
    return np.array(indexes)

def variableFromSentence(lang, sentence, seq_len):
    indexes = indexesFromSentence(lang, sentence, seq_len)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if USE_CUDA:
        return result.cuda()
    else:
        return result

def variablesFromPair(pair, seq_len):
    input_variable = variableFromSentence(input_lang, pair[0], seq_len)
    target_variable = variableFromSentence(output_lang, pair[1], seq_len)
    return (input_variable, target_variable)


# # Building the models

# ## The Encoder

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, batch_size, uniform_init=False, dropout_p=0.2):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.batch_size = batch_size
        self.lstm = nn.LSTM(hidden_size, hidden_size, dropout = dropout_p)
        self.dropout = nn.Dropout(self.dropout_p)
        
        if uniform_init == True:
            self.init_weights()
        
    def forward(self, x, hidden):
        seq_len = len(x)
        embedded = self.embedding(x).view(seq_len, self.batch_size, -1)
        output = self.dropout(embedded)
        for i in range(self.n_layers):
            output, hidden = self.lstm(output, hidden)
        return output, hidden
    
    def init_weights(self):
        initrange = 0.1
        self.lstm.weight_ih_l0.data.uniform_(-initrange, initrange)
        self.lstm.weight_hh_l0.data.uniform_(-initrange, initrange)
        self.lstm.bias_ih_l0.data.uniform_(-initrange, initrange)
        self.lstm.bias_hh_l0.data.uniform_(-initrange, initrange)
        
    def init_hidden(self):
        #the paper says to use "uniform initialization of parameters in [−0.1,0.1]: does that include these?"
        h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        if USE_CUDA:
            return h0.cuda(), c0.cuda()
        else:
            return h0, c0


# ### Luong Attention

class Attn(nn.Module):
    def __init__(self, hidden_size, batch_size, method = 'general'):
        super(Attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)

        attn_energies = Variable(torch.zeros(self.batch_size, seq_len)) # B x 1 x S
        if USE_CUDA: attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        for i in range(seq_len):
            attn_energies[:,i] = self.score(hidden, encoder_outputs[i])

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        # I confirmed that this computes row-wise, which is what we need
        after_softmax = F.softmax(attn_energies)
        return after_softmax.unsqueeze(0)
    
    def score(self, hidden, encoder_output):
        
        if self.method == 'dot':
            energy = torch.diag(torch.matmul(hidden, torch.transpose(encoder_output,0,1)))
            #print ("energy", energy)
            return energy
        
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = torch.diag(torch.matmul(hidden, torch.transpose(energy,0,1)))
            return energy


# Build a decoder that plugs this Attn module in after the RNN to calculate attention weights, and apply those weights to the encoder outputs to get a context vector.

class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers, batch_size, uniform_init = False, dropout_p=0.2):
        super(AttnDecoderRNN, self).__init__()
        
        # Keep parameters for reference
        self.attn = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        
        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.n_layers = n_layers
        self.lstm = nn.LSTM(hidden_size*2, hidden_size, dropout = dropout_p)
        self.out = nn.Linear(hidden_size*2, output_size)
        
        if uniform_init == True:
            self.init_weights()
    
    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        
        # Get the embedding of the current input word (last output word)
        
        hidden = last_hidden
        word_embedded = self.embedding(word_input).view(1, self.batch_size, -1) # S=1 x B x N
        output = word_embedded

        #Combine embedded input word and last context, run through LSTM
        
        for i in range(self.n_layers):
            output = torch.cat((output, last_context.unsqueeze(0)), 2)
            output, hidden = self.lstm(output, hidden)

        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attn(word_embedded.squeeze(0), encoder_outputs)
        context = torch.bmm(attn_weights.transpose(0, 1), encoder_outputs.transpose(0, 1)) # B x 1 x N
        
        # Final output layer (next word prediction) using the hidden state and context vector
        output = output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1) # B x S=1 x N -> B x N
        output = F.log_softmax(self.out(torch.cat((output, context), 1)))
        
        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights
    
    def init_weights(self):
        initrange = 0.1
        self.lstm.weight_ih_l0.data.uniform_(-initrange, initrange)
        self.lstm.weight_hh_l0.data.uniform_(-initrange, initrange)
        self.lstm.bias_ih_l0.data.uniform_(-initrange, initrange)
        self.lstm.bias_hh_l0.data.uniform_(-initrange, initrange)


# # Training
# 
# ## Defining a training iteration

clip = 5.0

def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, batch_size, seq_length):

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # Added onto for each word

    # Get size of input and target sentences
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    # Run words through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
    
    # Prepare input and output variables
    # One SOS token for each sentence, so length is batch size
    decoder_input = Variable(torch.LongTensor([[SOS_token]*batch_size]))
    decoder_context = Variable(torch.zeros(batch_size, decoder.hidden_size))
    decoder_hidden = encoder_hidden # Use last hidden state from encoder to start decoder
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()
        
    for di in range(target_length):
        decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
        
        loss += criterion(decoder_output, target_variable[di])
        
        # Get most likely word index (highest value) from output
        topv, topi = decoder_output.data.topk(1)
        
        ni = topi.squeeze(1)

        decoder_input = Variable(ni) # Chosen word is next input
        if USE_CUDA: 
            decoder_input = decoder_input.cuda()

    # Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.data[0] / target_length

#Custom schedule for learning rate

def custom_schedule(epoch):
    if (epoch < 4):
        return (1)
    elif epoch == 4:
        return (.5)
    elif epoch >= 4:
        return (.5**(1+(epoch - 3)))

hidden_size = 512
BATCH_SIZE = 32
N_LAYERS = 4
SEQ_LENGTH = 10
if sys.argv[1] == "True":
    learning_rate = 1
else:
    learning_rate = 0.001

attn_model = Attn(hidden_size, BATCH_SIZE)
# Initialize models

if sys.argv[2] == "True":
    encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers=N_LAYERS, batch_size = BATCH_SIZE, uniform_init = True)
    decoder = AttnDecoderRNN(attn_model, hidden_size, output_lang.n_words, n_layers = N_LAYERS, batch_size=BATCH_SIZE, uniform_init = True) 

else:
    encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers=N_LAYERS, batch_size = BATCH_SIZE)
    decoder = AttnDecoderRNN(attn_model, hidden_size, output_lang.n_words, n_layers = N_LAYERS, batch_size=BATCH_SIZE) 

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    decoder.cuda()

encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate) #should be optim.SGD according to the paper
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

if sys.argv[1] == "True":
    scheduler1 = lr.LambdaLR(encoder_optimizer, lr_lambda=custom_schedule)
    scheduler2 = lr.LambdaLR(decoder_optimizer, lr_lambda=custom_schedule)

criterion = nn.NLLLoss()

class TranslationDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """
    
    def __init__(self, data_list, seq_length):
        """
        @param data_list: list of IMDBDatum
        """
        self.data_list = data_list
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        return indexesFromSentence(input_lang, self.data_list[key][0], seq_len = self.seq_length), \
    indexesFromSentence(output_lang, self.data_list[key][1], seq_len = self.seq_length)

dataset = TranslationDataset(pairs, SEQ_LENGTH)

train_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                       batch_size=BATCH_SIZE,
                                       shuffle=True)

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

# Keep track of time elapsed and running averages
start = time.time()
plot_losses = []
print_loss_total = 0 # Reset every print_every
print_loss_num = 0 # Reset every plot_every

#set up for training
n_epochs = 30

print("begin training ")
print(start)
print("\n")

#train
for epoch in range(1,n_epochs+1):
    if sys.argv[1] == "True":
        scheduler1.step()
        scheduler2.step()
    for i, (input_tensor, target_tensor) in enumerate(train_loader):
        if input_tensor.size()[0]==BATCH_SIZE:
            #this is needed because the loader puts the remainder datum in a too-small batch
            input_tensor = input_tensor.squeeze(1).transpose(0,1)
            target_tensor = target_tensor.squeeze(1).transpose(0,1)
            input_variable = Variable(input_tensor)
            target_variable = Variable(target_tensor)
            if USE_CUDA:
                input_variable = input_variable.cuda()
                target_variable = target_variable.cuda()

            loss = train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer,\
                         criterion, BATCH_SIZE, SEQ_LENGTH)

            # Keep track of loss
            print_loss_total += loss
            print_loss_num += 1

        else:
            pass

    print_loss_avg = print_loss_total / print_loss_num
    print_loss_total = 0
    print_loss_num = 0
    # file-output.py
    print(str(time_since(start, epoch/n_epochs)) + ' ' + str(print_loss_avg) + "\n") 

torch.save(encoder.state_dict(), "saved_encoder.pth")
torch.save(decoder.state_dict(), "saved_decoder.pth")

# # Evaluating the network

def evaluate(pairs, seq_length = SEQ_LENGTH, batch_size = BATCH_SIZE):
    
    # get a random batch
    shuffle(pairs)
    batch_pairs = pairs[0:batch_size]
    batch_input = [pair[0] for pair in pairs]
    batch_target = [pair[1] for pair in pairs]
    batch_vars = [variablesFromPair(pair, seq_len=SEQ_LENGTH) for pair in batch_pairs]
    
    # stack that batch
    inputs = [variable[0].data for variable in batch_vars]
    input_variable = Variable(torch.stack(inputs, 1).squeeze(), volatile=False)
    targets = [variable[1].data for variable in batch_vars]
    target_variable = Variable(torch.stack(targets, 1).squeeze(), volatile=False)
    
    # Get size of input and target sentences
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    # Run words through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
    
    # Prepare input and output variables
    # One SOS token for each stence, so length is batch size
    decoder_input = Variable(torch.LongTensor([[SOS_token]*batch_size]))
    decoder_context = Variable(torch.zeros(batch_size, decoder.hidden_size))
    decoder_hidden = encoder_hidden
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()
    
    decoded_words = []
    decoder_attentions = torch.zeros(seq_length, seq_length)
    
    # Run through decoder
    
    for di in range(seq_length):
        decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)

        topv, topi = decoder_output.data.topk(1)
        ni = topi.squeeze(1)
        
        batch_decoded_words = []
        for word_index in ni.cpu().numpy():
            batch_decoded_words.append(output_lang.index2word[word_index])
        
        decoded_words.append(batch_decoded_words)
            
        decoder_input = Variable(ni) # Chosen word is next input
        if USE_CUDA: decoder_input = decoder_input.cuda()
            
    decoded_words_matrix = np.array(decoded_words).reshape((SEQ_LENGTH, BATCH_SIZE))
        
    decoded_sentences = []

    for column in range(decoded_words_matrix.shape[1]):
        decoded_sentence = ' '.join(list(decoded_words_matrix[:, column]))
        decoded_sentences.append(decoded_sentence)
    
    return batch_input, batch_target, decoded_sentences, decoder_attentions[:di+1, :len(encoder_outputs)]


example_input, example_target, decoded_sentences, decoder_attentions = evaluate(pairs)

for i in range(20):
    print("input: " + str(example_input[i]) + "\n")
    print("target: " + str(example_target[i]) + "\n")
    print("translation: " + str(decoded_sentences[i]) + "\n" + "\n")

