#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable

import os
import argparse
from layer import QRNNLayer
from model import QRNNModel

from data.util import fopen
from data.util import load_inv_dict

import data.data_utils as data_utils
from data.data_utils import seq2words
from data.data_utils import prepare_batch
from data.data_iterator import TextIterator

use_cuda = torch.cuda.is_available()

def load_model():
    if os.path.exists('model/model.pkl'):
        print 'Reloading model parameters..'
        checkpoint = torch.load('model/model.pkl')
        model = QRNNModel(QRNNLayer, checkpoint['num_layers'], checkpoint['kernel_size'],
                          checkpoint['hidden_size'], checkpoint['emb_size'],
                          checkpoint['num_enc_symbols'], checkpoint['num_dec_symbols'])
        print (checkpoint['state_dict'])
    else:
        raise ValueError('No such file:[{}]'.format(config.model_path))

    return model, checkpoint


def decode():
    model, config = load_model()

    # Load source data to decode
    test_set = TextIterator(source='dev_source_seqs.txt',
                            source_dict='vocab.p',
                            batch_size=64, maxlen=250,
                            n_words_source=125,
                            shuffle_each_epoch=False,
                            sort_by_length=False,)
    target_inv_dict = load_inv_dict('vocab.p')

    if use_cuda:
        print 'Using gpu..'
        model = model.cuda()

    try:
        print 'Decoding starts..'
        fout = fopen('dev_target_seqs.txt', 'w')

        for idx, source_seq in enumerate(test_set):

            source, source_len = prepare_batch(source_seq)

            preds_prev = torch.zeros(len(source), 100).long()
            preds_prev[:,0] += data_utils.start_token
            preds = torch.zeros(len(source), 100).long()

            if use_cuda:
                source = Variable(source.cuda())
                source_len = Variable(source_len.cuda())
                preds_prev = Variable(preds_prev.cuda())
                preds = preds.cuda()
            else:
                source = Variable(source)
                source_len = Variable(source_len)
                preds_prev = Variable(preds_prev)

            states, memories = model.encode(source, source_len)

            for t in xrange(100):
                # logits: [batch_size x max_decode_step, tgt_vocab_size]
                _, logits = model.decode(preds_prev[:,:t+1], states, memories)
                # outputs: [batch_size, max_decode_step]
                outputs = torch.max(logits, dim=1)[1].view(len(source), -1)
                preds[:,t] = outputs[:,t].data
                if t < 100 - 1:
                    preds_prev[:,t+1] = outputs[:,t]

            print ("len preds", len(preds))
            print ("preds", preds[0:100])
            print ("target_inv_dict", target_inv_dict)

            for i in xrange(len(preds)):
                print ("seq2words", seq2words(preds[i], target_inv_dict))
                #fout.write(str(seq2words(preds[i], target_inv_dict)) + '\n')
                #fout.flush()

            print '  {}th line decoded'.format(idx * 64)
        print 'Decoding terminated'

    except IOError:
        pass
    finally:
        fout.close()

decode()
