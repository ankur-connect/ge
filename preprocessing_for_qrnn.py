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

USE_CUDA = torch.cuda.is_available()

with open("/Users/carolineroper/Documents/School/Natural Language Processing/Char-NMT/data/train.txt", "r", encoding="utf-8") as f:
    
    DE_seq = []
    EN_seq = []
    
    for i, line in enumerate(f):
        
        line = line.split('<JOIN>')
        DE_seq.append(line[0].replace(" <EOS>", ""))
        EN_seq.append(line[1].replace(" <EOS>", "").replace(" \n",""))

thefile = open('train_source_seqs.txt', 'w', encoding = 'utf-8')
for item in DE_seq:
    thefile.write("%s\n" % item)

thefile = open('train_target_seqs.txt', 'w', encoding = 'utf-8')
for item in EN_seq:
    thefile.write("%s\n" % item)

with open("/Users/carolineroper/Documents/School/Natural Language Processing/Char-NMT/data/dev.txt", "r", encoding="utf-8") as f:
    
    DE_seq = []
    EN_seq = []
    
    for i, line in enumerate(f):
        
        line = line.split('<JOIN>')
        DE_seq.append(line[0].replace("<EOS>", ""))
        EN_seq.append(line[1].replace("<EOS>", "").replace("\n", ""))
        

thefile = open('dev_source_seqs.txt', 'w', encoding = "utf-8")
for item in DE_seq:
    thefile.write("%s\n" % item)

thefile = open('dev_target_seqs.txt', 'w', encoding = 'utf-8')
for item in EN_seq:
    thefile.write("%s\n" % item)

with open("/Users/carolineroper/Downloads/lihan_directory/data/test.txt", "r", encoding="utf-8") as f:
    
    DE_seq = []
    EN_seq = []
    
    for i, line in enumerate(f):
        
        line = line.split('<JOIN>')
        DE_seq.append(line[0].replace(" <EOS>", ""))
        EN_seq.append(line[1].replace(" <EOS>", "").replace(" \n",""))

thefile = open('test_source_seqs.txt', 'w', encoding = 'utf-8')
for item in DE_seq:
    thefile.write("%s\n" % item)

thefile = open('test_target_seqs.txt', 'w', encoding = 'utf-8')
for item in EN_seq:
    thefile.write("%s\n" % item)

