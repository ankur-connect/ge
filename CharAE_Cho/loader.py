''' loads pickles produced via preprocessing.py
    instantiates a pytorch loader according to input parameters
'''
from torch.utils.data import TensorDataset,Dataset, DataLoader
#from util.preprocessing import *
import numpy as np
import pickle
import torch 
#denoising pass: clean and noisy unlabeled data
#backtranslation pass: clean unlabeled data   
#supervised pass: labeled data with vector to check

def loader( batch_size = 64, shuffle = True, train_portion = 0.9, seq_format = False):

    mypath = './data/'
    # from os import listdir
    # from os.path import isfile, join
    # onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    names = [str(i) for i in range(10000, 210000, 10000)]

    source_data, target_data = False, False
    
    for number in names:
        target_chunk = pickle.load(open(mypath + '%sl.p' %number, "rb"))
        source_chunk = pickle.load(open(mypath + '%s.p' %number, "rb"))
        
        #if source_chunk.shape != (10000, 200):
        #    continue
        if  type(source_data) == bool:
            source_data, target_data = source_chunk, target_chunk
            continue
        #print( target_chunk.shape, source_chunk.shape)        
        #print(target_chunk.shape, source_chunk.shape)

        source_data = np.vstack( (source_data, source_chunk ) )
        target_data = np.vstack( ( target_data, target_chunk ) )

    source_data = np.array(source_data)
    #print(source_data.shape)
    target_data = np.array(target_data)

    #dataset = pickle.load(open(data_pickle_path, "rb"))
    num_samples = source_data.shape[0]
    print(num_samples)
    uniform_sampling = np.random.random_sample((num_samples,))

    train_dataset = source_data[ uniform_sampling < train_portion]
    #print(train_dataset.shape)
    train_labels = target_data[  uniform_sampling < train_portion]
    #print(train_labels.shape, train_dataset.dtype)
    valid_dataset = source_data[ uniform_sampling >= train_portion ]
    valid_labels = target_data[  uniform_sampling >= train_portion]
    #print(train_dataset.shape, type(train_dataset))

    #print(len(train_dataset[0]), len(train_dataset[4]))
    source = TensorDataset(torch.from_numpy(train_dataset), 
    torch.from_numpy(train_labels)) 

    train_loader = DataLoader(
    source, batch_size = batch_size, shuffle = shuffle, drop_last=True)

    valid_source = TensorDataset(torch.from_numpy(valid_dataset),
    torch.from_numpy(valid_labels)) 

    valid_loader = DataLoader(
    valid_source, batch_size = batch_size, shuffle = shuffle, drop_last=True)
    
    return train_loader, valid_loader

if __name__ == '__main__':
    loader()