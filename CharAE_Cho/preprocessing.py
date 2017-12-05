from CharAE_Cho import *

from util.tsne import hist
import pandas as pd 

# super redundant, just need to load all data into memory then call encoder.fit()
def loader(raw_txt_path):
      labels, vals = [], []
      lengths = []
      max_len = 0
      with open(raw_txt_path) as file:
            for line_index, line in enumerate(file):
                  max_len += 1
                  val, label = line.split('<JOIN>')[0], line.split('<JOIN>')[-1]
                  lengths.append(len(val))
                  if len(val) > 200 or len(val) < 50:
                        continue
                  
                  labels.append(label)
                  
                  #if len(val) > 500:
                  #      max_len = len(val)

                  vals.append(val)
      df = pd.DataFrame({'labels': labels, 'vals': vals})
      
      #hist('Percentage of Non-Gap Symbols by Sample', lengths)
      #print(max_len)

      return df 

df = loader('./data/train.txt')

from sklearn.preprocessing import LabelEncoder
import pickle
import torch

def encode(df, raw_txt_path = './data/train.txt', dictionary = './data/dic_lookup.p'):
      seq_list, target_list = df['vals'].tolist(), df['labels'].tolist() 

      val_encoder = LabelEncoder()
      #label_encoder = LabelEncoder()
      encoded_vals, encoded_labels = [], []

      special_tokens = ["<SOS>", "<EOS>", "<PAD>"]#, "<UNK>"]
      exotic_tokens = ['£', 'ū', '你', '”', '–', 'ë', 'É', 'ç', 'é', 'è', 'Ç', 'ä', 
      '…', 'ć', '葱', 'ã', 'ï', 'í', 'ê', '£', 'à', '’', 'ó', 'ī', '€', 'á', 
      'ø', '“', 'Å', '♪', '♫', 'ñ', '¡', '²', 'â', '‘', '送', 'Č', 'ô', 'ā', 
      '—', 'ü', '\xa0', 'ö', '»', '¼', 'Ü', '¥', '‟', 'ý', 
      '\ufeff', '×', '³', '\xad', '\x85', '©', '\x8a', 'š', 'ß', 
      'Ӓ', 'Ö', '，', '¾', '›', '\x9a', '‹', '\x9f', '½', '‒', '™', 
      '„', '«', 'ú', 'β', '´', '°', 'к', '®', '\x96', 'œ', 'Ä', 'ō', 'î']

      val_encoder.fit( special_tokens + list(set(''.join(seq_list) + ''.join(target_list) )  )  )
      #val_encoder.fit( special_tokens + list(set(''.join(target_list))))
      print(len(special_tokens + list(set(''.join(seq_list) + ''.join(target_list))) ) )

      with open(raw_txt_path) as file:
      #       #label_encoder.fit(   )
            #print(  val_encoder.transform( ['<SOS>','5']), val_encoder.inverse_transform([31]) )
            #print( val_encoder.get_params() )

            for line_index, line in enumerate(file):
                  #if line_index < 190000:
                  #      continue

                  val, label = line.split('<JOIN>')[0], line.split('<JOIN>')[-1]
                  if len(val) > 200 or len(label) >200:
                        continue
                  
                  for t in  special_tokens + exotic_tokens:
                        val = val.replace(t, "")  # delete
                        label = label.replace(t, "" )

                  values, labels= np.array(list(val)), np.array(list(label))
                  int_encoded_vals = val_encoder.transform(values)
                  int_encoded_labels = val_encoder.transform(labels)
                  #print(int_encoded_vals)
                  # poor padding 
                  
                  for pad in range(200- len(int_encoded_vals)):
                        int_encoded_vals = np.append(int_encoded_vals, val_encoder.transform(['<PAD>']))
                  
                  #if len(int_encoded_labels) > 200:
                  #      int_encoded_labels = int_encoded_labels[:200]

                  for pad in range(200- len(int_encoded_labels)):
                        int_encoded_labels = np.append(int_encoded_labels, val_encoder.transform(['<PAD>']))
                  
                  encoded_vals.append(int_encoded_vals)
                  #print(int_encoded_vals.shape, int_encoded_labels.shape)
                  encoded_labels.append(int_encoded_labels)

                  if line_index % 10000 == 0 and line_index > 2:
                        print(line_index, np.array(encoded_vals).shape, np.array(encoded_labels).shape)
                        pickle.dump( np.array(encoded_vals), open( "./data/%s.p" %line_index, "wb" ), protocol=4 )
                        pickle.dump( np.array(encoded_labels), open( "./data/%sl.p" %line_index, "wb" ), protocol=4 )
                        encoded_vals, encoded_labels = [], []

                  if line_index == 200:
                        print(val, int_encoded_vals)
                        print(label, int_encoded_labels)
                        print(len(int_encoded_labels))
                        
      

encode(df, './data/train.txt')
