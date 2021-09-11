# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
import gensim
from itertools import zip_longest

from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input, Embedding, Bidirectional, GRU, Activation
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from keras.regularizers import l2
from keras.initializers import Constant
import time
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# %matplotlib inline

all_train_files = glob.glob('.../ADFA-LD/Training_Data_Master/*.txt')

train_list = []
for file in all_train_files:
    with open(file) as f:
        for term in f:
            term = term.strip()
            train_list.append(term)
        f.close()

df_train = pd.DataFrame(train_list) 
df_train['label'] = 0
df_train.columns = ['syscalls', 'label']

df_train.to_csv(".../normal_train.csv")

all_valid_files = glob.glob('.../ADFA-LD/Validation_Data_Master/*.txt')

valid_list = []
for file in all_valid_files:
    with open(file) as f:
        for term in f:
            term = term.strip()
            valid_list.append(term)
        f.close()

df_valid = pd.DataFrame(valid_list) 
df_valid['label'] = 0
df_valid.columns = ['syscalls', 'label']

df_valid.to_csv(".../normal_test.csv")

print("adduser files reading...")
for i in range(1,11):
  files = glob.glob('.../ADFA-LD/Attack_Data_Master/Adduser_' + str(i) + '/*.txt')

attack_list = []
for file in files:
    with open(file) as f:
        for term in f:
            term = term.strip()
            attack_list.append(term)
        f.close()

df_adduser = pd.DataFrame(attack_list) 
df_adduser['label']=1

print("hydra-ftp files reading...")

for i in range(1,11):
  files = glob.glob('.../ADFA-LD/Attack_Data_Master/Hydra_FTP_' + str(i) + '/*.txt')

attack_list = []
for file in files:
    with open(file) as f:
        for term in f:
            term = term.strip()
            attack_list.append(term)
        f.close()

df_hydraftp = pd.DataFrame(attack_list) 
df_hydraftp['label']=2

print("hydra-ssh files reading...")

for i in range(1,11):
  files = glob.glob('.../ADFA-LD/Attack_Data_Master/Hydra_SH_' + str(i) + '/*.txt')

attack_list = []
for file in files:
    with open(file) as f:
        for term in f:
            term = term.strip()
            attack_list.append(term)
        f.close()

df_hydrassh = pd.DataFrame(attack_list) 
df_hydrassh['label']=3

print("java meterpreter files reading...")

for i in range(1,11):
  files = glob.glob('.../ADFA-LD/Attack_Data_Master/Java_Meterpreter_' + str(i) + '/*.txt')

attack_list = []
for file in files:
    with open(file) as f:
        for term in f:
            term = term.strip()
            attack_list.append(term)
        f.close()

df_javamtrptr = pd.DataFrame(attack_list) 
df_javamtrptr['label']=4

print("meterpreter files reading...")

for i in range(1,11):
  files = glob.glob('.../ADFA-LD/Attack_Data_Master/Meterpreter_' + str(i) + '/*.txt')

attack_list = []
for file in files:
    with open(file) as f:
        for term in f:
            term = term.strip()
            attack_list.append(term)
        f.close()

df_mtrptr = pd.DataFrame(attack_list) 
df_mtrptr['label']=5

print("web-shell files reading...")

for i in range(1,11):
  files = glob.glob('.../ADFA-LD/Attack_Data_Master/Webshell_' + str(i) + '/*.txt')

attack_list = []
for file in files:
    with open(file) as f:
        for term in f:
            term = term.strip()
            attack_list.append(term)
        f.close()

df_webshell = pd.DataFrame(attack_list) 
df_webshell['label']=6

df_attack = pd.concat([df_adduser, df_hydraftp, df_hydrassh,df_javamtrptr, df_mtrptr, df_webshell], ignore_index=True)
df_attack.columns = ['syscalls', 'label'] 
df_attack = df_attack.sample(frac=1).reset_index(drop=True)

df_attack.to_csv('.../attack_data.csv')
