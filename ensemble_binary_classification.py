"""
# **Binary Classification**
"""

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

df_train = pd.read_csv('.../normal_train.csv', index_col=0)
df_valid = pd.read_csv('.../normal_test.csv', index_col=0)
df_attack = pd.read_csv('.../attack_data.csv', index_col=0)
df_attack['label']=1

df = pd.concat([df_train, df_valid, df_attack], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)

lines = df['syscalls'].values.tolist()
syscall_lines = []

#tokenizing the traces
for line in lines:
  tokens = word_tokenize(line)
  syscall_lines.append(tokens)

#train word2vec
model = gensim.models.Word2Vec(sentences=syscall_lines, size=200, window=5, workers=4, min_count=1)
words = list(model.wv.vocab)

filename = '.../embedding_word2vec_200.txt'
model.wv.save_word2vec_format(filename, binary=False)

import os
embedding_index = {}
system_call = []
embedding = []
f =open(os.path.join('', '.../embedding_word2vec_200.txt'), encoding='utf-8')
for line in f:
  values = line.split()
  word = values[0]
  coefs = np.asarray(values[1:])
  embedding_index[word] = coefs
  system_call.append(word)
  embedding.append(values[1:])
f.close()

#converting embedding file to csv
embedding.pop(0)
system_call.pop(0)
df_embedding = pd.DataFrame.from_records(zip_longest(system_call, embedding), columns=['System Call', 'Embedding'])
df_embedding.to_csv('.../embedding_200.csv', index=False)

#convert tokens to corresponding integer index
tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(syscall_lines)
sequences = tokenizer_obj.texts_to_sequences(syscall_lines)
word_index = tokenizer_obj.word_index

#create an embedding matrix containing embedding for each unique token 
num_words = len(word_index)+1
embedding_matrix = np.zeros((num_words, 200))
for word, i in word_index.items():
  if i>num_words:
    continue
  embedding_vector = embedding_index.get(word)
  if embedding_vector is not None:
    embedding_matrix[i]= embedding_vector

#padding system calls traces to fixed length
syscall_pad = pad_sequences(sequences, maxlen = 400)
label = df['label'].values
print('shape of syscall tensor: ', syscall_pad.shape)
print('shape of label tensor: ', label.shape)

#split into train and test
val_split = 0.5
indices = np.arange(syscall_pad.shape[0])
np.random.shuffle(indices)
syscall_pad = syscall_pad[indices]
label = label[indices]
num_val_samples = int(val_split*syscall_pad.shape[0])

X_train_pad = syscall_pad[: -num_val_samples]
Y_train = label[: -num_val_samples]
X_test_pad = syscall_pad[-num_val_samples :]
Y_test = label[-num_val_samples :]

#Model 1 : LSTM
model = Sequential() 
emb_layer = Embedding(num_words, 200, embeddings_initializer=Constant(embedding_matrix),
                    input_length=400, trainable=False)
model.add(emb_layer)
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy']) 
print(model.summary())

#Model 2 : Bi-LSTM
model2 = Sequential() 
emb_layer = Embedding(num_words, 200, embeddings_initializer=Constant(embedding_matrix),
                    input_length=400, trainable=False)
model2.add(emb_layer)
model2.add(Bidirectional(LSTM(128))) 
model2.add(Dense(1, activation='sigmoid'))
model2.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy']) 
print(model2.summary())

#Model 3 : GRU
model3 = Sequential() 
emb_layer = Embedding(num_words, 200, embeddings_initializer=Constant(embedding_matrix),
                    input_length=400, trainable=False)
model3.add(emb_layer)
model3.add(GRU(128, dropout=0.2, recurrent_dropout=0.2))
model3.add(Dense(1, activation='sigmoid')) 
model3.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy']) 
print(model3.summary())

#Model 4 : Bi-GRU
model4 = Sequential() 
emb_layer = Embedding(num_words, 200, embeddings_initializer=Constant(embedding_matrix),
                    input_length=400, trainable=False)
model4.add(emb_layer)
model4.add(Bidirectional(GRU(128))) 
model4.add(Dense(1, activation='sigmoid')) 
model4.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy']) 
print(model4.summary())

#train ensemble base classifiers
classifiers = {"lstm":model,"bilstm": model2,
               "gru": model3, "bigru": model4
               }
trainstart=time.time()
for key in classifiers:
    # Get classifier
    classifier = classifiers[key]
    # Fit classifier
    classifier.fit(X_train_pad, Y_train, epochs=25, batch_size=32, validation_split=0.2, 
                    callbacks=[EarlyStopping(monitor='val_loss',patience=3, min_delta=0.0001)])
    # Save fitted classifier
    classifiers[key] = classifier

trainend=time.time()
trainingtime = trainend - trainstart
print('training time',trainingtime)

results = pd.DataFrame()
for key in classifiers:
  y_pred = classifiers[key].predict(X_test_pad)
  # Save results in pandas dataframe object
  results[f"{key}"]=y_pred.reshape(y_pred.shape[0])

# Add the test set to the results object
results["Target"] = Y_test

x=results[['lstm','bilstm', 'gru', 'bigru']] 
y=results['Target']
samp_len=int(0.7*len(results))
trainx_ens=x[:samp_len].values
trainy_ens=y[:samp_len].values
testx_ens=x[samp_len:].values
testy_ens=y[samp_len:].values
trainx_ens = trainx_ens.reshape(-1, 1, 4)
testx_ens=testx_ens.reshape(-1, 1, 4)

#create meta classifier
model_meta = Sequential()
model_meta.add(LSTM(2, input_shape=(1,4)))
model_meta.add(Dense(1, activation='sigmoid'))
model_meta.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model_meta.summary())

#train meta-classifier
history=model_meta.fit(trainx_ens, trainy_ens, epochs=30, batch_size=32, validation_split=0.2, 
                    callbacks=[EarlyStopping(monitor='val_loss',patience=3, min_delta=0.0001)] )

#function to plot validation curves
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs",fontsize=15)
  plt.ylabel("Loss",fontsize=15)
  
  plt.legend(['training '+string, 'validation '+string], fontsize=15)
  plt.savefig(string+'.eps', bbox_inches='tight')
  plt.savefig(string+'.png', bbox_inches='tight')
  plt.show()
   
#plot validation curves 
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

#evaluate performance of ensemble framework 
Y_predicted = model_meta.predict(testx_ens)
Y_pred = [float(np.round(x)) for x in Y_predicted]

print(classification_report(testy_ens, Y_pred))

cm = confusion_matrix(testy_ens, Y_pred)
precision=cm[1][1]/(cm[1][1]+cm[0][1])
recall=cm[1][1]/(cm[1][1]+cm[1][0])
fscore=(2*prc*rcl)/(prc+rcl)
fpr = cm[0][1]/(cm[0][0]+cm[0][1]
                
