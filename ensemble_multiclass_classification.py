"""
# **Multiclass Classification**
"""

df_train = pd.read_csv('/content/drive/My Drive/ADFA-LD/normal_train.csv', index_col=0)
df_valid = pd.read_csv('/content/drive/My Drive/ADFA-LD/normal_test.csv', index_col=0)
df_attack = pd.read_csv('/content/drive/My Drive/ADFA-LD/attack_data.csv', index_col=0)

df = pd.concat([df_train, df_valid, df_attack], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)

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

y_train= to_categorical(Y_train)
y_test= to_categorical(Y_test)

#Model 1 : LSTM
model = Sequential() 
emb_layer = Embedding(num_words, 200, embeddings_initializer=Constant(embedding_matrix),
                    input_length=400, trainable=False)
model.add(emb_layer)
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(7, activation='softmax'))  
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy']) 
print(model.summary())

#Model 2 : Bi-LSTM
model2 = Sequential() 
emb_layer = Embedding(num_words, 200, embeddings_initializer=Constant(embedding_matrix),
                    input_length=400, trainable=False)
model2.add(emb_layer)
model2.add(Bidirectional(LSTM(128))) 
model2.add(Dense(7, activation='softmax')) 
model2.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy']) 
print(model2.summary())

#Model 3 : GRU
model3 = Sequential() 
emb_layer = Embedding(num_words, 200, embeddings_initializer=Constant(embedding_matrix),
                    input_length=400, trainable=False)
model3.add(emb_layer)
model3.add(GRU(128, dropout=0.2, recurrent_dropout=0.2))
model3.add(Dense(7, activation='softmax')) 
model3.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy']) 
print(model3.summary())

#Model 4 : Bi-GRU
model4 = Sequential() 
emb_layer = Embedding(num_words, 200, embeddings_initializer=Constant(embedding_matrix),
                    input_length=400, trainable=False)
model4.add(emb_layer)
model4.add(Bidirectional(GRU(128))) 
model4.add(Dense(7, activation='softmax')) 
model4.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy']) 
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
    classifier.fit(X_train_pad, y_train, epochs=25, batch_size=32, validation_split=0.2, 
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
  Y_pred = np.argmax(y_pred, axis=1)
  results[f"{key}"]=Y_pred.reshape(Y_pred.shape[0])

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

y_train_ens= to_categorical(trainy_ens)
y_test_ens= to_categorical(testy_ens)

#create meta classifier
model_meta = Sequential()
model_meta.add(LSTM(2, input_shape=(1,4)))
model_meta.add(Dense(7, activation='softmax'))
model_meta.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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
Y_pred = np.argmax(Y_predicted, axis=1)

print(classification_report(testy_ens, Y_pred))

from sklearn.metrics import precision_score, recall_score, f1_score
print('Precision: ', precision_score(testy_ens, Y_pred, average='weighted'))

print('Recall: ', recall_score(testy_ens, Y_pred, average='weighted'))

print('F1 score: ', f1_score(testy_ens, Y_pred, average='weighted'))