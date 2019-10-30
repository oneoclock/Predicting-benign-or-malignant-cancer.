import pandas as pd
#import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
#import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


dataset_path="C:/Users/Hiral/Desktop/DLNLP/a1.1/breast-cancer.csv"

raw_dataset = pd.read_csv(dataset_path)
dataset=raw_dataset.copy()
dataset.pop("id")
#print (dataset)

y=dataset['diagnosis']
x=dataset.loc[:, dataset.columns != 'diagnosis']
x.drop(x.columns[[-1,]], axis=1, inplace=True)

le = preprocessing.LabelEncoder()
le.fit(y)
list(le.classes_)
y=le.transform(y)
y=pd.Series(y)
#print(y)

xTrain, xTest, yTrain, yTest=train_test_split(x, y, test_size = 0.3,random_state=0)

statistics1=xTrain.describe().transpose()
statistics2=xTest.describe().transpose()

print(statistics1['mean'])

norm_xTrain=(xTrain-statistics1['mean'])/statistics1['std']
norm_xTest=(xTest-statistics2['mean'])/statistics2['std']

print(norm_xTrain)

#model = Sequential()
#model.add(Dense(2,  # output dim is 2, one score per each class
#                activation='softmax',
#                kernel_regularizer=L1L2(l1=0.0, l2=0.1),
#                input_dim=len(feature_vector)))


def build_model1():
  model = keras.Sequential([
    layers.Dense(1, activation='sigmoid', input_shape=[len(xTrain.keys())])
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.1)

  model.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
  return model

model1 = build_model1()
#---------------------------------------------------------------------------
def build_model2():
  model = keras.Sequential([
    layers.Dense(1, activation='sigmoid', input_shape=[len(xTrain.keys())])
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.01)

  model.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
  return model

model2 = build_model2()
#---------------------------------------------------------------------------
def build_model3():
  model = keras.Sequential([
    layers.Dense(1, activation='sigmoid', input_shape=[len(xTrain.keys())])
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
  return model

model3 = build_model3()
#---------------------------------------------------------------------------
def build_model4():
  model = keras.Sequential([
    layers.Dense(1, activation='sigmoid', input_shape=[len(xTrain.keys())])
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.0001)

  model.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
  return model

model4 = build_model4()
#---------------------------------------------------------------------------
def build_model5():
  model = keras.Sequential([
    layers.Dense(1, activation='sigmoid', input_shape=[len(xTrain.keys())])
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.00001)

  model.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
  return model

model5 = build_model5()
#---------------------------------------------------------------------------
#model.summary()

#example_batch = norm_xTest[:10]
#example_result = model.predict(example_batch)
#print(example_result)

print(len(xTrain.keys()))
#############################################################################


class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history1 = model1.fit(
    norm_xTrain, yTrain,
    epochs=EPOCHS, validation_split = 0.2, verbose=0,
    callbacks=[PrintDot()])

hist = pd.DataFrame(history1.history)
hist['epoch'] = history1.epoch
hist.tail()
#----------------------------------------------------------------------
history2 = model2.fit(
    norm_xTrain, yTrain,
    epochs=EPOCHS, validation_split = 0.2, verbose=0,
    callbacks=[PrintDot()])

hist = pd.DataFrame(history2.history)
hist['epoch'] = history2.epoch
hist.tail()
#----------------------------------------------------------------------
history3 = model3.fit(
    norm_xTrain, yTrain,
    epochs=EPOCHS, validation_split = 0.2, verbose=0,
    callbacks=[PrintDot()])

hist = pd.DataFrame(history3.history)
hist['epoch'] = history3.epoch
hist.tail()
#----------------------------------------------------------------------
history4 = model4.fit(
    norm_xTrain, yTrain,
    epochs=EPOCHS, validation_split = 0.2, verbose=0,
    callbacks=[PrintDot()])

hist = pd.DataFrame(history4.history)
hist['epoch'] = history4.epoch
hist.tail()
#----------------------------------------------------------------------
history5 = model5.fit(
    norm_xTrain, yTrain,
    epochs=EPOCHS, validation_split = 0.2, verbose=0,
    callbacks=[PrintDot()])

hist = pd.DataFrame(history5.history)
hist['epoch'] = history5.epoch
hist.tail()
#----------------------------------------------------------------------
print("")

#print(hist)

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy error')
    plt.plot(hist['epoch'], hist['acc'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_acc'],
             label = 'Val Error')
    plt.ylim([0.6,1.2])
    plt.legend()
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist['epoch'], hist['loss'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_loss'],
             label = 'Val Error')
    plt.ylim([0,1])
    plt.legend()

plot_history(history1)
plot_history(history2)
plot_history(history3)
plot_history(history4)
plot_history(history5)

test_predictions = model1.predict(norm_xTest).flatten()

plt.scatter(yTest, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
