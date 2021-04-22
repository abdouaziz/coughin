import tensorflow as tf 
import os 
import pandas as pd 
import matplotlib.pyplot as plt 
import json 
import librosa , librosa.display
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow import keras


file ="data.json"

def load_data(filename=file):

    with open(filename , "r") as f :
        data = json.load(f)

    inputs = np.array(data["mffc"])
    targets =  np.array(data["labels"])

    return inputs , targets

def plot_graph(history , string):
  plt.plot(history.history[string])
  plt.plot(history.history["val_"+string])
  plt.xlabel(string)
  plt.ylabel("val_"+string)
  plt.legend([string , "val_"+string])
  plt.show()





if __name__ =="__main__":

    inputs , targets = load_data()

    inputs_train , inputs_test , targets_train , targets_test = train_test_split(inputs , targets , test_size =0.3)

    model = keras.Sequential([

        # input layer
        keras.layers.Flatten(input_shape=(inputs_train.shape[1], inputs_train.shape[2])),

        # 1st dense layer
        keras.layers.Dense(512, activation='relu'),

        # 2nd dense layer
        keras.layers.Dense(256, activation='relu'),

        # 3rd dense layer
        keras.layers.Dense(64, activation='relu'),

        # output layer
        keras.layers.Dense(1, activation='softmax')
    ])

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # train model
    history = model.fit(inputs_train, targets_train, validation_data=(inputs_test, targets_test), batch_size=32, epochs=50)


    plot_graph(history , "accuracy")
    plot_graph(history , "loss")



    
