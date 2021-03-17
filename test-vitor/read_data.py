import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses           

class Denoise(Model):
    def __init__(self):
        super(Denoise, self).__init__()
        self.encoder = tf.keras.Sequential([
          layers.Input(shape=(256, 256, 3)), 
          layers.Conv2D(16, (3,3), activation='relu', padding='same', strides=2),
          layers.Conv2D(8, (3,3), activation='relu', padding='same', strides=2)])
        self.decoder = tf.keras.Sequential([
          layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
          layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
          layers.Conv2D(1, kernel_size=(3,3), activation='sigmoid', padding='same')])
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def read_data(cleanPath="../images/", noisePath="../hazed/", nTrain=50):
    # x_train
    # x_test
    # x_train_noise
    # x_test_noise
    

    x_train = tf.keras.preprocessing.image_dataset_from_directory(
    cleanPath,
    validation_split=0.2,
    subset="training",
    color_mode="rgb",
    seed=1,
    image_size=(256, 256),
    batch_size=64)

    x_test = tf.keras.preprocessing.image_dataset_from_directory(
    cleanPath,
    validation_split=0.2,
    subset="validation",
    color_mode="rgb",
    seed=1,
    image_size=(256, 256),
    batch_size=64)

    class_names = x_train.class_names
    print(class_names)

    print("train size: ", x_train.cardinality().numpy())
    print("test size: ", x_test.cardinality().numpy())

    

    class Denoise(Model):
      def __init__(self):
        super(Denoise, self).__init__()
        self.encoder = tf.keras.Sequential([
          layers.Input(shape=(256, 256, 3)), 
          layers.Conv2D(16, (3,3), activation='relu', padding='same', strides=2),
          layers.Conv2D(8, (3,3), activation='relu', padding='same', strides=2)])
        self.decoder = tf.keras.Sequential([
          layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
          layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
          layers.Conv2D(1, kernel_size=(3,3), activation='sigmoid', padding='same')])

      def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    autoencoder = Denoise()
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    
    autoencoder.fit(x_train,
        epochs=2,
        shuffle=True,
        validation_data=(x_test))
    #normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    #normalized_ds = x_train.map(lambda x, y: (normalization_layer(x), y))
    #image_batch, labels_batch = next(iter(normalized_ds))
    
    #autoencoder = Denoise()
    #autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())   
    #autoencoder.fit(x_train,
    #  epochs=1,
    #  shuffle=True,
    #  validation_data=(x_test))
    

    #first_image = image_batch[0]
    #
    #print(np.min(first_image), np.max(first_image))
#https://www.tensorflow.org/tutorials/load_data/images
#https://www.tensorflow.org/tutorials/generative/autoencoder



read_data()