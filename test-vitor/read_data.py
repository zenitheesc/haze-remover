import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt           

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


    


    #normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    #normalized_ds = x_train.map(lambda x, y: (normalization_layer(x), y))
    #image_batch, labels_batch = next(iter(normalized_ds))
#
    #first_image = image_batch[0]
    #
    #print(np.min(first_image), np.max(first_image))

read_data()
#https://www.tensorflow.org/tutorials/load_data/images
#https://www.tensorflow.org/tutorials/generative/autoencoder