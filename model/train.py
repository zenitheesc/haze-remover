import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
from .denoise_model import Denoise
from .load_images import numpy_array_images, dataset_split
import os

def train(dataset_path="../images/", epochs=30, test_split=0.2, model_path="../models"):

    cleanFiles, hazedFiles, fileNames = numpy_array_images(dataset_path)
    train, test = dataset_split(cleanFiles, hazedFiles, test_split)

    print(train[0].shape)
    print(test[0].shape)
    print(len(fileNames))

    autoencoder = load_model(model_path)
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    autoencoder.fit(train[0], train[1],
                    epochs=epochs,
                    shuffle=True,
                    validation_data=(test[0], test[1]))
    
    save_model(autoencoder, model_path)
    
    return fileNames, test, autoencoder



def save_model(autoencoder, model_path="../models"):

    if(not os.path.exists(model_path)):
        os.mkdir(model_path)

    autoencoder.save(model_path)

def load_model(model_path="../models"):

    if(not os.path.exists(model_path)):
        #os.mkdir(model_path)
        return Denoise()
    
    restored_keras_model = tf.keras.models.load_model(model_path)

    return restored_keras_model