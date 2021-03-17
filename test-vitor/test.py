import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
import matplotlib.pyplot as plt 

def fileNames(filePath="../images/clean/"):
    cleanFiles = []
    hazedFiles = []
    
    i = 0
    for file in os.listdir(filePath):
        
        clean = cv2.imread(filePath + file)
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        #clean = clean/255.0
        
        cleanFiles.append(clean)

        hazed = cv2.imread("../images/hazed/" + file)
        hazed = cv2.cvtColor(hazed, cv2.COLOR_BGR2RGB)
        #hazed = hazed/255.0
        hazedFiles.append(hazed)

        print(i)
        i += 1
        if(i == 300):
            break
    
    
    return np.array(cleanFiles), np.array(hazedFiles)

a, b = fileNames()

a = a.astype('float32') / 255.
b = b.astype('float32') / 255.

x_train = a[0:150]
x_test = a[150:300]
x_train_hazed = b[0:150]
x_test_hazed = b[150:300]

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

autoencoder.fit(x_train_hazed, x_train,
                epochs=10,
                shuffle=True,
                validation_data=(x_test_hazed, x_test))

encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

n = 1
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(x_test[i])
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_imgs[i])
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()