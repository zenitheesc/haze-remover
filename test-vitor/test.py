import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
import matplotlib.pyplot as plt

def fileNames(cleanPath="/content/images/clean/"):
  cleanFiles = []
  hazedFiles = []
  
  i = 0
  for file in os.listdir(cleanPath):
      
    clean = cv2.imread(cleanPath + file)
    clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
    #clean = clean/255.0
    
    cleanFiles.append(clean)
    hazed = cv2.imread("/content/images/hazed/" + file)
    hazed = cv2.cvtColor(hazed, cv2.COLOR_BGR2RGB)
    #hazed = hazed/255.0
    hazedFiles.append(hazed)
    print(i)
    i += 1
    #if(i == 600):
        #break
  return np.array(cleanFiles), np.array(hazedFiles)

a, b = fileNames()

a = a.astype('float32') / 255.
b = b.astype('float32') / 255.
print(a[0])
x_train = a[0:352]
x_test = a[352:704]
x_train_hazed = b[0:352]
x_test_hazed = b[352:704]

print(x_train.shape)
print(x_test.shape)
print(x_train_hazed.shape)
print(x_test_hazed.shape)


class Denoise(Model):
  def __init__(self):
    super(Denoise, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(256, 256, 3)),
      layers.Conv2D(72, (5,5), activation='relu', padding='same', strides=2),
      layers.Conv2D(52, (5,5), activation='relu', padding='same', strides=2),
      layers.Conv2D(32, (5,5), activation='relu', padding='same', strides=2)])
    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(32, kernel_size=5, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(52, kernel_size=5, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(72, kernel_size=5, strides=2, activation='relu', padding='same'),
      layers.Conv2D(3, kernel_size=(5,5), activation='sigmoid', padding='same')])
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Denoise()
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

autoencoder.fit(x_train_hazed, x_train,
                epochs=80,
                shuffle=True,
                validation_data=(x_test_hazed, x_test))

encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

n = 1
plt.figure(figsize=(20, 4))
for i in range(n):

  # display original + noise
  ax = plt.subplot(2, n, i + 1)
  plt.title("original + noise")
  #cv2.imwrite("/content/result/"+"hazed_"+str(i), x_test_hazed[i])
  #plt.imshow(tf.squeeze(x_test_hazed[i]))
  plt.imshow(tf.squeeze(hazedFile))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  
  # display reconstruction
  bx = plt.subplot(2, n, i + n + 1)
  plt.title("reconstructed")
  #cv2.imwrite("/content/result/"+"reconstructed_"+str(i), decoded_imgs[i])
  #plt.imshow(tf.squeeze(decoded_imgs[i]))
  plt.imshow(tf.squeeze(decoded))
  plt.gray()
  bx.get_xaxis().set_visible(False)
  bx.get_yaxis().set_visible(False)


plt.show()