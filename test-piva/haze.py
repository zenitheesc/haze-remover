import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import losses

os.chdir("C:\\Users\\Rodrigo\\Desktop\\Rodrigo\\USP\\Zenith\\Rede Neural\\GitHub\\Visao")

class HazeRemover(tf.keras.Model): 
  def __init__(self):

    super(HazeRemover, self).__init__() 

    self.encoder = tf.keras.Sequential([ 
      Input(shape=(256, 256, 3)),
      Conv2D(120, (3,3), activation='relu', padding='same', strides=2),
      Conv2D(64, (3,3), activation='relu', padding='same', strides=2),
      Dense(24, activation='relu')])
    self.decoder = tf.keras.Sequential([ 
      Dense(24, activation='relu'),
      Conv2DTranspose(60, kernel_size=3, strides=2, activation='relu', padding='same'),
      Conv2DTranspose(100, kernel_size=3, strides=2, activation='relu', padding='same'),
      Conv2D(3, kernel_size=(3,3), activation='sigmoid', padding='same')])
  
  def call(self, x): 
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

def process(image):
    image = tf.cast(image/255. ,tf.float32)
    return image

BatchSize = 24

x_train = tf.keras.preprocessing.image_dataset_from_directory(
  "images\\clean",
  label_mode = None,
  class_names = None,
  color_mode = 'rgb',
  batch_size = BatchSize,
  image_size = (256, 256),
  shuffle = False,
  validation_split = 0.1,
  subset = "training"
)
x_test = tf.keras.preprocessing.image_dataset_from_directory(
  "images\\clean",
  label_mode = None,
  class_names = None,
  color_mode = 'rgb',
  batch_size = BatchSize,
  image_size = (256, 256),
  shuffle = False,
  validation_split = 0.1,
  subset = "validation"
)

x_train_noisy = tf.keras.preprocessing.image_dataset_from_directory(
  "images\\hazed",
  label_mode = None,
  class_names = None,
  color_mode = 'rgb',
  batch_size = BatchSize,
  image_size = (256, 256),
  shuffle = False,
  validation_split = 0.1,
  subset = "training"
)
x_test_noisy = tf.keras.preprocessing.image_dataset_from_directory(
  "images\\hazed",
  label_mode = None,
  class_names = None,
  color_mode = 'rgb',
  batch_size = BatchSize,
  image_size = (256, 256),
  shuffle = False,
  validation_split = 0.1,
  subset = "validation"
)

train = x_train.map(process)
train_noisy = x_train_noisy.map(process)
test = x_test.map(process)
test_noisy = x_test_noisy.map(process)

dataset_train = tf.data.Dataset.zip( (train, train_noisy) )
dataset_val = tf.data.Dataset.zip( (test, test_noisy) )

checkpointPath = "test-piva\\checkpoints\\model_{epoch:02d}_{val_loss:.5f}.hdf5"
checkpoint = ModelCheckpoint(
  checkpointPath,
  verbose = 1,
  monitor = "val_loss",
  save_best_only = False,
  save_weights_only = False,
  mode = 'auto',
  save_freq = "epoch"
)

autoencoder = HazeRemover()
autoencoder.compile(optimizer='adam',
                    loss = ['mse', 'binary_crossentropy'],
                    loss_weights = [12.5, 2.5])

autoencoder.fit(
  dataset_train,
  epochs = 50,
  shuffle = False,
  validation_data = dataset_val,
  callbacks = [checkpoint]
)

autoencoder.save("test-piva\\models")

test_haze = []
for images in x_test_noisy.take(1):
    for i in range(BatchSize):
        img = images[i].numpy()
        img = img / 255
        img = img.astype(np.float32)
        test_haze.append(img)

test = []
for images in x_test.take(1):
    for i in range(BatchSize):
        img = images[i].numpy()
        img = img / 255
        img = img.astype(np.float32)
        test.append(img)

test_haze = np.array(test_haze)
test = np.array(test)

encoded_imgs=autoencoder.encoder(test_haze).numpy()
decoded_imgs=autoencoder.decoder(encoded_imgs)

n = 8
plt.figure(figsize=(20, 7))
plt.gray()
for i in range(n): 
  # display original + noise 
  bx = plt.subplot(3, n, i + 1) 
  plt.title("original + noise") 
  plt.imshow(tf.squeeze(test_haze[i])) 
  bx.get_xaxis().set_visible(False) 
  bx.get_yaxis().set_visible(False) 
  
  # display reconstruction 
  cx = plt.subplot(3, n, i + n + 1) 
  plt.title("reconstructed") 
  plt.imshow(tf.squeeze(decoded_imgs[i])) 
  cx.get_xaxis().set_visible(False) 
  cx.get_yaxis().set_visible(False) 
  
  # display original 
  ax = plt.subplot(3, n, i + 2*n + 1) 
  plt.title("original") 
  plt.imshow(tf.squeeze(test[i])) 
  ax.get_xaxis().set_visible(False) 
  ax.get_yaxis().set_visible(False) 

plt.show()
print("Finished")