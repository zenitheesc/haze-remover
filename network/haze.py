import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, Input, Dense, MaxPooling2D, UpSampling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import losses

#Trains the network on the available images

#Neural Network
class HazeRemover(tf.keras.Model): 
  def __init__(self):

    super(HazeRemover, self).__init__() 

    self.encoder = tf.keras.Sequential([ 
      Input(shape=(256, 256, 3)),
      Conv2D(96, (3,3), activation='relu', padding='same', strides=2),
      MaxPooling2D((2,2), padding='same'),
      #Conv2D(64, (3,3), activation='relu', padding='same', strides=2),
      #MaxPooling2D((2,2), padding='same'),
      Conv2D(48, (3,3), activation='relu', padding='same', strides=2),
      Dense(9, activation='relu')])
    self.decoder = tf.keras.Sequential([ 
      Conv2DTranspose(48, kernel_size=3, strides=2, activation='relu', padding='same'),
      UpSampling2D((2,2)),
      Conv2DTranspose(96, kernel_size=3, strides=2, activation='relu', padding='same'),
      Conv2D(3, kernel_size=(3,3), activation='sigmoid', padding='same')])

  def call(self, x): 
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

#Divides uint8 images by 255 and converts to float32
def process(image):
    image = tf.cast(image/255. ,tf.float32)
    return image

BatchSize = 24

#Training dataset of clean images
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
#Validation dataset of clean images
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

#Training dataset of hazed images
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
#Validation dataset of hazed images
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

#Changes image to float32
train = x_train.map(process)
train_noisy = x_train_noisy.map(process)
test = x_test.map(process)
test_noisy = x_test_noisy.map(process)

#Joins the pairs of datasets
dataset_train = tf.data.Dataset.zip( (train_noisy, train) )
dataset_val = tf.data.Dataset.zip( (test_noisy, test) )

#shuffles the datasets
dataset_train = dataset_train.shuffle()
dataset_val = dataset_val.shuffle()

#Saves checkpoints
checkpointPath = "network\\checkpoints\\model_{epoch:02d}_{val_loss:.5f}.hdf5"
checkpoint = ModelCheckpoint(
  checkpointPath,
  verbose = 1,
  monitor = "val_loss",
  save_best_only = False,
  save_weights_only = False,
  mode = 'auto',
  save_freq = "epoch"
)

#Loads the model, if available. Creates a new one if not
try:
  autoencoder = tf.keras.models.load_model("network\\models")
  print("###########################")
  print("#########Carregado#########")
  print("###########################")
except:
  #Creates and compiles the network
  autoencoder = HazeRemover()
  autoencoder.compile(optimizer='adam',
                      loss = ['mse', 'binary_crossentropy'],
                      loss_weights = [12.5, 2.5])

  autoencoder.build((None,256,256,3))

#Prints the summary of the network
autoencoder.encoder.summary()
autoencoder.decoder.summary()
autoencoder.summary()

#Trains the model
autoencoder.fit(
  dataset_train,
  epochs = 15,
  shuffle = False,
  validation_data = dataset_val,
  callbacks = [checkpoint]
)

#Saves the network
autoencoder.save("network\\models")

#Loads a batch of images
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

#Converts the images to pass it through the model
test_haze = np.array(test_haze)
test = np.array(test)

#Applies model to the images
encoded_imgs=autoencoder.encoder(test_haze).numpy()
decoded_imgs=autoencoder.decoder(encoded_imgs)

#Shows results
n = 8
plt.figure(figsize=(20, 7))
plt.gray()
for i in range(n): 
  #Display original + noise 
  bx = plt.subplot(3, n, i + 1) 
  plt.title("original + noise") 
  plt.imshow(tf.squeeze(test_haze[i])) 
  bx.get_xaxis().set_visible(False) 
  bx.get_yaxis().set_visible(False) 
  
  #Display reconstruction 
  cx = plt.subplot(3, n, i + n + 1) 
  plt.title("reconstructed") 
  plt.imshow(tf.squeeze(decoded_imgs[i])) 
  cx.get_xaxis().set_visible(False) 
  cx.get_yaxis().set_visible(False) 
  
  #Display original 
  ax = plt.subplot(3, n, i + 2*n + 1) 
  plt.title("original") 
  plt.imshow(tf.squeeze(test[i])) 
  ax.get_xaxis().set_visible(False) 
  ax.get_yaxis().set_visible(False) 

plt.show()
print("Finished")