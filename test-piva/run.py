import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, Input
from tensorflow.keras.callbacks import ModelCheckpoint
import os



x_test = tf.keras.preprocessing.image_dataset_from_directory(
  "C:\\Users\\Rodrigo\\Desktop\\Rodrigo\\USP\\Zenith\\Rede Neural\\GitHub\\Visao\\images\\clean",
  label_mode = None,
  class_names = None,
  color_mode = 'rgb',
  batch_size = 32,
  image_size = (256, 256),
  shuffle = False,
  validation_split = 0.1,
  subset = "training"
)
x_test_noisy = tf.keras.preprocessing.image_dataset_from_directory(
  "C:\\Users\\Rodrigo\\Desktop\\Rodrigo\\USP\\Zenith\\Rede Neural\\GitHub\\Visao\\images\\hazed",
  label_mode = None,
  class_names = None,
  color_mode = 'rgb',
  batch_size = 32,
  image_size = (256, 256),
  shuffle = False,
  validation_split = 0.1,
  subset = "training"
)

test_haze = []
for images in x_test_noisy.take(1):
    for i in range(32):
        img = images[i].numpy()
        img = img / 255
        img = img.astype(np.float32)
        test_haze.append(img)

test = []
for images in x_test.take(1):
    for i in range(32):
        img = images[i].numpy()
        img = img / 255
        img = img.astype(np.float32)
        test.append(img)

test_haze = np.array(test_haze)
test = np.array(test)

# Load Model
model = tf.keras.models.load_model("test-piva\\models")

encoded_imgs=model.encoder(test_haze).numpy()
decoded_imgs=model.decoder(encoded_imgs)

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