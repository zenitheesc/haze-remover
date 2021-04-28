import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, Input
from tensorflow.keras.callbacks import ModelCheckpoint
import os

#Runs model on validation images

BatchSize = 32

#Loads images into validation subset
x_test = tf.keras.preprocessing.image_dataset_from_directory(
	"images\\clean",
	label_mode = None,
	class_names = None,
	color_mode = 'rgb',
	batch_size = BatchSize,
	image_size = (256, 256),
	shuffle = True,
	seed = 1771,
	validation_split = 0.1,
	subset = "validation"
)
x_test_noisy = tf.keras.preprocessing.image_dataset_from_directory(
	"images\\hazed",
	label_mode = None,
	class_names = None,
	color_mode = 'rgb',
	batch_size = BatchSize,
	image_size = (256, 256),
	shuffle = True,
	seed = 1771,
	validation_split = 0.1,
	subset = "validation"
)

#Converts images to float32
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

#Loads Model
model = tf.keras.models.load_model("network\\models")

#Applies Model
encoded_imgs=model.encoder(test_haze).numpy()
decoded_imgs=model.decoder(encoded_imgs)

offset = 2
n = 8

#Shows results
plt.figure(figsize=(20, 7))
plt.subplots_adjust(left=0.01, right=0.99, bottom=0, top=1, wspace=0.05, hspace=0)
plt.gray()
for i in range(n):
	# display original + noise 
	bx = plt.subplot(3, n, i + 1) 
	plt.title("original + noise") 
	plt.imshow(tf.squeeze(test_haze[i + offset])) 
	bx.get_xaxis().set_visible(False) 
	bx.get_yaxis().set_visible(False) 

	# display reconstruction 
	cx = plt.subplot(3, n, i + n + 1) 
	plt.title("reconstructed") 
	plt.imshow(tf.squeeze(decoded_imgs[i + offset])) 
	cx.get_xaxis().set_visible(False) 
	cx.get_yaxis().set_visible(False) 

	# display original 
	ax = plt.subplot(3, n, i + 2*n + 1) 
	plt.title("original") 
	plt.imshow(tf.squeeze(test[i + offset])) 
	ax.get_xaxis().set_visible(False) 
	ax.get_yaxis().set_visible(False) 

plt.show()
print("Finished")