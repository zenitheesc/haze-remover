import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, Input
from tensorflow.keras.callbacks import ModelCheckpoint

class HazeRemover(tf.keras.Model): 
  def __init__(self):

    super(HazeRemover, self).__init__() 

    self.encoder = tf.keras.Sequential([ 
      Input(shape=(256, 256, 3)), 
      Conv2D(128, (3,3), activation='relu', padding='same', strides=2), 
      Conv2D(64, (3,3), activation='relu', padding='same', strides=2),
      Conv2D(32, (3,3), activation='relu', padding='same', strides=2)])
    self.decoder = tf.keras.Sequential([ 
      Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same'), 
      Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='same'), 
      Conv2DTranspose(128, kernel_size=3, strides=2, activation='relu', padding='same'), 
      Conv2D(3, kernel_size=(3,3), activation='sigmoid', padding='same')]) 
  
  def call(self, x): 
    encoded = self.encoder(x) 
    decoded = self.decoder(encoded) 
    return decoded

def process(image):
    image = tf.cast(image/255. ,tf.float32)
    return image


x_train = tf.keras.preprocessing.image_dataset_from_directory(
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
x_test = tf.keras.preprocessing.image_dataset_from_directory(
  "C:\\Users\\Rodrigo\\Desktop\\Rodrigo\\USP\\Zenith\\Rede Neural\\GitHub\\Visao\\images\\clean",
  label_mode = None,
  class_names = None,
  color_mode = 'rgb',
  batch_size = 32,
  image_size = (256, 256),
  shuffle = False,
  validation_split = 0.1,
  subset = "validation"
)

x_train_noisy = tf.keras.preprocessing.image_dataset_from_directory(
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
x_test_noisy = tf.keras.preprocessing.image_dataset_from_directory(
  "C:\\Users\\Rodrigo\\Desktop\\Rodrigo\\USP\\Zenith\\Rede Neural\\GitHub\\Visao\\images\\hazed",
  label_mode = None,
  class_names = None,
  color_mode = 'rgb',
  batch_size = 32,
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


checkpoint = ModelCheckpoint(
  "best_model.hdf5",
  monitor = 'loss',
  verbose = 1,
  save_best_only = True,
  mode = 'auto',
  save_freq = 1
)

autoencoder = HazeRemover()
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(
  dataset_train,
  epochs = 1,
  shuffle = False,
  validation_data = dataset_val
)

autoencoder.save("test-piva\\models")

encoded_imgs=autoencoder.encoder(x_test_noisy).numpy()
decoded_imgs=autoencoder.decoder(encoded_imgs)

n = 10
plt.figure(figsize=(20, 7))
plt.gray()
for i in range(n): 
  # display original + noise 
  bx = plt.subplot(3, n, i + 1) 
  plt.title("original + noise") 
  plt.imshow(tf.squeeze(x_test_noisy[i])) 
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
  plt.imshow(tf.squeeze(x_test[i])) 
  ax.get_xaxis().set_visible(False) 
  ax.get_yaxis().set_visible(False) 

plt.show()
print("Finished")