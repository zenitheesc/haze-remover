import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses

def fileNames(filePath="../images/clean/"):
    fname = []
    
    i = 0
    for file in os.listdir(filePath):
        fname.append(file)
        #print(fname[i])
        i += 1
    print("Files names loaded into fname list")
    print(str(i) + " file names were loaded")
    
    return fname

def addPath(fileName, filePath):
    return filePath + fileName

def imageRead(file_path: tf.Tensor):
    print(file_path)
    img = tf.io.read_file(file_path)

    img = tf.image.decode_png(img, channels=3, dtype=tf.dtypes.uint8) 
    #convert unit8 tensor to floats in the [0,1]range
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

filesList = fileNames()

fnameClean = [addPath(x, "../images/clean/") for x in filesList]
fnameHazed = [addPath(x, "../images/hazed/") for x in filesList]

number_of_selected_samples=10
filelistClean_ds = tf.data.Dataset.from_tensor_slices(fnameClean[:number_of_selected_samples])
filelistHazed_ds = tf.data.Dataset.from_tensor_slices(fnameHazed[:number_of_selected_samples])


#splitting datasets
train_ratio = 0.80
ds_size= filelistClean_ds.cardinality().numpy() #filelistClean_ds and filelistHazed_ds must have the same number of files, so ds_size is the same for both

ds_train = filelistClean_ds.take(ds_size*train_ratio)
ds_test = filelistClean_ds.skip(ds_size*train_ratio)

ds_train_hazed = filelistHazed_ds.take(ds_size*train_ratio)
ds_test_hazed = filelistHazed_ds.skip(ds_size*train_ratio)

#for a in ds_train.take(3):
#  fname= a.numpy().decode("utf-8")
#  print(fname)
#
#for a in ds_train_hazed.take(3):
#  fname= a.numpy().decode("utf-8")
#  print(fname)
#this is a test to see if both has the same names


#print(list(ds_train.as_numpy_iterator()))
ds_train = ds_train.map(lambda x: tf.py_function(func=imageRead,
          inp=[x], Tout=(tf.float32)),
          #num_parallel_calls=tf.data.AUTOTUNE,
          deterministic=False)
ds_train.prefetch(ds_size-ds_size*train_ratio)

ds_test = ds_test.map(lambda x: tf.py_function(func=imageRead,
          inp=[x], Tout=(tf.float32)),
          #num_parallel_calls=tf.data.AUTOTUNE,
          deterministic=False)
ds_test.prefetch(ds_size-ds_size*train_ratio)

ds_train_hazed = ds_train_hazed.map(lambda x: tf.py_function(func=imageRead,
          inp=[x], Tout=(tf.float32)),
          #num_parallel_calls=tf.data.AUTOTUNE,
          deterministic=False)
ds_train_hazed.prefetch(ds_size-ds_size*train_ratio)

ds_test_hazed = ds_test_hazed.map(lambda x: tf.py_function(func=imageRead,
          inp=[x], Tout=(tf.float32)),
          #num_parallel_calls=tf.data.AUTOTUNE,
          deterministic=False)
ds_test_hazed.prefetch(ds_size-ds_size*train_ratio)

#ds_train = ds_train.shuffle(100).batch(64)
#ds_test = ds_test.shuffle(100).batch(64)
#ds_train_hazed = ds_train_hazed.batch(64)
#ds_test_hazed = ds_test_hazed.batch(64)
#print(type(ds_train))
#print(type(ds_train_hazed))

for image in ds_test_hazed.take(1):
    print("Image shape: ", image.shape)


class Denoise(Model):
    def __init__(self):
        super(Denoise, self).__init__()
        self.encoder = tf.keras.Sequential([
          layers.Input(shape=(28, 28, 1)), 
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

autoencoder.fit(ds_train_hazed, ds_train,
                epochs=10,
                shuffle=True,
                validation_data=(ds_test_hazed, ds_test))
