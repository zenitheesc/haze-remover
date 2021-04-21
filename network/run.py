import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os

batchSize = 16

print(os.getcwd())

entrada = cv2.imread("network\\sat_test.png")
entrada = cv2.cvtColor(entrada,cv2.COLOR_BGR2RGB)

print(entrada.shape)

num_x = int(entrada.shape[0]/256)
num_y = int(entrada.shape[1]/256)

print(num_x,num_y)

crops = []
crops.append([])

for i in range(num_x):
	for j in range(num_y):
		if(i*(j+1)%batchSize == 0 and (i != 0 or j!= 0)):
			crops.append([])
		img = entrada[256*i:256*(i+1), 256*j:256*(j+1)]
		img = img / 255
		img = img.astype(np.float32)
		crops[-1].append(img)

np_crops = []
for i in range(len(crops)):
	np_crops.append(np.array(crops[i]))

# Load Model
model = tf.keras.models.load_model("network\\models_old")

encoded_imgs = list()
decoded_imgs = list()

for i in range(len(np_crops)):
	encoded_imgs.append(model.encoder(np_crops[i]).numpy())
	decoded_imgs.append(model.decoder(encoded_imgs[i]))

count = 0
resultado = np.zeros(entrada.shape,dtype = np.float32)

decoded = []
cropped = []

for res in crops:
	for img in res:
		cropped.append(img)

for res in decoded_imgs:
	for img in res:
		i = int(count/num_y)
		j = count%num_y
		resultado[256*i:256*(i+1), 256*j:256*(j+1)] = img
		decoded.append(img)
		count += 1

resultado = resultado * 255
resultado = resultado.astype(np.uint8)

offset = 150
n = 8

plt.figure(figsize=(20, 7))
plt.subplots_adjust(left=0.001, right=0.999, bottom=0, top=1, wspace=0.01, hspace=0)
plt.gray()
for i in range(n):
	# display reconstruction
	cx = plt.subplot(2, n, i + 1)
	plt.title("reconstructed")
	plt.imshow(tf.squeeze(decoded[i + offset]))
	cx.get_xaxis().set_visible(False)
	cx.get_yaxis().set_visible(False)

	# display original
	ax = plt.subplot(2, n, i + n + 1)
	plt.title("original")
	plt.imshow(tf.squeeze(cropped[i + offset]))
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

plt.show()

plt.figure(figsize=(20, 7))
plt.subplots_adjust(left=0.001, right=0.999, bottom=0, top=1, wspace=0.01, hspace=0)
plt.gray()
# display original + noise
bx = plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(tf.squeeze(entrada))
bx.get_xaxis().set_visible(False)
bx.get_yaxis().set_visible(False)

# display reconstruction
cx = plt.subplot(1, 2, 2)
plt.title("Reconstructed")
plt.imshow(tf.squeeze(resultado))
cx.get_xaxis().set_visible(False)
cx.get_yaxis().set_visible(False)

plt.show()
print("Finished")
