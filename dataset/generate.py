# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import imutils
import time
import cv2
import os

def alphaBlend(img1, img2, mask):
    """ alphaBlend img1 and img 2 (of CV_8UC3) with mask (CV_8UC1 or CV_8UC3)
    """
    if mask.ndim==3 and mask.shape[-1] == 3:
        alpha = mask/255.0
    else:
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)/255.0
    blended = cv2.convertScaleAbs(img1*(1-alpha) + img2*alpha)
    return blended

def perlin(x, y, seed = -1):
	# permutation table
	if(seed != -1):
		np.random.seed(seed)
	p = np.arange(256,dtype=int)
	np.random.shuffle(p)
	p = np.stack([p,p]).flatten()
	# coordinates of the top-left
	xi = x.astype(int)
	yi = y.astype(int)
	# internal coordinates
	xf = x - xi
	yf = y - yi
	# fade factors
	u = fade(xf)
	v = fade(yf)
	# noise components
	n00 = gradient(p[p[xi]+yi],xf,yf)
	n01 = gradient(p[p[xi]+yi+1],xf,yf-1)
	n11 = gradient(p[p[xi+1]+yi+1],xf-1,yf-1)
	n10 = gradient(p[p[xi+1]+yi],xf-1,yf)
	# combine noises
	x1 = lerp(n00,n10,u)
	x2 = lerp(n01,n11,u) # FIX1: I was using n10 instead of n01
	return lerp(x1,x2,v) # FIX2: I also had to reverse x1 and x2 here

def lerp(a, b, x):
	"linear interpolation"
	return a + x * (b-a)

def fade(t):
	"6t^5 - 15t^4 + 10t^3"
	return 6 * t**5 - 15 * t**4 + 10 * t**3

def gradient(h, x, y):
	"grad converts h to the right gradient vector and return the dot product with (x,y)"
	vectors = np.array([[0,1],[0,-1],[1,0],[-1,0]])
	g = vectors[h%4]
	return g[:,:,0] * x + g[:,:,1] * y

def generate():
	kernel = np.ones((3,3),np.uint8)

	blank = np.zeros((256,256,3), np.uint8)
	blank[:] = (255,255,255)

	arquivos = os.listdir("images/originais")
	i = 0
	for file in arquivos:
		i += 1
		print("\n" + str(i) + " / " + str(len(arquivos)))

		total_clean_path = "images/clean/rotate_0_" + file
		total_original_path = "images/originais/" + file

		if(not(os.path.exists(total_clean_path))):
			
			first_name = file.split('_')[0]
			if(first_name == "rotate"):

				print("The last file has a name beggining with 'rotate' and there is no corresponding file in images/hazed/ folder")
				opt = input("Delete " + total_clean_path+" ?[y/n]")

				while (opt != 'y' and opt != 'n'):
					opt = input("Delete " + total_clean_path+" ?[y/n]")

				if(opt == 'y'):
					print(total_clean_path + " will be excluded")
					os.remove(total_clean_path)
				else:
					print(total_clean_path + " will be keeped in images/clean/ folder, but no noise neither rotation will be made with it")
					print("At the end of dataset setup, take a look at the number of files from images/clean/ and images/hazed/")
					
				time.sleep(3)
			else:
				print("quem passou " + file)
				normal_img = cv2.imread(total_original_path)
		
				images = rotate(normal_img)
		
				hazing(images, kernel, blank, file)

def rotate(normal_img):

	images = list()

	images.append(normal_img)
	images.append(imutils.rotate_bound(normal_img, 90))
	images.append(imutils.rotate_bound(normal_img, 180))
	images.append(imutils.rotate_bound(normal_img, 270))

	flip_img = cv2.flip(normal_img, 1)
	images.append(flip_img)
	images.append(imutils.rotate_bound(flip_img, 90))
	images.append(imutils.rotate_bound(flip_img, 180))
	images.append(imutils.rotate_bound(flip_img, 270))

	return images

def hazing(images, kernel, blank, fileName):
	i = 0
	for img in images:
		lin = np.linspace(0,2 + np.random.random_integers(5), 256, endpoint=True)
		x,y = np.meshgrid(lin, lin) # FIX3: I thought I had to invert x and y here but it was a mistake

		noise = perlin(x, y)
		
		noise = noise + 1
		noise = noise[:] / 2
		
		noise = noise[:] - 0.4
		noise[noise < 0] = 0
		noise = noise[:] * 400 #255
		noise = noise.astype(np.uint8)

		mask = noise

		imgName = "rotate_" + str(i) + "_" + fileName

		res = alphaBlend(img, blank, mask)
		
		#The images with no haze will be saved in the clean folder
		cv2.imwrite("images/clean/" + imgName, img)
		#The images with haze will be saved in the hazed folder
		cv2.imwrite("images/hazed/" + imgName, res)

		i += 1