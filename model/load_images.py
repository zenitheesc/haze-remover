from sys import exit
import os
import numpy as np
import cv2


def numpy_array_images(dataset_path="../images/"):
    cleanFiles = []
    hazedFiles = []
    fileNames = []
    

    # Getting all memory using os.popen()
    total_memory, used_memory, free_memory = map(
    	int, os.popen('free -t -m').readlines()[-1].split()[1:])

    ram_before = round((used_memory/total_memory) * 100, 2)
    print("RAM memory % used before images load:", ram_before)

    for file in os.listdir(dataset_path + "clean/"):
          
        clean = cv2.imread(dataset_path + "clean/" + file)
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)

        cleanFiles.append(clean)

        hazed = cv2.imread(dataset_path + "hazed/" + file)
        hazed = cv2.cvtColor(hazed, cv2.COLOR_BGR2RGB)

        hazedFiles.append(hazed)

        fileNames.append(file)
        
        total_memory, used_memory, free_memory = map(
    	int, os.popen('free -t -m').readlines()[-1].split()[1:])

        ram_current = round((used_memory/total_memory) * 100, 2)

        if(ram_current - ram_before >= 2): # In my case, the increase of 2% in the use of ram was a good limit.
            break
    
    print("RAM memory % used after images load:", ram_current)

    cleanFiles = np.array(cleanFiles)
    hazedFiles = np.array(hazedFiles)

    cleanFiles = cleanFiles.astype('float32') / 255.
    hazedFiles = hazedFiles.astype('float32') / 255.
    
    return cleanFiles, hazedFiles, fileNames

def dataset_split(cleanFiles, hazedFiles, test_split=0.2):
    
    if(cleanFiles. shape != hazedFiles.shape):
        exit("clean and hazed images dataset has differents shapes")

    if(test_split < 0 or test_split > 1):
        exit("test_split value must be between 0 and 1")

    total = cleanFiles.shape[0]

    n_train = int((1 - test_split) * total)

    x_train = cleanFiles[0:n_train]
    x_test = cleanFiles[n_train:total]

    x_train_hazed = hazedFiles[0:n_train]
    x_test_hazed = hazedFiles[n_train:total]

    train = (x_train_hazed, x_train)
    test = (x_test_hazed, x_test)

    return train, test
