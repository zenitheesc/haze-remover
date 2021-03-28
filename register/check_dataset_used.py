import os
from .already_used import files_already_used

def clean_images_used(register_path="./coordinates.txt", dataset_path="./images/"): #This function remove from the dataset the images that were already trained in the autoencoder

    clean_incongruities(dataset_path)

    already_used = files_already_used(register_path)
    
    for file_already_trained in already_used:

        hazed_path = dataset_path + "hazed/" + file_already_trained
        clean_path = dataset_path + "clean/" + file_already_trained
        original_path = dataset_path + "originais/" + file_already_trained

        if(os.path.exists(hazed_path)):

            print("The file " + file_already_trained + " was already used to train de AI")
            print("It will be removed from data set")

            os.remove(hazed_path)
            os.remove(clean_path)

        if(os.path.exists(original_path)):
            os.remove(original_path)

def clean_incongruities(dataset_path="./images/"):

    hazed_path = dataset_path + "hazed/"
    clean_path = dataset_path + "clean/"
    original_path = dataset_path + "originais/"

    for file in os.listdir(clean_path):

        if(not(os.path.exists(hazed_path + file))):
    
            print("File " + file + " found in " + clean_path + " but not in " + hazed_path)
            print("Removing " + clean_path + file)

            os.remove(clean_path + file)

            if(os.path.exists(original_path + file)):
                os.remove(original_path + file)

    for file in os.listdir(hazed_path):

        if(not(os.path.exists(clean_path + file))):

            print("File " + file + " found in " + hazed_path + " but not in " + clean_path)
            print("Removing " + hazed_path + file)

            os.remove(hazed_path + file)

            if(os.path.exists(original_path + file)):
                os.remove(original_path + file)



