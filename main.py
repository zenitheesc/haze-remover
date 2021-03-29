from dataset import google_mps_api, generate
from register import check_dataset_used, updating_register
from model import train
import matplotlib.pyplot as plt
import tensorflow as tf
import os

if __name__ == "__main__":
    
    if(not (os.path.exists("images"))):
        os.mkdir("images")
        os.mkdir("images/hazed")
        os.mkdir("images/clean")
        os.mkdir("images/originais")
    else:
        if(not (os.path.exists("images/hazed"))):	
            os.mkdir("images/hazed")
        if(not (os.path.exists("images/clean"))):
            os.mkdir("images/clean")
        if(not (os.path.exists("images/originais"))):
            os.mkdir("images/originais")
        
    #google_mps_api.imgDownload(centerLat=-21.946989, centerLon=-47.732020)
    #google_mps_api.imgDownload(centerLat=-11.074495, centerLon=-42.051459)
    #google_mps_api.imgDownload(centerLat=-10.198966, centerLon=-46.872713)
    #google_mps_api.imgDownload(centerLat=-16.618904, centerLon=-70.077475)
    #google_mps_api.imgDownload(centerLat=-23.669320, centerLon=-46.559369)
    #google_mps_api.imgDownload(centerLat=38.080750, centerLon=81.888615)
    #generate.generate()

    register_path="./coordinates.txt"
    dataset_path="./images/"
    model_path="./models"

    check_dataset_used.clean_images_used(register_path, dataset_path)#Delete the images already used to train the model, this is made by using coordinates.txt
    fileNames, test, autoencoder = train.train(dataset_path, 5, 0.2, model_path)
    updating_register.new_coordinates(fileNames, register_path)

    restored_keras_model = tf.keras.models.load_model(model_path)

    encoded_imgs = restored_keras_model.encoder(test[0]).numpy()
    decoded_imgs = restored_keras_model.decoder(encoded_imgs).numpy()


    n = 10
    plt.figure(figsize=(20, 7))
    for i in range(n): 
        # display original + noise 
        bx = plt.subplot(3, n, i + 1) 
        plt.title("original + noise") 
        plt.imshow(tf.squeeze(test[0][i])) 
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
        plt.imshow(tf.squeeze(test[1][i])) 
        ax.get_xaxis().set_visible(False) 
        ax.get_yaxis().set_visible(False) 


    plt.show()
#-11.0744952,-42.0514597
#-10.1989664,-46.8727137
#-16.6189048,-70.077475
#24.4984753,5.458369
#38.0807507,81.8886153
#-23.6693206,-46.5593699 sbc