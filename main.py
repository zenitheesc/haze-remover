import dataset
from dataset import google_mps_api
from dataset import generate
import os

if __name__ == "__main__":
    
    if(not (os.path.exists("hazed"))):
        os.mkdir("hazed")

    if(not (os.path.exists("images"))):
        os.mkdir("images")

    google_mps_api.imgDownload()
    generate.generate()