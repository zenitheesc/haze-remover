import dataset
from dataset import google_mps_api
from dataset import generate
import os

if __name__ == "__main__":
    
    if(not (os.path.exists("images"))):
    	os.mkdir("images")
    	os.mkdir("images/hazed")
    	os.mkdir("images/clean")

    elif(not (os.path.exists("images/hazed"))):	
        os.mkdir("images/hazed")

    if(not (os.path.exists("images/clean"))):
        os.mkdir("images/clean")
        
    google_mps_api.imgDownload(centerLat=-21.946989, centerLon=-47.732020)
    google_mps_api.imgDownload(centerLat=-11.074495, centerLon=-42.051459)
    google_mps_api.imgDownload(centerLat=-10.198966, centerLon=-46.872713)
    google_mps_api.imgDownload(centerLat=-16.618904, centerLon=-70.077475)
    google_mps_api.imgDownload(centerLat=-23.669320, centerLon=-46.559369)
    google_mps_api.imgDownload(centerLat=38.080750, centerLon=81.888615)
    generate.generate()
    
#-11.0744952,-42.0514597
#-10.1989664,-46.8727137
#-16.6189048,-70.077475
#24.4984753,5.458369
#38.0807507,81.8886153
#-23.6693206,-46.5593699 sbc
    