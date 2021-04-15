import dataset
from dataset import google_mps_api
from dataset import generate
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

    zoom = 8
    
    list_coords = [
        [-25.309753, -50.730875], #Sudeste Brasil
        [-7.968279, -40.764780], #Nordeste Brasil
        [-4.685834, -63.156655], #Amazonia
        [-40.510790, -68.362434], #Argentina
        [39.810931, -110.132505], #Utah, USA
        [23.645234, 11.900882], #Fronteira Argélia/Nigér/Líbia
        [70.580181, 102.190111], #Norte Russia
        [-23.100467, 132.055375], #Austrália
        [-0.442852, 23.886118], #Congo
        [36.901435, 100.194718], #Lago Qinghai, China
		[-31.985363, 17.425811], #Africa do Sul
		[42.728016, -124.475192] #Oregon
    ]

    for coord in list_coords:
        google_mps_api.imgDownload(centerLat=coord[0], centerLon=coord[1], zoom = zoom)
    
    generate.generate()