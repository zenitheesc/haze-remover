import requests
from PIL import Image
from dotenv import load_dotenv
import os
from register.check_download import check_images

load_dotenv()

BASE_URL = "https://maps.googleapis.com/maps/api/staticmap?"
API_KEY = os.getenv("GOOGLE_API_KEY")

LOOP_RANGES = (2, 2) #latitude longitude, the total number of image downloads will be the multiplication between the two numbers 

def imgDownload(centerLat=-21.9469896, centerLon=-47.7320201, zoom=15, path="images/originais/"):
	
	center = (centerLat, centerLon)

	coordinate_list = check_images(LOOP_RANGES, center, path)
	
	if not coordinate_list: 
		return
	
	for coordinate in coordinate_list:
	
		URL = BASE_URL + "center=" + coordinate + "&zoom=" + str(zoom) + "&size=256x280&maptype=satellite&key=" + API_KEY
		
		response = requests.get(URL)
		print("Status code for " + coordinate + ": " + str(response.status_code))
		# You will get 403 as status_code if your API Key is invalid

		file_total_path = path + coordinate + ".png"

		with open(file_total_path, 'wb') as file:
		   
		   file.write(response.content)
		
		
		img = Image.open(file_total_path)
		cropped_filtred = img.crop((0,0,256,256))
		cropped_filtred.save(file_total_path)		

		