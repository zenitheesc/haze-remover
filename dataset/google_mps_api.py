import requests
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv()

BASE_URL = "https://maps.googleapis.com/maps/api/staticmap?"
API_KEY = os.getenv("GOOGLE_API_KEY")

def imgDownload(centerLat=-21.9469896, centerLon=-47.7320201, zoom=15, path="images/originais/"):
	for latitude in range(0, 20, 1):
		for longitude in range (0, 20, 1):
			
			coordenate = str(centerLat + latitude/20.0) + "," + str(centerLon + longitude/20.0)

			file_total_path = path + coordenate + ".png"

			if(not(os.path.exists(file_total_path))):
				URL = BASE_URL + "center=" + coordenate + "&zoom=" + str(zoom) + "&size=256x280&maptype=satellite&key=" + API_KEY
				# HTTP request
				response = requests.get(URL)
				print(response)
				# storing the response in a file (image)
				
				with open(file_total_path, 'wb') as file:
				   # writing data into the file
				   file.write(response.content)

				# You will get 403 as status_code if your API Key is invalid
				
				img = Image.open(file_total_path)

				cropped_filtred = img.crop((0,0,256,256))
				cropped_filtred.save(file_total_path)
			else:
				print("File " + file_total_path + " already exists")