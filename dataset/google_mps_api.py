import requests
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv()

BASE_URL = "https://maps.googleapis.com/maps/api/staticmap?"
API_KEY = os.getenv("GOOGLE_API_KEY")

def imgDownload(centerLat=-21.9469896, centerLon=-47.7320201, zoom=15):
	for latitude in range(0, 5, 1):
		for longitude in range (0, 2, 1):
			
			coordenate = str(centerLat + latitude/10) + "," + str(centerLon + longitude/10)

			URL = BASE_URL + "center=" + coordenate + "&zoom=" + str(zoom) + "&size=256x280&maptype=satellite&scale=2&key=" + API_KEY
			# HTTP request
			response = requests.get(URL)
			print(response)
			# storing the response in a file (image)
			name = "images/" + coordenate + ".png"
			with open(name, 'wb') as file:
			   # writing data into the file
			   file.write(response.content)

			# You will get 403 as status_code if your API Key is invalid

			img = Image.open(name)

			cropped_filtred = img.crop((0,0,256,256))
			cropped_filtred.save(name)