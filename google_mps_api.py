import requests
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv()

BASE_URL = "https://maps.googleapis.com/maps/api/staticmap?"
API_KEY = os.getenv("GOOGLE_API_KEY")
ZOOM = "15"
CENTER = [-21.9469896,-47.7320201]


for latitude in range(0, 100, 1):
	for longitude in range (0, 10, 1):

		coordenate = str(CENTER[0] + latitude/10) + "," + str(CENTER[1] + longitude/10)

		URL = BASE_URL + "center=" + coordenate + "&zoom=" + ZOOM + "&size=256x280&maptype=satellite&scale=2&key=" + API_KEY
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