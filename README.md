<h1 align="center" style="color:white; background-color:black">Haze Remover</h1>
<h4 align="center">Software development to remove fog from images captured during probe missions.</h4>

<p align="center">
	<a href="http://zenith.eesc.usp.br/">
    <img src="https://img.shields.io/badge/Zenith-Embarcados-black?style=for-the-badge"/>
    </a>
    <a href="https://eesc.usp.br/">
    <img src="https://img.shields.io/badge/Linked%20to-EESC--USP-black?style=for-the-badge"/>
    </a>
    <a href="https://github.com/zenitheesc/Visao/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/zenitheesc/Visao?style=for-the-badge"/>
    </a>
    <a href="https://github.com/zenitheesc/Visao/issues">
    <img src="https://img.shields.io/github/issues/zenitheesc/Visao?style=for-the-badge"/>
    </a>
    <a href="https://github.com/zenitheesc/Visao/commits/main">
    <img src="https://img.shields.io/github/commit-activity/m/zenitheesc/Visao?style=for-the-badge">
    </a>
    <a href="https://github.com/zenitheesc/Visao/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/zenitheesc/Visao?style=for-the-badge"/>
    </a>
    <a href="https://github.com/zenitheesc/Visao/commits/main">
    <img src="https://img.shields.io/github/last-commit/zenitheesc/Visao?style=for-the-badge"/>
    </a>
    <a href="https://github.com/zenitheesc/Visao/issues">
    <img src="https://img.shields.io/github/issues-raw/zenitheesc/Visao?style=for-the-badge" />
    </a>
    <a href="https://github.com/zenitheesc/Visao/pulls">
    <img src = "https://img.shields.io/github/issues-pr-raw/zenitheesc/Visao?style=for-the-badge">
    </a>
</p>

<p align="center">
    <a href="#environment-and-tools">Environment and Tools</a> •
    <a href="#steps-to-run-and-debug">Steps to run and debug</a>
    <!--<a href="#how-to-contribute">How to contribute?</a> •-->
</p>

## Environment and tools
- [Python](https://www.python.org/): ^3.8.5,
- [Google Maps Static API](https://developers.google.com/maps/documentation/maps-static/overview),
- [OpenCV](https://opencv.org/): ^4.2.0,
- [Tensorflow](https://www.tensorflow.org/): ^2.5.0,
## Steps to run and debug

 
The first thing you need to do is create a .env file at the root of the project. 

```bash
.
├── dataset
│   ├── generate.py
│   ├── google_mps_api.py
│   └── __init__.py
├── error.png
├── images
│   ├── clean
│   ├── hazed
│   └── originais
├── LICENSE
├── main.py
├── network
│   ├── haze.py
│   ├── run.py
│   └── validation.py
├── README.md
└── .env  <---

```
Inside .env file you must add the following line

```
GOOGLE_API_KEY="YOUR_API_KEY"
```
To generate your google maps api key follow the instructions in https://developers.google.com/maps/documentation/maps-static/overview

All the code related with the data set generation is located at **dataset** module.

After generate and setup your api key, now you can just run the following code at the root of the project directory.

```
python3 main.py
```
This will generate a certain number of images, half of the images will be clean images and they will be saved at **/images/clean**. The other half will be the hazed images, they will be saved at **/images/hazed** folder. These folders will look like this.

**/images/clean** folder
<p align="center">
    <img src="https://raw.githubusercontent.com/zenitheesc/Visao/assets/images.png"/>
</p>
<br>

**/images/hazed** folder
<p align="center">
    <img src="https://raw.githubusercontent.com/zenitheesc/Visao/assets/results.png"/>
</p>

Before determining the number of images you are going to use, there are a few things you need to know about the code operation.

The function responsible for downloading images is the **imgDownload** located at **dataset/google_maps_api.py**. This function works by taking a coordinate position and then changing the longitude and latitude values ​​in two **for** loops.

```python

for latitude in range(0, 15, 1):
		for longitude in range (0, 30, 1):

```
So the number of images that will be downloaded after call **imgDownload** once are going to be 15*30 = 450 images. But you can change these values depending on your need. So far, these are only the clean images.

After downloading the images, the function **generate**, which is responsible for generating the hazed images, will be called. This function is located at **dataset/generate.py**. In addition to generating fog in the images, this function increases the number of images by rotating them 7 times.

Thus, the total number of the images after using one coordinate, by calling **imgDownload** once and than **generate**, are going to be 450 * 8 = 3600. Note that, in the **main** function, located at **./main.py**, we are using 13 different coordinates parameters, so we are calling the **imgDownload** function 13 times. If you use all coordinates, your data set will consist of 3600 * 13 = 46800 images in **/images/clean** and another 46800 images in **/images/hazed**. The folder **/images/originais** contains only the images that have been downloaded, they have no transformation.

To simplify this analysis, use the following formula

```
number_of_clean_images = number_of_hazed_images = range_in_loop * number_of_coordinates * 8

number_of_images_in_originais = range_in_loop

total = number_of_clean_images + number_of_hazed_images + number_of_images_in_originais
```
In our case
```
range_in_loop = 15 * 30 = 450
number_of_coordinates = 13

number_of_clean_images = number_of_hazed_images = 450 * 13 * 8 = 46800

number_of_images_in_originais = 450

total = 46800 + 46800 + 450 = 94050
```


<!--- ## How to contribute

`(optional, depends on the project) list of simple rules to help people work on the project.`

`Examples: How to format a pull request\n How to format an issue` --->


<p align="center">
    <a href="http://zenith.eesc.usp.br">
    <img src="https://img.shields.io/badge/Check%20out-Zenith's Oficial Website-black?style=for-the-badge" />
    </a> 
    <a href="https://www.facebook.com/zenitheesc">
    <img src="https://img.shields.io/badge/Like%20us%20on-facebook-blue?style=for-the-badge"/>
    </a> 
    <a href="https://www.instagram.com/zenith_eesc/">
    <img src="https://img.shields.io/badge/Follow%20us%20on-Instagram-red?style=for-the-badge"/>
    </a>

</p>
<p align = "center">
<a href="zenith.eesc@gmail.com">zenith.eesc@gmail.com</a>
</p>
