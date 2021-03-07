<h1 align="center" style="color:white; background-color:black">Computer Vision</h1>
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

`For now, the only code available is for configuring the data set.`
 
The first thing you need to do is create a .env file at the root of the project. 

```bash
.
├── dataset
│   ├── generate.py
│   ├── google_mps_api.py
│   └── __init__.py
├── LICENSE
├── main.py
├── README.md
└── .env  <---
```
Inside .env file you must add the following line

```
GOOGLE_API_KEY="YOUR_API_KEY"
```
To generate your google maps api key follow the instructions in https://developers.google.com/maps/documentation/maps-static/overview

All the code related with the data set generation is located at **dataset** module.
To run and setup the data set you can just run the following code at the root of the project directory.

```
python3 main.py
```
This will generate 16000 images, 8000 clean images at **/images** folder and 8000 haze images at **/results** folder.

**/images** folder
<p align="center">
    <img src="https://raw.githubusercontent.com/zenitheesc/Visao/assets/images.png"/>
</p>
<br>

**/results** folder
<p align="center">
    <img src="https://raw.githubusercontent.com/zenitheesc/Visao/assets/results.png"/>
</p>
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
