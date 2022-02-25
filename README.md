# API for Urban Semantic Segmentation

## Preprequisite

Make sure you have a modern version of `docker` (>1.13.0) and `docker-compose` installed.

## Setup

This repo contains a simple server client architecture to enable making image segmentation inferences through a simple webpage

```
├── README.md
├── backend
│   ├── Dockerfile
│   ├── legend.png
│   ├── main.py
│   ├── model_weights
│   │   ├── semantic-segmentation-adas-0001.bin
│   │   ├── semantic-segmentation-adas-0001.xml
│   │   └── semantic_segmentation.ipynb
│   ├── plot.py
│   ├── requirements.txt
│   ├── semantic_segmentation_model.py
│   └── test_main.py
├── docker-compose.yml
├── frontend
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
└── storage
```
To setup the app and run it, follow these steps 
1. Clone this repository
2. access the repository then run the following command to build 2 docker images and compose them
```
docker-compose up -d
```
This might take some time inorder to build 2 docker images, one called `backend` and the other one `frontend`

3. Access the frontend webpage through the following url
```
http://0.0.0.0:8501/
```
Now you should have your docker containers up and running

## Usage
The webapp is a simple app 
<img width="1268" alt="image" src="https://user-images.githubusercontent.com/36573471/155707903-28c10580-0787-4049-aae3-476154923597.png">
Use the sidebar to upload and image
<img width="1659" alt="image" src="https://user-images.githubusercontent.com/36573471/155708016-a4827352-68e3-453e-8eb8-c472247f9fb4.png">
once an image is uploaded, it gets displayed for you 
after that you can click on the button `predict` to run your prediction
A onverlay of a predicted segmentation mask and 


