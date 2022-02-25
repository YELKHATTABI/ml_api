# API for Urban Semantic Segmentation

## Model description 

Reference : https://docs.openvino.ai/2019_R1/_semantic_segmentation_adas_0001_description_semantic_segmentation_adas_0001.html

The model is a pretrained segmentation model that outputs a mask of size `H=1024, W=2048` with integers of values in `[0, ..., 19]` that represents the following classes :

0. road
1. sidewalk
2. building
3. wall
4. fence
5. pole
6. traffic light
7. traffic sign
8. vegetation
9. terrain
10. sky
11. person
12. rider
13. car
14. truck
15. bus
16. train
17. motorcycle
18. bicycle
19. ego-vehicle

## Openvino 
https://docs.openvino.ai/latest/index.html

Openvino is a library developped by intel to optimize deeplearning computations on intel CPU and other equipement. For this application, openvino inference engine, as well and optimized segmentation model, where used to make the up lighteweighted and usable on any machine with intel x86 cpu architecture

Used weights are downloadable from here : https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/1/semantic-segmentation-adas-0001/FP32/
 

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
A onverlay of a predicted segmentation mask and and the input image
<img width="1638" alt="image" src="https://user-images.githubusercontent.com/36573471/155710614-45915781-1548-4d9f-961a-a67ce88abb3f.png">


