import cv2
import numpy as np

CLASSES = {
    0:{
        "label" : "road",
        "color":(249, 65, 68),
        },
    1:{
        "label" : "sidewalk",
        "color":(243, 114, 44),
        },
    2:{
        "label" : "building",
        "color":(248, 150, 30),
        },
    3:{
        "label" : "wall",
        "color":(249, 199, 79),
        },
    4:{
        "label" : "fence",
        "color":(199, 199, 79),
        },
    5:{
        "label" : "pole",
        "color":(144, 190, 109),
        },
    6:{
        "label" : "traffic_light",
        "color":(123, 170, 139),
        },
    7:{
        "label" : "traffic_sign",
        "color":(123, 72, 88),
        },
    8:{
        "label" : "vegetation",
        "color":(87, 117, 10),
        },
    9:{
        "label" : "terrain",
        "color":(51, 101, 138),
        },
    10:{
        "label" : "sky",
        "color":(134, 187, 216),
        },
    11:{
        "label" : "person",
        "color":(247, 37, 133),
        },
    12:{
        "label" : "rider",
        "color":(114, 9, 183),
        },
    13:{
        "label" : "car",
        "color":(67, 97, 238),
        },
    14:{
        "label" : "truck",
        "color":(217, 237, 146),
        },
    15:{
        "label" : "bus",
        "color":(3, 4, 94),
        },
    16:{
        "label" : "train",
        "color":(255, 114, 60),
        },
    17:{
        "label" : "motorcycle",
        "color":(2, 62, 138),
        },
    18:{
        "label" : "bicycle",
        "color":(0, 150, 199),
    },
    19:{
        "label" : "ego-vehic",
        "color":(72, 202, 228),
    },
}


def generate_legend(classes):
    num_classes = len(classes)
    legend = np.ones((num_classes*50,300,3),dtype=np.uint8)*255
    for i in list(classes.keys()):
        legend[50*i:50*(i+1),:50,:]=classes[i]["color"]
        legend = cv2.putText(
            img=legend,
            text=classes[i]["label"],
            org=(60,35+(50*i)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 0.8,
            color=(10,10,10),
            thickness=2,
            lineType=cv2.LINE_AA)
    return legend


def generate_images_from_mask(image,raw_mask,classes):
    image_width,image_height = image.shape[1],image.shape[0]
    H,W = raw_mask.shape
    mask = np.zeros((H,W,3),dtype=np.uint8)

    for i in list(classes.keys()):
        mask[raw_mask == i] = classes[i]["color"]

    mask = cv2.resize(mask,((image_width,image_height)))

    overlay = cv2.addWeighted(image,0.5,mask,0.5,0)
    
    return mask,overlay