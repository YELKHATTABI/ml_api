"""
script containing helper functions to display correcly segmentation model predicitons
"""
import cv2
from typing import Tuple, Dict
import numpy as np

# Constant variable used for plotting mask with colors
CLASSES = {
    0: {"label": "Road", "color": (249, 65, 68),},
    1: {"label": "Sidewalk", "color": (243, 114, 44),},
    2: {"label": "Building", "color": (248, 150, 30),},
    3: {"label": "Wall", "color": (249, 199, 79),},
    4: {"label": "Fence", "color": (199, 199, 79),},
    5: {"label": "Pole", "color": (144, 190, 109),},
    6: {"label": "Traffic light", "color": (123, 170, 139),},
    7: {"label": "Traffic sign", "color": (123, 72, 88),},
    8: {"label": "Vegetation", "color": (87, 117, 10),},
    9: {"label": "Terrain", "color": (51, 101, 138),},
    10: {"label": "Sky", "color": (134, 187, 216),},
    11: {"label": "Person", "color": (247, 37, 133),},
    12: {"label": "Rider", "color": (114, 9, 183),},
    13: {"label": "Car", "color": (67, 97, 238),},
    14: {"label": "Truck", "color": (217, 237, 146),},
    15: {"label": "Bus", "color": (3, 4, 94),},
    16: {"label": "Train", "color": (255, 114, 60),},
    17: {"label": "Motorcyle", "color": (2, 62, 138),},
    18: {"label": "Bicycle", "color": (0, 150, 199),},
    19: {"label": "Ego vehicle", "color": (72, 202, 228),},
}


def generate_legend(classes: dict) -> np.array:
    """Create a legend wich is a mosaic of colors and a text for each class

    Args:
        classes (dict): class descriptions by label and color

    Returns:
        legend(np.array): image represented as a legend
    """
    num_classes = len(classes)
    legend = np.ones((num_classes * 50, 300, 3), dtype=np.uint8) * 255
    for i in list(classes.keys()):
        legend[50 * i : 50 * (i + 1), :50, :] = classes[i]["color"]
        legend = cv2.putText(
            img=legend,
            text=classes[i]["label"],
            org=(60, 35 + (50 * i)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=(10, 10, 10),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
    return legend


def generate_images_from_mask(
    image: np.array, raw_mask: np.array, classes: dict
) -> Tuple[np.array]:
    """_summary_

    Args:
        image (np.array): image predicted
        raw_mask (np.array): model predictions
        classes (dict): class descrition by class and color

    Returns:
        (mask,overlay) (Tuple[np.array]): mask of the model predictions and mask overlayed
        on the image
    """
    image_width, image_height = image.shape[1], image.shape[0]
    mask_height, mask_width = raw_mask.shape
    mask = np.zeros((mask_height, mask_width, 3), dtype=np.uint8)

    for i in list(classes.keys()):
        mask[raw_mask == i] = classes[i]["color"]

    mask = cv2.resize(mask, ((image_width, image_height)))

    overlay = cv2.addWeighted(image, 0.5, mask, 0.5, 0)

    return mask, overlay
