import uuid
import cv2
import uvicorn
from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile
import numpy as np
from PIL import Image
from semantic_segmentation_model import Model
from plot import CLASSES, generate_images_from_mask, generate_legend

global model, legend, legend_name

legend = generate_legend(CLASSES)
app = FastAPI()
model = Model("semantic-segmentation-adas-0001")
model.initialize_vino_model()


legend_name = "../frontend/legend.png"
cv2.imwrite(legend_name, generate_legend(CLASSES)[..., ::-1])


@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}


# post end point to receive image from frontend and send back predictions paths
@app.post("/predict")
def get_image(file: UploadFile = File(...)):
    image = np.array(Image.open(file.file))
    raw_mask = model.predict(image)
    mask, overlay = generate_images_from_mask(image, raw_mask, CLASSES)
    str_uuid = str(uuid.uuid4())
    mask_name = f"../frontend/{str_uuid}_mask.png"
    overlay_name = f"../frontend/{str_uuid}_overlay.png"
    cv2.imwrite(mask_name, mask[..., ::-1])
    cv2.imwrite(overlay_name, overlay[..., ::-1])
    return {
        "mask_name": mask_name,
        "overlay_name": overlay_name,
        "legend_name": legend_name,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
