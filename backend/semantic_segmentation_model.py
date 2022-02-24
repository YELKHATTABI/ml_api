import cv2
import numpy as np
from openvino.inference_engine import IECore
from pathlib import Path
import os

PATH_TO_VINO_WEIGHTS = Path(__file__).resolve().parent / "model_weights"

class Model:
    def __init__(self,name):
        self.exec_net = None
        self.net = None
        self.input_layer_name = None
        self.output_layer_name = None
        self.model_name = name
        
    def initialize_vino_model(self):
        """
        Init an OpenVINO Efficient net ready for embedding data.

        Returns:
            vino_inference_engine_model: loaded model ready to use for inference
        """
        ie = IECore()

        # Reading the saved model xml/bin files
        if not (PATH_TO_VINO_WEIGHTS / f"{self.model_name}.bin").is_file():
            print("Weights are not downloaded")
            print("downloading weights")
            os.system("curl ")
            print("file downloaded")
        net = ie.read_network(
            model=str(PATH_TO_VINO_WEIGHTS / f"{self.model_name}.xml"),
            weights=str(PATH_TO_VINO_WEIGHTS / f"{self.model_name}.bin"),
        )

        # Reshaping the input (xml/bin model is saved with a fixed input shape)
        

        # Loading the network in memory
        exec_net = ie.load_network(network=net, device_name="CPU")
        self.input_layer_name = next(iter(exec_net.input_info))
        self.output_layer_name = next(iter(exec_net.outputs))
        self.exec_net = exec_net
        self.net = net
    
    def preprocess_image(self,image):
    
        # For now we convert the image to colour if it's gray
        
        # get input shape (we mainly care about height and width)
        _, _, H, W = self.net.input_info[self.input_layer_name].tensor_desc.dims
        
        # Convert image to colour if it is gray
        if len(image.shape) == 2 or (
            len(image.shape) == 3 and image.shape[-1] == 1
        ):
            image = cv2.cvtColor(image.squeeze(), cv2.COLOR_GRAY2BGR)
        # Convert image to the size of the input layer of the model
        image = cv2.resize(image, (W,H), cv2.INTER_CUBIC)
        
        image = np.moveaxis(image, -1, 0)  # openvino works with (n, c, h, w) format
        image = np.expand_dims(image,0)

        return image
    
    def predict(self, image):
        original_y,original_x = image.shape[:-1]
        preprocessed_image = self.preprocess_image(image)
        raw_mask  = self.exec_net.infer(inputs={self.input_layer_name: preprocessed_image})[self.output_layer_name]
        raw_mask = raw_mask[0,0]        
        return raw_mask




    



