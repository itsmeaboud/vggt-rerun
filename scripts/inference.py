import sys
from pathlib import Path
# Add the parent folder to path (one level up)
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from torchvision import transforms
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from visualizer import visualize_result

from typing import List, Dict, Union





class VGGTInferencePipeline:

    def __init__(self, 
                 model_name: str = "facebook/VGGT-1B", 
                 device: str = 'cpu',
                 process_width: int = 800):
        
        self.device = torch.device(device)
        self.model = VGGT.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.pre_process = transforms.Compose([transforms.ToTensor()])
        self.process_width = process_width
        print("Model Loaded")


    def _resize_image(self,
                      img: Image.Image) -> Image.Image:
        
        w, h = img.size
        # Only resize if the image is actually larger than our target
        if w > self.process_width:
            aspect_ratio = h / w
            new_h = int(self.process_width * aspect_ratio)
            return img.resize((self.process_width, new_h), Image.Resampling.LANCZOS)
        return img
        

    def _pad_image(self, 
                   img: Image.Image, 
                   patch_size: int = 14) -> Image.Image:        
        
        w, h = img.size
        new_w = (w // 14) * 14
        new_h = (h // 14) * 14
        if new_w != w or new_h != h :
            return img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        return img
    def predict(self, 
                image_paths: List[Path],) -> Dict[str, torch.Tensor] :
        
   

        
        # Prepare the image tensor [S, C, H, W] for feed forward
        batch_images = load_and_preprocess_images(image_paths)
        
        with torch.no_grad():
            predictions = self.model(batch_images)

        # [S, C, H, W] -> [S, H, W, C] for visulaization
        predictions["images"] = batch_images.permute(0, 2, 3, 1)

        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions['pose_enc'], batch_images[0].shape[-2:])
        predictions['extrinsic'] = extrinsic
        predictions['intrinsic'] = intrinsic

        # Convert tensor to numpy and remove batch dim for rerun visulizing
        for key in predictions:
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].squeeze(0).numpy()



        return predictions


if __name__ == "__main__" :

    debug = False
    WORK_DIR = Path("../examples/images")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = VGGTInferencePipeline(device = device)
    
    image_paths = list(WORK_DIR.glob("*"))
    print(WORK_DIR)
    out_dict = pipeline.predict(image_paths = image_paths)

    if image_paths:
         
        # Run prediction
        predictions = pipeline.predict(image_paths = image_paths)
        print("Inference Complete.")
        
        visualize_result(predictions, percentage = 90)

        
    else:
        print("No images were found.")






        


        


        
    
