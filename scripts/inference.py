import torch
from PIL import Image
from pathlib import Path
from torchvision import transforms
from vggt.models.vggt import VGGT
from typing import List, Dict, Union
import sys
from pathlib import Path

# Add the parent folder to path (one level up)
sys.path.append(str(Path(__file__).parent.parent))


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
        
        batch_tensor = []
        
        for path in image_paths:
            img = Image.open(path)
            img = self._resize_image(img)
            img = self._pad_image(img)
            batch_tensor.append(self.pre_process(img))

        if not batch_tensor:
            raise ValueError ("No images provided.")
        
        batch = torch.stack(batch_tensor, dim = 0).to(self.device)

        with torch.no_grad():
            out = self.model(batch)

        # Return CPU tensor for rerun visulizing

        return {"world_points": out["world_points"].cpu(),
                "world_points_conf": out["world_points_conf"].cpu(),
                "depth": out["depth"].cpu(),
                "depth_conf": out["depth_conf"].cpu(),
                "pose": out["pose_enc"].cpu(),
                "pose_list": out["pose_enc_list"],
                "images": batch.cpu()}
    

if __name__ == "__main__" :

    debug = True
    WORK_DIR = Path("mywork/images")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = VGGTInferencePipeline(device = device)
    
    image_paths = list(WORK_DIR.glob("*.png"))
    out_dict = pipeline.predict(image_paths = image_paths)

    if image_paths:
         
        # Run prediction
        result = pipeline.predict(image_paths = image_paths)
        print("Inference Complete.")
        print(f"Points Shape: {result['world_points'].shape}")
        print(f"Poses Shape: {result['pose'].shape}")

        if debug:
            torch.save(result["world_points"] ,"mywork/tensor/wp.pt")
            torch.save(result["images"], "mywork/tensor/images.pt")
            torch.save(result["world_points_conf"], "mywork/tensor/world_points_conf.pt")
            torch.save(result["pose"] ,"mywork/tensor/pose.pt")

    else:
        print("No images were found.")






        


        


        
    
