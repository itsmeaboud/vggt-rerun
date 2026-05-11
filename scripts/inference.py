import sys
from pathlib import Path
# Add the parent folder to path (one level up)
#sys.path.append(str(Path(__file__).parent.parent))
from typing import List, Tuple, Optional, Literal, Sequence
from dataclasses import dataclass
import logging

import numpy as np
import torch
from jaxtyping import Float32, Float
import time
from pathlib import Path

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images


logger = logging.getLogger(__name__)

PreProcessMode = Literal['crop', 'pad']


def tensor_sequence_to_numpy(
        tensor: Float32[torch.Tensor, "1 S ..."] | Float32[np.ndarray, "1 S ..."]
        ) -> Float32[np.ndarray, "S ..."]:
    
    """converting [1, S, ...]output tensor of model to a numpy array [S, ...]"""
    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().cpu().numpy()
    else:
        array = tensor
    if array.ndim > 0 and array.shape[0] == 1:
        array = array[0]
    return array

@dataclass
class VGGTOutput: 
    world_points: Float32[np.ndarray, "S H W 3"]
    world_points_conf: Float32[np.ndarray, "S H W"]
    depth: Float32[np.ndarray, "S H W 1"]
    depth_conf: Float32[np.ndarray, "S H W"]
    images: Float32[np.ndarray, "S H W 3"]
    extrinsic: Float32[np.ndarray, "S 3 4"]
    intrinsic: Float32[np.ndarray, "S 3 3"]
    shape: Tuple[int, int]
    frames: int
    pose_enc: Float32[np.ndarray, "S 9"]


class VGGTInferencePipeline:

    def __init__(
            self, 
            model_name: str = "facebook/VGGT-1B", 
            device: str = 'cpu',
            preprocess_mode: PreProcessMode = "crop"
            ):
        
        self.device = torch.device(device)
        self.preprocess_mode = preprocess_mode
        self.model = VGGT.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info("VGGT model loaded on %s", self.device)


    def predict(self, image_paths: Sequence[str | Path]) -> VGGTOutput :

        paths = [Path(path)for path in image_paths]
        if not paths:
            raise ValueError("At least one image is required")

        # Prepare the image tensor [S, C, H, W] for feed forward
        batch_images: Float32[torch.Tensor, "S 3 H W"] = load_and_preprocess_images(paths, mode = self.preprocess_mode)
        batch_images = batch_images.to(self.device, non_blocking = True)
        height, width = batch_images.shape[-2:]
        frames = batch_images.shape[0]

        logger.info("Inference started for %d frame(s)", frames)
        start = time.perf_counter()
        with torch.inference_mode():
            #prediction keys ->['pose_enc', 'pose_enc_list', 'depth', 'depth_conf', 'world_points', 'world_points_conf'])
            predictions = self.model(batch_images)
        logger.info("Inference finished in %.2fs", time.perf_counter() - start)

        # [S, C, H, W] -> [S, H, W, C] for visulaization
        images = batch_images.detach().cpu().permute(0, 2, 3, 1).numpy()

        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions['pose_enc'], batch_images[0].shape[-2:])

        depth = tensor_sequence_to_numpy(predictions['depth'])
        depth_conf = tensor_sequence_to_numpy(predictions['depth_conf'])
        extrinsic = tensor_sequence_to_numpy(extrinsic)
        intrinsic = tensor_sequence_to_numpy(intrinsic)
        world_points_conf = tensor_sequence_to_numpy(predictions['world_points_conf'])
        pose_enc = tensor_sequence_to_numpy(predictions['pose_enc'])
        world_points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)


        return VGGTOutput(
            world_points = world_points,
            world_points_conf = world_points_conf,
            depth = depth,
            depth_conf = depth_conf,
            images = images,
            extrinsic = extrinsic,
            intrinsic = intrinsic,
            shape = (height, width),
            frames = frames,
            pose_enc = pose_enc
        )



    

#['pose_enc', 'pose_enc_list', 'depth', 'depth_conf', 'world_points', 'world_points_conf'])

if __name__ == "__main__" :

    debug = False
    WORK_DIR = Path("/home/aboud/Desktop/Projects/vggt-rerun/demo/images")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #pipeline = VGGTInferencePipeline(device = device)
    
    image_paths = list(WORK_DIR.glob("*"))
    print(WORK_DIR)
    #out_dict = pipeline.predict(image_paths = image_paths)
    mask = np.random.randint(2, size = (100, 200)).astype(dtype=np.bool)
    pts = np.random.randint(10, size = (100, 200, 3))
    print(mask.astype(dtype=np.bool))
    print(mask)
    print(mask.shape)
    print(pts.shape)
    print(pts[mask].shape)

    depth = np.random.rand(3, 500, 600, 1)
    ext = np.random.rand(3, 3, 4)
    intr =  np.random.rand(3, 3, 3)
    intr[:, 0, 1], intr[:, 1, 0] = 0, 0
    unproject_depth_map_to_point_map(depth, ext, intr)
    if image_paths:
         
        # Run prediction
        #predictions = pipeline.predict(image_paths = image_paths)
        print("Inference Complete.")
    
        
    else:
        print("No images were found.")






        


        


        
    
