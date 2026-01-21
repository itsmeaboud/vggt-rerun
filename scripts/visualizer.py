import rerun as rr
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add the parent folder to path (one level up)
sys.path.append(str(Path(__file__).parent.parent))

def visualize_result(results: dict, 
                     threshold: float = 0.7, 
                     mode: str = "rgb"):


    rr.init("VGGT Demo", spawn=True)

    print("Streaming data to Rerun Timeline...")

    #rr.log("world", rr.Clear(recursive=True))
    #rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, timeless=True)

    world_points = results['world_points'].squeeze(0)
    images = results['images']
    confidence = results["world_points_conf"].squeeze(0)
    poses = results["pose"].squeeze(0)
    frames = world_points.shape[0]

    cmap = plt.get_cmap('turbo')

    for i in range(frames):
        
  
        rr.set_time("frame_idx", sequence = i)

        img_np = images[i].permute(1, 2, 0).numpy()
        img_np = np.flipud(img_np)
        pts_np = world_points[i].reshape(-1, 3).numpy()
        conf_np = confidence[i].reshape(-1).numpy()

        mask = conf_np >= threshold
        filtered_pts = pts_np[mask]

        if mode == 'confidence':
            cols = cmap(conf_np[mask])[:, :3]
        else:
            cols = img_np.reshape(-1, 3)[mask]

        rr.log(
            "world/points", 
            rr.Points3D(filtered_pts, colors=cols, radii=0.01)
        )

        pose = poses[i].numpy()
        q = [pose[4], pose[5], pose[6], pose[3]] 

        rr.log(
            "world/camera",
            rr.Transform3D(
                translation=pose[:3],
                rotation=rr.Quaternion(xyzw=q)
            )
        )

        height = img_np.shape[0]
        width = img_np.shape[1]

        f_len_x = pose[7].item() * width  
        f_len_y = pose[8].item() * height 

        rr.log(
            "world/camera",
            rr.Pinhole(
                focal_length = (f_len_x, f_len_y),
                width = width,
                height = height,
                

                image_plane_distance = 0.1, 
                
                #camera_xyz=rr.ViewCoordinates.RDF 
            )
        )



        rr.log("world/camera", rr.Image(img_np))

    print(f"✅ Processed {frames} frames.")

if __name__ == "__main__":
    
    # Load Tensors
    print("Loading tensors...")
    base_path = "mywork/tensor"
    pts = torch.load(f"{base_path}/wp.pt", map_location='cpu')
    images = torch.load(f"{base_path}/images.pt", map_location='cpu')
    conf = torch.load(f"{base_path}/world_points_conf.pt", map_location='cpu')
    poses = torch.load(f"{base_path}/pose.pt", map_location='cpu')

    results = {
        "world_points": pts,
        "world_points_conf": conf,
        "pose": poses,
        "images": images
    }
            
    visualize_result(results=results)
    
