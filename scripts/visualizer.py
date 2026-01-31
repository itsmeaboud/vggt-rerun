import sys
from pathlib import Path
# Add the parent folder to path (one level up)
sys.path.append(str(Path(__file__).parent.parent))
import rerun as rr
import rerun.blueprint as rrb
import torch
import numpy as np
import time
import matplotlib.pyplot as plt

def create_vggt_blueprint(num_frames):

    # 3D view: Points + Frustums Only
    view3d = rrb.Spatial3DView(
        name="3D Map",
        origin='world',
        contents=[
            "+ $origin/**",                     
            "- world/active_camera/**",         # Hide the moving 'active' camera copy from 3D
            "- world/camera_**/image",          # Hide all RGB planes in 3D
            "- world/camera_**/depth",          # Hide all Depth planes in 3D
            "- world/camera_**/confidence",    # Hide all Confidence planes in 3D
            "+ world/camera_*"                 # RE-INCLUDE the wireframe frustums
        ]
    )


    # 2D view: Image, Depth, Confidence
 

    view2d = rrb.Vertical(
        name = f"Active Frame",
        contents = [
            rrb.Spatial2DView(name = "Image", origin = f"world/active_camera/image"),
            rrb.Spatial2DView(name = "Depth", origin = f"world/active_camera/depth"),
            rrb.Spatial2DView(name = "Confidence", origin = f"world/active_camera/confidence")              
        ]
    )

    # Final layout
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            contents = [
                view3d,
                rrb.Vertical(name = "2D Inspector", contents = [view2d])
            ],
            column_shares = [2, 1]
        ),
        collapse_panels = True
    )

    return blueprint





def visualize_result(data: dict, 
                     percentage: float = 70.0, 
                     mode: str = "rgb"):



    print("Streaming data to Rerun Timeline...")

  
    # Retrieve data
    world_points = data['world_points']  # (S, H, W, 3)
    confidence = data["world_points_conf"]   # (S, H, W)
    images = data['images']              # (S, H, W, C)
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis = 0)
    
    depth_maps = data["depth"]

    extrinsic = data["extrinsic"]        # (S, 3, 4)
    intrinsic = data["intrinsic"]        # (S, 3, 4)

    frames = images.shape[0]



    parent_path = Path("world") 

    rr.log(f"{parent_path}",
           rr.ViewCoordinates.RDF,
           static=True
           )
    
    rr.log(
        f"{parent_path}",
        rr.Transform3D(rotation = rr.RotationAxisAngle(axis=(0, 1, 0), radians=-np.pi / 4)),
        static=True,
    )


    
    for idx in range(frames):

        rr.set_time("frame_idx", sequence = idx)


        # Prepare data frame for visualization
        wp_flatten = world_points[idx].reshape(-1, 3)
        colors_flatten = images[idx].reshape(-1, 3)
        print(colors_flatten.shape)
        image_rgb = images[idx]
        depth_map = depth_maps[idx]
        conf_map = confidence[idx]

        

        threshold = np.percentile(conf_map, percentage)
        mask = conf_map > threshold
        mask_flatten = mask.reshape(-1)
        count = np.count_nonzero(mask_flatten)
        wp_filterd = wp_flatten[mask_flatten]
        colors_filtered = colors_flatten[mask_flatten]

        conf_map = conf_map * mask

        conf_max = conf_map.max()
        conf_map /= conf_max


        # 1. Log 3D points to unique paths (This creates the 'buildup' you want)
        rr.log(
            f"world/points/frame_{idx}",
            rr.Points3D(wp_filterd, colors=colors_filtered)
        )

        # 2. Log to unique camera path (for the 3D 'trail' of frustums)
        cam_path = f"world/camera_{idx}"
        rr.log(cam_path, rr.Pinhole(image_from_camera=intrinsic[idx][:,:3], 
                                    width=image_rgb.shape[1], height=image_rgb.shape[0]))
        rr.log(cam_path, rr.Transform3D(translation=extrinsic[idx][:3, 3], 
                                         mat3x3=extrinsic[idx][:3, :3]))

        # 3. Log to 'active_camera' path (This drives the 2D views)
        active_path = "world/active_camera"
        rr.log(active_path, rr.Pinhole(image_from_camera=intrinsic[idx][:,:3], 
                                       width=image_rgb.shape[1], height=image_rgb.shape[0]))
        rr.log(active_path, rr.Transform3D(translation=extrinsic[idx][:3, 3], 
                                           mat3x3=extrinsic[idx][:3, :3]))
        
        # Log the actual 2D images to the active path
        rr.log(f"{active_path}/image", rr.Image(image_rgb))
        rr.log(f"{active_path}/depth", rr.DepthImage(depth_map))
        rr.log(f"{active_path}/confidence", rr.Image(conf_map))

    # Send the blueprint after the loop
    rr.send_blueprint(create_vggt_blueprint(frames))
    print(f"✅ Processed {frames} frames.")

if __name__ == "__main__":

  pass 