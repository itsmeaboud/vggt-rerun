import sys
from pathlib import Path
# Add the parent folder to path (one level up)
sys.path.append(str(Path(__file__).parent.parent))
import rerun as rr
import rerun.blueprint as rrb
import torch
import numpy as np
import time
import matplotlib.cm
from typing import Tuple, Dict
from jaxtyping import Float, Int, Bool, UInt8
from numpy import ndarray
from rerun.blueprint import Blueprint
import tempfile

def create_vggt_blueprint(num_frames: Int) -> Blueprint:

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


def filter_and_normalize_confidence(conf_map: Float[ndarray, "H W"],
                                    percentile: Float
                                    ) -> Tuple[Bool[ndarray, "H W"], Float[ndarray, "H W"], Float] :

    threshold = np.percentile(conf_map, percentile)

    mask = conf_map > threshold

    conf_map_filtered = conf_map * mask

    conf_map_norm = conf_map_filtered / conf_map_filtered.max()

    return mask, conf_map_norm, threshold
    


def visualize_result(data: Dict, 
                     percentage: Float = 20.0, 
                     mode: str = "rgb",
                     recording_id: str | None = None) -> str: 
    



    print("Streaming data to Rerun Timeline...")
    print("Writing Rerun recording....")

    # Retrieve data from VGGT model output
    world_points: Float[ndarray, "S H W 3"] = data['world_points']

    confidence: Float[ndarray, "S H W"] = data["world_points_conf"]

    images: Float[ndarray, "S H W 3"] = data['images']


    # If only one image append batch = 1
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis = 0)
    
    depth_maps: Float[ndarray, "S H W"] = data["depth"]

    extrinsic: Float[ndarray, "S 3 4"] = data["extrinsic"]
    intrinsic: Float[ndarray, "S 3 4"] = data["intrinsic"]

    frames: Int = images.shape[0]

    temp = tempfile.NamedTemporaryFile(prefix = "output_", suffix = ".rrd", delete = False )
    temp_path = temp.name
    temp.close

    blueprint = create_vggt_blueprint(frames)
    recording = rr.RecordingStream(application_id = "VGGT",
                                recording_id = recording_id)
    recording.save(path = temp.name, default_blueprint = blueprint)



    recording.log("world", rr.Clear(recursive = True))

    parent_path = Path("world") 

    recording.log(f"{parent_path}",
           rr.ViewCoordinates.RDF,
           static=True
           )
    
    recording.log(
        f"{parent_path}",
        rr.Transform3D(rotation = rr.RotationAxisAngle(axis=(0, 1, 0), radians=-np.pi / 4)),
        static=True,
    )


    
    for idx in range(frames):

        recording.set_time("frame_idx", sequence = idx)


        # Prepare data frame for visualization
        wp_flatten = world_points[idx].reshape(-1, 3)

        image_rgb = images[idx]
        colors_flatten = image_rgb.reshape(-1, 3)

        depth_map = depth_maps[idx]
        conf_map = confidence[idx]

        # Filter the points
        mask, conf_map_norm, _ = filter_and_normalize_confidence(conf_map, percentage)
        mask = mask.reshape(-1)
        wp_filtered = wp_flatten[mask]
        colors_filtered = colors_flatten[mask]

        # Heatmap for confidence
        # Get color map
        cmap = matplotlib.cm.get_cmap('turbo')
        # Map confidence
        mapped_colors = cmap(conf_map / 100.0)
        # Extract RGB only          
        conf_map_colored = mapped_colors[:, :, :3]


        if mode == "confidence":
            colors_filtered = conf_map_colored.reshape(-1, 3)[mask]


        # 1. Log 3D points to unique paths (This creates the 'buildup' you want)
        recording.log(
            f"world/points/frame_{idx}",
            rr.Points3D(wp_filtered, colors=colors_filtered)
        )

        # 2. Log to unique camera path (for the 3D 'trail' of frustums)
        cam_path = f"world/camera_{idx}"
        recording.log(cam_path, rr.Pinhole(image_from_camera=intrinsic[idx][:,:3], 
                                    width=image_rgb.shape[1], height=image_rgb.shape[0]))
        recording.log(cam_path, rr.Transform3D(translation=extrinsic[idx][:3, 3], 
                                         mat3x3=extrinsic[idx][:3, :3]))

        # 3. Log to 'active_camera' path (This drives the 2D views)
        active_path = "world/active_camera"
        recording.log(active_path, rr.Pinhole(image_from_camera=intrinsic[idx][:,:3], 
                                       width=image_rgb.shape[1], height=image_rgb.shape[0]))
        recording.log(active_path, rr.Transform3D(translation=extrinsic[idx][:3, 3], 
                                           mat3x3=extrinsic[idx][:3, :3]))
        
        # Log the actual 2D images to the active path
        recording.log(f"{active_path}/image", rr.Image(image_rgb))
        recording.log(f"{active_path}/depth", rr.DepthImage(depth_map))
        recording.log(f"{active_path}/confidence", rr.Image(conf_map_colored))

    # Send the blueprint after the loop
    #rr.send_blueprint(create_vggt_blueprint(frames))
    recording.flush(timeout_sec = 10)
    recording.disconnect()

    state = f"Processed {frames} frames."
    print(state)
    
    return temp_path , state

if __name__ == "__main__":

  pass 