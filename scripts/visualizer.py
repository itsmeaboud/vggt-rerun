from __future__ import annotations

import sys
from pathlib import Path
# Add the parent folder to path (one level up)
sys.path.append(str(Path(__file__).parent.parent))
import rerun as rr
import rerun.blueprint as rrb
import torch
import numpy as np
from typing import Tuple, Dict, TYPE_CHECKING
from jaxtyping import Float, Int, Bool, UInt8
from numpy import ndarray
import tempfile

from scripts.inference import VGGTOutput
from scripts.point_cloud import (
    ColorMode,
    point_cloud_to_frame,
    confidence_image,
    rgb_to_uint8,
)

if TYPE_CHECKING:
    from inference import VGGTOutput


def create_vggt_blueprint(num_frames: Int) -> rrb.Blueprint:

    # 3D view: Points + Frustums Only
    view3d = rrb.Spatial3DView(
        name="3D Map",
        origin='world',
        contents=[
            "+ $origin/**",                     
            "- world/active_camera/**",         
            "- world/camera_**/image",          
            "- world/camera_**/depth",          
            "- world/camera_**/confidence",    
            "+ world/camera_*"                 
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

    return rrb.Blueprint(
        rrb.Horizontal(
            contents = [
                view3d,
                rrb.Vertical(name = "2D Inspector", contents = [view2d])
            ],
            column_shares = [2, 1]
        ),
        collapse_panels = True
    )



def new_rrd_path() -> Path:
    with tempfile.NamedTemporaryFile(prefix = "output_", suffix = ".rrd", delete = False) as temp:
        return Path(temp.name)
    



def write_to_rrd(
        data: VGGTOutput,
        percentile: float | None = None,
        color_mode: ColorMode = "rgb",
        recording_id: str | None = None,
        output_path: str | Path | None = None
) -> Path:
    
    output = Path(output_path) if output_path else new_rrd_path()
    height, width = data.shape

    recording = rr.RecordingStream(application_id = "VGGT", recording_id = recording_id)
    try:
        recording.save(path = str(output), default_blueprint = create_vggt_blueprint(data.frames))
        recording.log("world", rr.Clear(recursive = True))
        recording.log("world", rr.ViewCoordinates.RDF, static = True)
        recording.log(
            "world",
            rr.Transform3D(
                rotation = rr.RotationAxisAngle(axis = (0, 1, 0), radians = np.pi /4),   
            ),
            static = True
        )

        for frame_idx in range(data.frames):
            recording.set_time("frame_idx", sequence = frame_idx)
            points, colors = point_cloud_to_frame(
                data = data,
                frame_idx = frame_idx,
                percentile = percentile,
                color_mode = color_mode
            )

            recording.log(
                f"world/points/frame_{frame_idx}",
                rr.Points3D(points, colors = colors)
            )

            camera_path = f"world/camera_{frame_idx}"
            recording.log(
                camera_path,
                rr.Pinhole(image_from_camera = data.intrinsic[frame_idx], width = width, height = height)
            )
            recording.log(
                camera_path,
                rr.Transform3D(
                    translation = data.extrinsic[frame_idx][:3, 3],
                    mat3x3 = data.extrinsic[frame_idx][:3, :3]
                )
            )

            active_path = "world/active_camera"
            depth = data.depth[frame_idx]
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            recording.log(f"{active_path}/image", rr.Image(rgb_to_uint8(data.images[frame_idx])))
            recording.log(f"{active_path}/depth", rr.DepthImage(depth))
            recording.log(f"{active_path}/confidence", rr.Image(confidence_image(data.depth_conf[frame_idx])))

        recording.flush(timeout_sec = 10)
    finally:
        recording.disconnect()

    return output


def visualize_result(
        data: VGGTOutput,
        percentage: float | None = None,
        mode: ColorMode = "rgb",
        recording_id: str | None = None,
) -> tuple[VGGTOutput, str, str]:
    
    rrd_path = write_to_rrd(
        data,
        percentile = percentage,
        color_mode = mode,
        recording_id = recording_id
    )

    state = f"Processed {data.frames} frames"
    return data, str(rrd_path), state
if __name__ == "__main__":

  pass 
