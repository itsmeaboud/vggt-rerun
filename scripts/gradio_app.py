import traceback
import gradio as gr
from pathlib import Path
import torch
import sys
import os
from uuid import uuid4
from gradio_rerun import Rerun
sys.path.append(str(Path(__file__).parent.parent))

from inference import VGGTInferencePipeline
from visualizer import visualize_result

def delete_rrd(rrd_path: str):

    if os.path.isfile(rrd_path):
        os.unlink(rrd_path)
        print("rrd temp file deleted")
    
    return
    


print("Initializing VGGT Model")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on device: {device}")

pipeline = VGGTInferencePipeline(device = device)


def process_data(file_list, percentile, view):
    """
    This function handles the "Click"
    It takes inputs from the UI, runs the model, and triggers the visualizer 
    """


    if not file_list:
        return None, "Error: Please upload images first"

    image_paths = sorted([Path(f.name) for f in file_list])

    try:
        status_msg = f"Processing {len(image_paths)} images on device {device}"
        print(status_msg)

        results = pipeline.predict(image_paths)
        run_id = str(uuid4())

        rrd_path, state = visualize_result(results, percentage = percentile, mode = view, recording_id = run_id)

        return rrd_path, state, rrd_path
    
    except Exception as e:

        full_error = traceback.format_exc()
        print(full_error)

        return None, f"CRASH DETECTED:\n\n{full_error}", None

with gr.Blocks(title = "VGGT 3D") as demo:

    temp_file_path = gr.State("")
    gr.Markdown("# VGGT 3D Reconstruction")
    gr.Markdown("Upload a sequence of images to reconstruct the 3D geometry")

    with gr.Row():
        with gr.Column():

            # Input 1: Image uploader
            img_input = gr.File(
                label = "1. Upload Image Sequence", 
                file_count = "multiple",
                file_types = ["image"]
            )

            # Input 2: Confidnce Slider
            percentile_slider = gr.Slider(
                minimum = 0.0,
                maximum = 100.0, 
                value = 20.0, 
                step = 1.0, 
                label = "2. Confidence Percentile Filter", 
                info = "Higher = Strict (Top % only). Lower = Relaxed (Show more points)."
            )

            # Input 3: View mode
            mode_radio = gr.Radio(
                choices = ["rgb", "confidence"], 
                value = "rgb",
                label = "3. Visualization Mode",
                info = "RGB: Real Colors | Confidence: Heatmap of Uncertainty"
            )

            # Status
            
            status = gr.Textbox(
                label = "System Status",
                interactive = False
            )
            

            run_btn = gr.Button("Reconstruct Scene", variant = "primary")

        with gr.Column(scale = 3):
            
            viewer = Rerun(
                height = 640
                )

    first_event = run_btn.click(
                fn = process_data,
                inputs = [img_input, percentile_slider, mode_radio],
                outputs = [viewer, status, temp_file_path]
        )
    
    first_event.then(
        fn = delete_rrd,
        inputs = [temp_file_path]

    )

    

    


    if __name__ == '__main__':
        print("Starting Gradio Server...")
        demo.launch(inbrowser = True, share = False)

    
