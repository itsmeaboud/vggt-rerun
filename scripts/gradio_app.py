import traceback
import time
import gradio as gr
from pathlib import Path
import torch
import os
import sys
from pathlib import Path
import rerun as rr

sys.path.append(str(Path(__file__).parent.parent))

from inference import VGGTInferencePipeline
from visualizer import visualize_result

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
        return "Error: Please upload images first"

    image_paths = sorted([Path(f.name) for f in file_list])

    try:
        status_msg = f"Processing {len(image_paths)} images on device {device}"
        print(status_msg)

        results = pipeline.predict(image_paths)
        run_id = f"VGGT_{int(time.time())}"

        rr.init("VGGT", recording_id = run_id, spawn = True)
        rr.log("world", rr.Clear(recursive = True))

        visualize_result(results, percentage = percentile, mode = view)

        return "Success! Check the Rerun tab that just opened"
    
    except Exception as e:
        #return f"Error: {str(e)}"
        full_error = traceback.format_exc()
        print(full_error)

        return f"CRASH DETECTED:\n\n{full_error}" #return f"Error: {str(e)}"   

with gr.Blocks(title = "VGGT 3D") as demo:

    gr.Markdown("# VGGT 3D Reconstruction Demo")
    gr.Markdown("Upload a sequence of images to reconstruct the 3D geometry")

    with gr.Row():
        with gr.Column(scale = 1):

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

            run_btn = gr.Button("Reconstruct Scene", variant = "primary")

        with gr.Column(scale = 1):
            output_text = gr.Textbox(
                label = "System Status", 
                interactive = False
            )

        gr.Markdown("""
            ### Instructions:
            1. Drag & Drop your images.
            2. Cick **Reconstruct Scene**.
            3. A new tab will open with the **Rerun Viewer**.
            4. Use the slider in Rerun to the frames build up.
        """)

    run_btn.click(
        fn = process_data,
        inputs = [img_input, percentile_slider, mode_radio],
        outputs = [output_text]
    )



    if __name__ == '__main__':
        print("Starting Gradio Server...")
        demo.launch(inbrowser = True, share = False)

    