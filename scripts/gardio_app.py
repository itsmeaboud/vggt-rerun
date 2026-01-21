import gradio as gr
from pathlib import Path
import torch
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from inference import VGGTInferencePipeline
from visualizer import visualize_result

print("Initializing VGGT Model")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on device: {device}")

pipeline = VGGTInferencePipeline(device = device)




def process_data(file_list, conf_threshold, view):
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

        visualize_result(results, threshold = conf_threshold, mode = view)

        return "Success! Check the Rerun tab that just opened"
    
    except Exception as e:
        return f"Error: {str(e)}"
    

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
            threshold_slider = gr.Slider(
                minimum = 0.0,
                maximum = 1.0, 
                value = 0.7, 
                step = 0.05, 
                label = "2. Confidence Threshold", 
                info = "Higher value = Cleaner but fewer points. Lower value = More points but more noise."
            )

            # Input 3: View mode
            mode_radio = gr.Radio(
                choices = ["rbg", "confidence"], 
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
        inputs = [img_input, threshold_slider, mode_radio],
        outputs = [output_text]
    )



    if __name__ == '__main__':
        print("Starting Gradio Server...")
        demo.launch(inbrowser = True, share = False)

    