from __future__ import annotations

import logging
import os
import sys
import traceback
from pathlib import Path
from uuid import uuid4

import gradio as gr
import torch
from gradio_rerun import Rerun


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.inference import VGGTInferencePipeline, VGGTOutput
from scripts.point_cloud import ColorMode
from scripts.visualizer import visualize_result


logger = logging.getLogger(__name__)


def delete_rrd(rrd_path: str | None) -> None:
    if rrd_path and os.path.isfile(rrd_path):
        os.unlink(rrd_path)


def process_data(
    file_list,
    pipeline: VGGTInferencePipeline,
    *,
    view: ColorMode = "rgb",
    percentile: float | None = None,
    old_rrd_path: str | None = None,
) -> tuple[VGGTOutput | None, str | None, str, str | None]:
    if not file_list:
        return None, None, "Error: Please upload images first.", old_rrd_path

    delete_rrd(old_rrd_path)
    image_paths = sorted(Path(file.name) for file in file_list)

    try:
        results = pipeline.predict(image_paths)
        run_id = str(uuid4())
        data, rrd_path, state = visualize_result(
            results,
            percentage=percentile,
            mode=view,
            recording_id=run_id,
        )
        return data, rrd_path, state, rrd_path
    except Exception:
        full_error = traceback.format_exc()
        logger.exception("Reconstruction failed")
        return None, None, f"CRASH DETECTED:\n\n{full_error}", None


def post_process(
    data: VGGTOutput | None,
    threshold: float,
    *,
    view: ColorMode = "rgb",
    old_rrd_path: str | None = None,
) -> tuple[str | None, str | None]:
    if data is None:
        return None, old_rrd_path

    delete_rrd(old_rrd_path)
    run_id = str(uuid4())
    _, rrd_path, _ = visualize_result(
        data,
        percentage=threshold,
        mode=view,
        recording_id=run_id,
    )
    return rrd_path, rrd_path


def build_demo(
    pipeline: VGGTInferencePipeline | None = None,
    *,
    device: str | None = None,
) -> gr.Blocks:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if pipeline is None:
        logger.info("Initializing VGGT model on %s", device)
        pipeline = VGGTInferencePipeline(device=device)

    def on_reconstruct(file_list, view, old_rrd_path):
        return process_data(
            file_list,
            pipeline,
            view=view,
            old_rrd_path=old_rrd_path,
        )

    def on_apply_filter(data, threshold, view, old_rrd_path):
        return post_process(
            data,
            threshold,
            view=view,
            old_rrd_path=old_rrd_path,
        )

    with gr.Blocks(title="VGGT 3D") as demo:
        temp_file = gr.State("")
        vggt_output = gr.State(None)

        gr.Markdown("# VGGT 3D Reconstruction")
        gr.Markdown("Upload a sequence of images to reconstruct")

        with gr.Row():
            with gr.Column():
                img_input = gr.File(
                    label="1. Upload Image Sequence",
                    file_count="multiple",
                    file_types=["image"],
                )

                mode_radio = gr.Radio(
                    choices=["rgb", "confidence"],
                    value="rgb",
                    label="2. Visualization Mode",
                    info="RGB: Real Colors | Confidence: Confidence heatmap",
                )

                percentile_slider = gr.Slider(
                    minimum=0.0,
                    maximum=100.0,
                    value=20.0,
                    step=1.0,
                    label="3. Confidence Percentile Filter",
                    info="Higher values keep only higher-confidence points",
                )

                status = gr.Textbox(
                    label="System Status",
                    interactive=False,
                )

                run_btn = gr.Button("Reconstruct Scene", variant="primary")
                filter_btn = gr.Button("Apply Filter")

            with gr.Column(scale=3):
                viewer = Rerun(height=640)

        run_btn.click(
            fn=on_reconstruct,
            inputs=[img_input, mode_radio, temp_file],
            outputs=[vggt_output, viewer, status, temp_file],
        )
        filter_btn.click(
            fn=on_apply_filter,
            inputs=[vggt_output, percentile_slider, mode_radio, temp_file],
            outputs=[viewer, temp_file],
        )

    return demo


demo = build_demo()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo.launch(inbrowser=True, share=False)
