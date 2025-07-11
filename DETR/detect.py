# Inference with DETR model from Hugging Face

import argparse
from transformers import AutoImageProcessor, DetrForObjectDetection
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
from PIL import Image, ImageDraw, ImageFont
import random
import logging
import cv2
import numpy as np
import os
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Bird species detection with DETR")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="RTSP stream URL or path to a video file"
    )
    return parser.parse_args()

def draw_boxes_pil(frame, results, id2label, id2color):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    labels_count = np.bincount(results["labels"].cpu().numpy(), minlength=len(id2label))
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        color = id2color[label.item()]
        count = labels_count[label.item()]
        label = id2label[label.item()]
        logger.info(f"Detected {label} with confidence {round(score.item(), 3)} at location {box}")
        # Draw bounding boxes on the image
        x, y, x2, y2 = tuple(box)
        draw.rectangle((x, y, x2, y2), outline=color, width=4)
        
        # Add the label text at the top-left corner of the bounding box
        text = f"{label} {count} ({round(score.item(), 3)})"
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
        left, top, right, bottom = font.getmask(text).getbbox() # list: [left, top, right, bottom]
        text_location = (x, y - (bottom-top))  # Position text above the bounding box
        draw.rectangle([text_location, (x+right, y)], fill=color)  # Background for text
        draw.text(text_location, text, fill="white", font=font)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def get_predictions(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.3, target_sizes=target_sizes)[0]
    return results

if __name__ == "__main__":
    args = parse_args()
    input = args.input

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Name of repo on the hub or path to a local folder
    model_name = "detr-resnet-50-birdspecies-v4/"
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = DetrForObjectDetection.from_pretrained(model_name)
    model.to(device)
    model.eval() # Set the model to evaluation mode

    # rtsp = "rtsp://61.221.202.2/v2"
    cap = cv2.VideoCapture(input)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    rtsp_output_url = "rtsp://localhost:8554/mystream"
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{frame_width}x{frame_height}',
        '-r', str(int(fps)),
        '-i', '-', # Accept images from stdin'-c:v', 'libx264'
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-f', 'rtsp',
        rtsp_output_url
    ]
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    # If input is a video file (not RTSP), set up VideoWriter
    is_video_file = not input.lower().startswith("rtsp://")
    if is_video_file:
        output_video_path = "output_detected.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        logger.info(f"Saving output video to {output_video_path}")
    else:
        out_writer = None

    frame_count = 0
    # Assign a random color to each label
    id2color = {int(id): tuple(random.randint(0, 255) for _ in range(3)) for id in model.config.id2label.keys()}

    logger.info("Starting inference...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        results = get_predictions(frame)
        logger.info(f"Frame {frame_count}: {len(results['scores'])} detections")
        frame = draw_boxes_pil(frame, results, model.config.id2label, id2color)

        # Write processed frame to RTSP output via ffmpeg
        ffmpeg_proc.stdin.write(frame.tobytes())

        # If saving to video file, write frame
        if out_writer is not None:
            out_writer.write(frame)

        # Check for stop file
        if os.path.exists("stop.txt"):
            logger.info("Stop file detected. Exiting inference loop.")
            break

    cap.release()
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    cv2.destroyAllWindows()
