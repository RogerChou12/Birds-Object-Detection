# Birds-Object-Detection
This repository provides a solution for real-time or offline bird species detection in videos and RTSP streams using the DETR (DEtection TRansformer) model. It enables users to automatically identify, classify, and annotate bird species in video frames, supporting both local video files and live video streams. The project is designed for applications such as wildlife monitoring, ecological research, and automated video analysis, making bird detection accessible and efficient with modern deep learning techniques.

---

## Real-Time Bird Detection with DETR and RTSP Streaming

[`detect.py`](detect.py) performs real-time or offline bird species detection using the [DETR (DEtection TRansformer)](https://huggingface.co/facebook/detr-resnet-50) model from Hugging Face Transformers. It supports both RTSP streams and video files as input, overlays bounding boxes with labels, and optionally streams the annotated video via RTSP or saves it locally.

---

### Workflow Overview

- **Load model:**
  - Uses a fine-tuned `facebook/detr-resnet-50` model (or pretrained version)
  - Model path defaults to `"detr-resnet-50-birdspecies-v4/"`. Ensure this directory contains a fine-tuned DETR checkpoint or change it in `detect.py`.
- **Input:**
  - Accepts RTSP stream or local video file as input
- **Processing**:
  - Each frame is processed by a DETR model to detect birds.
  - Bounding boxes, confidence scores, instance counts, and class labels are drawn on the frames.
- **Outputs annotated video to**:
  - RTSP stream (via FFmpeg)
  - Local video file (if input is a file)
- **Optional: Graceful Termination:**
  - Create a file named `stop.txt` in the script's directory during inference. The script will detect the file and stop gracefully.

---

### Prerequisites
- Python 3.9+
- FFmpeg installed (`ffmpeg` in PATH)
- GPU with CUDA (optional but recommended)

---

### Dependencies

Install the required Python packages:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install transformers torchmetrics pillow opencv-python numpy
```
Installation of `transformers`: [Transformers](https://github.com/huggingface/transformers/tree/main#installation)

---

### Usage
Run with a video file:
```sh
python detect.py --input path/to/video.mp4
```
Run with an RTSP stream:
```sh
python detect.py --input rtsp://your_stream_url
```

---

### Setting Up the RTSP Server with mediaMTX

1. **Install mediaMTX:**  
   Download and install [mediaMTX](https://github.com/bluenviron/mediamtx) (formerly rtsp-simple-server) on your machine.

2. **Start the RTSP Server:**  
   ```sh
   ./mediamtx
   ```
   By default, it listens on `rtsp://localhost:8554`.

3. **Run the Detection Script:**  
   Edit the `rtsp_output_url` in `detect.py` if needed (default: `rtsp://localhost:8554/mystream`), then run:
   ```sh
   python detect.py
   ```

4. **View the Stream:**  
   Use an RTSP client (e.g., VLC) to open:
   ```
   rtsp://localhost:8554/mystream
   ```
   Python/mediamtx/FFmpeg are running on WSL Ubuntu 22.04  
   VLC is running on Windows 11
   ```
   rtsp://<WSL_IP>:8554/mystream
   ```

---

### Notes

- Make sure your input RTSP stream is accessible and the output RTSP URL matches your mediaMTX configuration.
- FFmpeg must also be installed and available in your system path.

---

## Training Bird Species Detection Model with [`train.py`](train.py)

This guide explains how to use `train.py` to finetune a DETR (DEtection TRansformer) model for bird species object detection, leveraging Hugging Face Transformers and Accelerate for efficient and distributed training.

The `train.py` script is designed to:
- Finetune a pretrained DETR model on a custom bird species detection dataset.
- Handle advanced data augmentations to improve model robustness.
- Support distributed and accelerated training with experiment tracking.
- Save and optionally push the trained model to the Hugging Face Model Hub.

---

### Workflow Overview

1. **Argument Parsing:**  
   Configure training parameters via command-line arguments.

2. **Dataset Loading & Splitting:**  
   Load your labeled dataset (in JSON format), and automatically split into train, validation, and test sets if needed.

3. **Class Mapping:**  
   Load class mappings from a `class_mapping.json` file for label management.

4. **Model, Processor, and Augmentation Setup:**  
   Load a DETR model and image processor. Define strong augmentations using Albumentations.

5. **Sampler & DataLoader:**  
   Use class-aware sampling for balanced batches; prepare PyTorch DataLoaders.

6. **Training Loop:**  
   Train the model, evaluate after each epoch, and periodically save checkpoints.

7. **Evaluation:**  
   Measure mean average precision (mAP) and other metrics on validation and test sets.

8. **Saving and Pushing Model:**  
   Save the final model locally and/or push it to the Hugging Face Model Hub.

---

### Dependencies

Install all dependencies before running the script:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate datasets albumentations pillow numpy torchmetrics tensorboard tqdm
```

---

---

### Usage
About training with accelerate: [here](https://github.com/huggingface/transformers/tree/main/examples/pytorch/object-detection#pytorch-version-no-trainer)
```sh
accelerate launch train.py \
    --model_name_or_path "facebook/detr-resnet-50" \
    --dataset_name your_dataset.json \
    --output_dir "your_output_directory" \
    --num_train_epochs 100 \
    --image_square_size 640 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --checkpointing_steps epoch \
    --learning_rate 5e-6 \
    --ignore_mismatched_sizes \
    --with_tracking
```

---

### Notes
- Distributed/multi-GPU training is supported via [Accelerate](https://github.com/huggingface/accelerate).
- For best results, ensure your dataset and class mapping files are correctly formatted.

---

### Reference
- [Hugging Face DETR Documentation](https://huggingface.co/facebook/detr-resnet-50)
- [Accelerate](https://github.com/huggingface/accelerate)
- [Albumentations](https://albumentations.ai/)

---

## Data Collection and Annotation

This document explains how to use `yolo_inference.py` and `plotAnno.py` to automatically collect, annotate, and analyze bird detection data from videos stored in AWS S3, using YOLOv5 for inference.

- [**yolo_inference.py**](DataCollect/yolo_inference.py): Automates bird detection in videos from S3, generates bounding box annotations, saves cropped images, and uploads results and annotations back to S3.
- [**plotAnno.py**](plotAnno.py): Provides quick statistical analysis and visualization of the resulting annotations, helping you understand detection quality and category distributions.

---

### 1. Data & Annotation Collection ([`yolo_inference.py`](DataCollect/yolo_inference.py))

- Connects to AWS S3, lists videos in the input bucket/prefix.
- Downloads each video, processes frames using YOLOv5 (pretrained for bird detection).
- Crops detected bird regions, saves cropped images and annotated frames to S3.
- Generates annotation JSON files for both frames and crops, uploading them to S3.

**Key Outputs:**
- `annotations.json` (bounding box and detection info for each frame)
- `annotations_crop.json` (cropped image info)
- Cropped bird images and annotated frames in specified S3 folders

### 2. Data Visualization & Analysis ([`plotAnno.py`](plotAnno.py))

- Loads `annotations.json`.
- Computes and plots:
  - Number of boxes per image
  - Detection confidence histogram
  - Box area histogram
  - Category (class) distribution pie chart
- Saves visualization as `data_analysis.png`

---

### Required Packages

Install with pip:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install boto3 botocore opencv-python numpy matplotlib pandas
```

---

### Usage

#### 1. Prepare S3 and Class Mapping

- Store your videos in an S3 bucket/prefix.
- Prepare `class_mapping.json` and `class_mapping_chinese.json` (for category ID/name mapping).
- Set your AWS credentials and bucket/prefix info in `yolo_inference.py`.

#### 2. Run YOLO Inference & Annotation

```bash
python DataCollect/yolo_inference.py
```
- This will process all videos in the specified S3 bucket/prefix, save results locally and upload images/annotations to S3.

#### 3. Visualize Annotation Statistics

```bash
python plotAnno.py
```
- This will produce `data_analysis.png` summarizing the annotation statistics.

---

### Notes

- Make sure you have sufficient permissions to access and write to your S3 buckets.
- Adjust YOLO model parameters and S3 settings in `yolo_inference.py` as needed for your use case.
- Visualization assumes `annotations.json` exists in the working directory.

---

### References

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [AWS S3 Python SDK (boto3)](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [OpenCV](https://opencv.org/)
- [Matplotlib](https://matplotlib.org/)
