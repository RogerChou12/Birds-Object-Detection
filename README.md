# Birds-Object-Detection
## Real-Time Bird Detection with DETR and RTSP Streaming

[`detect.py`](detect.py) performs real-time or offline bird species detection using the [DETR (DEtection TRansformer)](https://huggingface.co/facebook/detr-resnet-50) model from Hugging Face Transformers. It supports both RTSP streams and video files as input, overlays bounding boxes with labels, and optionally streams the annotated video via RTSP or saves it locally.

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

### Workflow Overview

- **Load model**: Uses a fine-tuned `facebook/detr-resnet-50` model (or pretrained version)
  - Model path defaults to `"detr-resnet-50-birdspecies-v4/"`. Ensure this directory contains a fine-tuned DETR checkpoint or change it in `detect.py`.
- **Input**: Accepts RTSP stream or local video file as input
- **Processing**:
  - Each frame is processed by a DETR model to detect birds.
  - Bounding boxes, confidence scores, instance counts, and class labels are drawn on the frames.
- **Outputs annotated video to**:
  - RTSP stream (via FFmpeg)
  - Local video file (if input is a file)
- **Optional: Graceful Termination**: Create a file named `stop.txt` in the script's directory during inference. The script will detect the file and stop gracefully.

---

### Prerequisites
- Python 3.8+
- FFmpeg installed (`ffmpeg` in PATH)
- GPU with CUDA (optional but recommended)

---

### Dependencies

Install the required Python packages:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install transformers torchmetrics pillow opencv-python numpy
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
