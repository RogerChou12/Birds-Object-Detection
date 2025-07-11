# Birds-Object-Detection
## Real-Time Bird Detection with DETR and RTSP Streaming

This script performs real-time or offline bird species detection using the [DETR (DEtection TRansformer)](https://huggingface.co/facebook/detr-resnet-50) model from Hugging Face Transformers. It supports both RTSP streams and video files as input, overlays bounding boxes with labels, and optionally streams the annotated video via RTSP or saves it locally.

---

### Workflow Overview

- Uses a fine-tuned `facebook/detr-resnet-50` model (or pretrained version)
- Accepts RTSP stream or local video file as input
- Outputs annotated video to:
  - RTSP stream (via FFmpeg)
  - Local video file (if input is a file)
- Supports overlay of labels, confidence scores, and instance counts
- Auto-terminates if `stop.txt` file is detected

---

### Dependencies

Install the required Python packages:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers torchmetrics pillow opencv-python numpy
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

### Notes

- Make sure your input RTSP stream is accessible and the output RTSP URL matches your mediaMTX configuration.
