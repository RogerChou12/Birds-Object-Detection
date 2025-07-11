import io
import os
import cv2
import json
import torch
import boto3
import tempfile
import numpy as np
from botocore.config import Config

def save_results_to_json(results, output_file):
    """
    Save inference results to a JSON file.

    Args:
        results (list): List of detection results.
        output_file (str): Path to the output JSON file.
    """
    # Write the results to a JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f)
    print(f"Results saved to {output_file}")

def upload_json_to_s3(s3, bucket_name, key, data):
    """
    Upload a JSON file to an S3 bucket directly from memory.

    Args:
        bucket_name (str): Name of the S3 bucket.
        key (str): S3 object key (path in the bucket).
        data (dict or list): JSON-serializable data to upload.
    """
    json_stream = io.BytesIO(json.dumps(data).encode('utf-8'))  # Convert JSON data to a stream
    s3.upload_fileobj(json_stream, bucket_name, key)
    print(f"Uploaded JSON to s3://{bucket_name}/{key}")

def upload_image_to_s3(s3, bucket_name, key, image):
    """
    Upload an image to an S3 bucket.

    Args:
        bucket_name (str): Name of the S3 bucket.
        key (str): S3 object key (path in the bucket).
        image (numpy.ndarray): Image to upload.
    """
    # Encode the image as a JPEG in memory
    _, buffer = cv2.imencode('.jpg', image)
    image_stream = io.BytesIO(buffer)
    # Upload the image to S3
    s3.upload_fileobj(image_stream, bucket_name, key)
    print(f"Uploaded image to s3://{bucket_name}/{key}")

def process_video_from_s3(s3_client, bucket_name, video_key, model, device, output_bucket, output_prefix, output_crop_prefix, video_count, image_id):
    """
    Process a video directly from S3 and save detected frames back to S3.

    Args:
        bucket_name (str): Name of the S3 bucket containing the video.
        video_key (str): S3 object key for the video.
        model: YOLO model for inference.
        device: PyTorch device (CPU or GPU).
        output_bucket (str): Name of the S3 bucket to save processed images.
        output_prefix (str): Prefix (folder path) for saving images in the output bucket.
        output_crop_prefix (str): Prefix (folder path) for saving cropped images in the output bucket.
        video_count (int): Counter for the video being processed.
    
    Returns:
        list: Annotations for the frames.
        list: Annotations for the cropped images.
    """
   
    s3 = s3_client
    with tempfile.NamedTemporaryFile() as temp_video:
        # Stream video from S3 to a temporary file
        s3.download_fileobj(bucket_name, video_key, temp_video)
        temp_video.seek(0)

        # Open the video file
        cap = cv2.VideoCapture(temp_video.name)
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
        frame_interval = int(fps/fps)  # Process 10 frames per second
        frame_count = 0
        results = []
        crop_results = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames to process one frame per second
            frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if frame_id % frame_interval != 0:
                continue

            # Crop the frame to 640x640
            crop_height = 640
            crop_width = 640
            start_y = max((frame.shape[0] - crop_height) // 2, 0)
            start_x = max((frame.shape[1] - crop_width) // 2, 0)
            frame = frame[start_y:start_y + crop_height, start_x:start_x + crop_width]
            height, width, channels = frame.shape  # Get frame dimensions
            print(f"Processing {video_key} {temp_video.name} frame {frame_id} with dimensions: {width}x{height}")

            # Perform inference
            detections = model(frame)

            # Extract Chinese name from file name (before the dash)
            filename = os.path.splitext(os.path.basename(video_key))[0]  # e.g., "ChineseName-001"
            category_name_chinese = filename.split('-')[0]  # e.g., "ChineseName"
            # Load mapping (if you saved it as class_mapping.json)
            with open("class_mapping_chinese.json", "r", encoding="utf-8") as f:
                id2label = json.load(f)
            label2id_chinese = {v: int(k) for k, v in id2label.items()}
            category_id = label2id_chinese[category_name_chinese]  # This is the integer id
            with open("class_mapping.json", "r", encoding="utf-8") as f:
                id2label = json.load(f)
            category_name = id2label[str(category_id)]  # Get the class name from the ID

            # Process each detection
            category = []
            class_name = []
            area = []
            bbox = []
            confidence = []
            if not detections.pandas().xyxy[0].empty:
                frame_filename = f"{video_count}_frame_{frame_count:04d}.jpg"
                for xmin, ymin, xmax, ymax, conf, cls, name in detections.pandas().xyxy[0].values:  # Pandas DataFrame: id, xmin, ymin, xmax, ymax, confidence, class, name
                    category.append(category_id)  # Append class ID
                    class_name.append(category_name)  # Append class name
                    area.append(int((xmax - xmin) * (ymax - ymin)))  # Append area of the bounding box
                    bbox.append([xmin, ymin, xmax, ymax])  # Append bounding box coordinates
                    confidence.append(float(conf))  # Append confidence score

                    # Save cropped images
                    crop_filename = f"{video_count}_frame_{frame_count:04d}_crop_{int(xmin)}_{int(ymin)}_{int(xmax)}_{int(ymax)}.jpg"
                    crop_key = f"{output_crop_prefix}{crop_filename}"
                    cropped_image = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
                    upload_image_to_s3(s3, output_bucket, crop_key, cropped_image)

                    # Add cropped image annotation
                    crop_results.append({
                        'filename': crop_filename,
                        'image': frame_filename,  # Frame file name
                        'video': video_key,
                        'bbox': [xmin, ymin, xmax, ymax]
                    })
                
                # Upload the frame directly to S3
                s3_key = f"{output_prefix}{frame_filename}"
                upload_image_to_s3(s3, output_bucket, s3_key, frame)

                results.append({
                        'video': video_key,  # Video file name
                        'image': frame_filename,  # Frame file name
                        'image_id':image_id, # image id
                        'width': width,
                        'height': height,
                        'objects': {
                            'category': category,  # Class ID
                            'class_name': class_name,  # Class name
                            'area': area,  # Area of the bounding box
                            'bbox': bbox, # xmin, ymin, xmax, ymax
                            'confidence': confidence  # Confidence score
                        }
                    })

                frame_count += 1
                image_id += 1

        cap.release()  # Release the video capture object
    print(f"Frames processed and uploaded to s3://{output_bucket}/{output_prefix}")
    return results, crop_results

def main():
    # S3 bucket and folder details
    Access_key_ID = "your-access-key"
    Secret_access_key = "your-secret-key"
    input_bucket = "your-input-bucket-name"  # S3 bucket name
    input_prefix = "S3 prefix for videos/"  # S3 prefix (folder path)
    output_bucket = "your-output-bucket-name"
    output_prefix = "folder-name"
    output_crop_prefix = "folder-name"
    annotation_key = "filename.json"  # S3 key for the annotations JSON file
    annotation_crop_key = "filename.json"
    # Load YOLO model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = torch.hub.load("ultralytics/yolov5", "yolov5x", trust_repo=True)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        exit(1)
    model.conf = 0.5  # NMS confidence threshold
    model.iou = 0.5  # NMS IoU threshold
    model.classes = [14]  # Filter by class 14: bird
    model.to(device)
    model.eval()

    # List videos in the S3 bucket
    s3 = boto3.client('s3', aws_access_key_id=Access_key_ID, aws_secret_access_key=Secret_access_key, region_name='your_region', config=Config(signature_version='s3v4'))
    response = s3.list_objects_v2(Bucket=input_bucket, Prefix=input_prefix)
    annotations = [] # Initialize an empty list to store annotations
    annotations_crop = []
    video_count = 0
    image_id = 0
    for obj in response.get('Contents', []):
        video_key = obj['Key']
        if video_key.lower().endswith(('.mp4', '.avi', '.mov')):
            print(f"Processing video: s3://{input_bucket}/{video_key}")
            video_results, crop_results = process_video_from_s3(s3, input_bucket, video_key, model, device, output_bucket, output_prefix, output_crop_prefix, video_count, image_id)
            annotations.extend(video_results) # Add the results of the current video to the annotations list
            annotations_crop.extend(crop_results)
            video_count += 1
    # Save results to JSON
    save_results_to_json(annotations, "annotations.json")
    save_results_to_json(annotations_crop, "annotations_crop.json")
    # Upload annotations to S3
    upload_json_to_s3(s3, output_bucket, annotation_key, annotations)
    upload_json_to_s3(s3, output_bucket, annotation_crop_key, annotations_crop)

if __name__ == "__main__":
    main()
