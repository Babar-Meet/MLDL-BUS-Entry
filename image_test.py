#!/usr/bin/env python3
"""
Image Testing System for Number Plate Detection
Tests detection and OCR on images in test_images folder.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Get BASE_DIR
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add to path
sys.path.insert(0, BASE_DIR)

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import easyocr
from PIL import Image, ImageDraw


# GPU Detection
def get_device():
    """Detect and return compute device"""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Torch CUDA Available: True")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print(f"Torch CUDA Available: False")
        print(f"Using CPU")
    return device


# Global device
DEVICE = get_device()


class ImageTester:
    """Image testing system for number plate detection"""
    
    def __init__(self):
        self.model = None
        self.reader = None
        self.use_gpu = (DEVICE == "cuda")
        self.device = DEVICE
        
        # Create required directories
        self.test_images_dir = Path(os.path.join(BASE_DIR, "test_images"))
        self.snapshot_dir = Path(os.path.join(BASE_DIR, "snapshots"))
        self.logs_dir = Path(os.path.join(BASE_DIR, "logs"))
        
        self.test_images_dir.mkdir(exist_ok=True)
        self.snapshot_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Metrics
        self.total_images = 0
        self.total_vehicles = 0
        self.total_plates = 0
        self.ocr_success = 0
        
        # Detailed OCR log
        self.detailed_ocr_log = []
    
    def load_models(self):
        """Load YOLO and OCR models"""
        print("[INFO] Loading YOLOv8 model...")
        
        models_folder = os.path.join(BASE_DIR, "models")
        model_path = os.path.join(models_folder, "yolov8n.pt")
        
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            self.model = YOLO('yolov8n.pt')
        
        # Move model to GPU if available
        if self.device == "cuda":
            self.model.to(self.device)
            print(f"[INFO] YOLOv8 model moved to GPU")
        else:
            print(f"[INFO] YOLOv8 model on CPU")
        
        print("[INFO] Loading EasyOCR model...")
        self.reader = easyocr.Reader(['en'], gpu=self.use_gpu)
        print("[OK] Models loaded")
    
    def clean_ocr_text(self, text: str) -> str:
        """Clean OCR text"""
        if not text:
            return ""
        text = text.strip()
        cleaned = "".join(c for c in text if c.isalnum())
        return cleaned
    
    def detect_vehicles(self, frame, conf_threshold: float = 0.5):
        """Detect vehicles in image"""
        results = self.model(frame, conf=conf_threshold, iou=0.45, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if len(boxes) == 0:
                continue
            
            img_height, img_width = frame.shape[:2]
            
            for box in boxes:
                cls = int(box.cls[0])
                # Class 2 = car, 3 = motorcycle, 5 = bus, 7 = truck in COCO
                if cls in [2, 3, 5, 7]:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(img_width, int(x2)), min(img_height, int(y2))
                    
                    detections.append({
                        'box': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'class': cls
                    })
        
        return detections
    
    def detect_plates_in_vehicles(self, frame, vehicle_detections):
        """Detect plates within vehicle detections"""
        plate_detections = []
        
        for idx, veh in enumerate(vehicle_detections):
            x1, y1, x2, y2 = veh['box']
            
            # Crop vehicle region
            vehicle_crop = frame[y1:y2, x1:x2]
            
            if vehicle_crop.size == 0:
                continue
            
            # Look for plates (typically in lower portion of vehicle)
            h, w = vehicle_crop.shape[:2]
            if h < 20 or w < 20:
                continue
            
            lower_region = vehicle_crop[int(h*0.5):, :]
            
            if lower_region.size == 0:
                continue
            
            # Try OCR
            try:
                results = self.reader.readtext(lower_region)
                
                if results:
                    for (bbox, text, prob) in results:
                        text_clean = self.clean_ocr_text(text)
                        
                        if prob > 0.3 and len(text_clean) >= 4:
                            # Calculate plate position relative to original image
                            plate_x1 = x1
                            plate_y1 = y1 + int(h * 0.5) + bbox[0][0]
                            plate_y2 = plate_y1 + (bbox[1][1] - bbox[0][1])
                            plate_x2 = x1 + bbox[1][0]
                            
                            plate_detections.append({
                                'text': text,
                                'text_clean': text_clean,
                                'confidence': prob,
                                'vehicle_idx': idx,
                                'plate_crop': lower_region.copy(),
                                'plate_box': (plate_x1, plate_y1, plate_x2, plate_y2)
                            })
                else:
                    # OCR failed
                    plate_detections.append({
                        'text': '',
                        'text_clean': '',
                        'confidence': 0.0,
                        'vehicle_idx': idx,
                        'plate_crop': lower_region.copy(),
                        'plate_box': None,
                        'failed': True
                    })
                            
            except Exception as e:
                plate_detections.append({
                    'text': '',
                    'text_clean': '',
                    'confidence': 0.0,
                    'vehicle_idx': idx,
                    'plate_crop': lower_region.copy(),
                    'plate_box': None,
                    'failed': True
                })
        
        return plate_detections
    
    def draw_annotations(self, image, vehicles, plates):
        """Draw bounding boxes on image"""
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        # Draw vehicle boxes (green)
        for veh in vehicles:
            x1, y1, x2, y2 = veh['box']
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        
        # Draw plate boxes (red) and labels
        for i, plate in enumerate(plates):
            if plate.get('failed', False):
                continue
            
            if plate['plate_box']:
                x1, y1, x2, y2 = plate['plate_box']
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
                
                # Draw OCR text
                text = plate['text_clean']
                if text:
                    draw.rectangle([x1, y1 - 20, x1 + len(text) * 10, y1], fill=(255, 0, 0))
                    draw.text((x1, y1 - 20), text, fill=(255, 255, 255))
        
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    def process_image(self, image_path: str):
        """Process a single image"""
        image_name = os.path.basename(image_path)
        
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[ERROR] Cannot load image: {image_name}")
            return None
        
        # Detect vehicles
        vehicles = self.detect_vehicles(frame)
        self.total_vehicles += len(vehicles)
        
        # Detect plates
        plates = self.detect_plates_in_vehicles(frame, vehicles)
        self.total_plates += len(plates)
        
        # Process each plate for OCR success count and logging
        for i, plate in enumerate(plates):
            if not plate.get('failed', False) and len(plate['text_clean']) >= 6:
                self.ocr_success += 1
                
                # Log detailed OCR
                log_entry = {
                    'image': image_name,
                    'plate_index': i + 1,
                    'bbox': plate['plate_box'],
                    'raw_ocr': plate['text'],
                    'cleaned_ocr': plate['text_clean'],
                    'confidence': plate['confidence']
                }
                self.detailed_ocr_log.append(log_entry)
            else:
                # Log failed OCR
                log_entry = {
                    'image': image_name,
                    'plate_index': i + 1,
                    'bbox': plate['plate_box'] if plate.get('plate_box') else (0, 0, 0, 0),
                    'raw_ocr': '',
                    'cleaned_ocr': '',
                    'confidence': 0.0,
                    'failed': True
                }
                self.detailed_ocr_log.append(log_entry)
        
        # Save annotated image
        annotated = self.draw_annotations(frame, vehicles, plates)
        safe_name = "".join(c for c in image_name if c.isalnum())
        annotated_path = self.snapshot_dir / f"annotated_{safe_name}"
        cv2.imwrite(str(annotated_path), annotated)
        
        # Save cropped plates
        for i, plate in enumerate(plates):
            if 'plate_crop' in plate and plate['plate_crop'] is not None:
                crop_path = self.snapshot_dir / f"crop_{safe_name}_{i+1}.jpg"
                cv2.imwrite(str(crop_path), plate['plate_crop'])
        
        return {
            'vehicles': len(vehicles),
            'plates': len(plates)
        }
    
    def get_images(self):
        """Get list of images from test_images folder"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        images = []
        
        if not self.test_images_dir.exists():
            return images
        
        for file in self.test_images_dir.iterdir():
            if file.is_file() and file.suffix.lower() in image_extensions:
                images.append(str(file))
        
        return sorted(images)


def save_image_ocr_log(log_entries: list, log_path: str):
    """Save detailed OCR log for images"""
    with open(log_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("  IMAGE OCR DETAILED LOG\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write("=" * 60 + "\n\n")
        
        for entry in log_entries:
            f.write("-" * 60 + "\n")
            f.write(f"Image: {entry['image']}\n")
            f.write(f"Plate Index: {entry['plate_index']}\n")
            
            bbox = entry['bbox']
            f.write(f"BBox: x1={bbox[0]} y1={bbox[1]} x2={bbox[2]} y2={bbox[3]}\n")
            
            if entry.get('failed', False):
                f.write("Raw OCR: OCR FAILED\n")
                f.write("Cleaned OCR: OCR FAILED\n")
                f.write("Confidence: N/A\n")
            else:
                f.write(f"Raw OCR: \"{entry['raw_ocr']}\"\n")
                f.write(f"Cleaned OCR: {entry['cleaned_ocr']}\n")
                f.write(f"Confidence: {entry['confidence']:.2f}\n")
            
            f.write("-" * 60 + "\n\n")
        
        # Summary
        total = len(log_entries)
        valid = sum(1 for e in log_entries if not e.get('failed', False) and len(e['cleaned_ocr']) >= 6)
        failed = sum(1 for e in log_entries if e.get('failed', False))
        
        f.write("=" * 60 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total Plates Detected: {total}\n")
        f.write(f"Valid Plates (len>=6): {valid}\n")
        f.write(f"Failed OCR: {failed}\n")
        
        if total > 0:
            valid_rate = (valid / total) * 100
            f.write(f"Valid Rate: {valid_rate:.1f}%\n")
        
        f.write("=" * 60 + "\n")


def save_image_summary(results: list, log_path: str):
    """Save image test summary"""
    with open(log_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("  IMAGE TEST SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        for r in results:
            f.write(f"Image: {r['image']}\n")
            f.write(f"  Vehicles: {r['vehicles']}\n")
            f.write(f"  Plates: {r['plates']}\n")
            f.write(f"  OCR Success: {r['ocr_success']}\n\n")
        
        # Total
        total_images = len(results)
        total_vehicles = sum(r['vehicles'] for r in results)
        total_plates = sum(r['plates'] for r in results)
        total_ocr = sum(r['ocr_success'] for r in results)
        
        f.write("=" * 60 + "\n")
        f.write("TOTALS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total Images Processed: {total_images}\n")
        f.write(f"Total Vehicles: {total_vehicles}\n")
        f.write(f"Total Plates: {total_plates}\n")
        f.write(f"Total OCR Success: {total_ocr}\n")
        f.write("=" * 60 + "\n")


def main():
    """Main image testing function"""
    print("\n" + "=" * 50)
    print("  IMAGE TESTING SYSTEM")
    print("  Number Plate Detection on Images")
    print("=" * 50)
    print(f"Running from: {BASE_DIR}")
    print()
    
    # Initialize tester
    tester = ImageTester()
    
    # Get images
    images = tester.get_images()
    
    if not images:
        print("No images found in test_images folder.")
        print("Please add .jpg, .jpeg, or .png images to test_images folder.")
        sys.exit(0)
    
    print(f"[INFO] Found {len(images)} image(s)")
    
    # Load models
    try:
        tester.load_models()
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        sys.exit(1)
    
    # Process each image
    results = []
    
    for idx, image_path in enumerate(images, 1):
        image_name = os.path.basename(image_path)
        print(f"\rProcessing: {image_name} ({idx}/{len(images)})", end="", flush=True)
        
        try:
            result = tester.process_image(image_path)
            if result:
                result['image'] = image_name
                results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Failed to process {image_name}: {e}")
            continue
    
    print()  # New line after progress
    
    # Save logs
    ocr_log_path = tester.logs_dir / "image_ocr_log.txt"
    save_image_ocr_log(tester.detailed_ocr_log, str(ocr_log_path))
    print(f"[OK] OCR log saved to: {ocr_log_path}")
    
    summary_log_path = tester.logs_dir / "image_test_summary.txt"
    save_image_summary(results, str(summary_log_path))
    print(f"[OK] Summary saved to: {summary_log_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("  IMAGE TEST SUMMARY")
    print("=" * 60)
    print(f"{'Image':<30} {'Vehicles':<12} {'Plates':<12} {'OCR':<12}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['image'][:29]:<30} {r['vehicles']:<12} {r['plates']:<12} {r['ocr_success']:<12}")
    
    print("-" * 60)
    
    total_images = len(results)
    total_vehicles = sum(r['vehicles'] for r in results)
    total_plates = sum(r['plates'] for r in results)
    total_ocr = sum(r['ocr_success'] for r in results)
    
    print(f"{'TOTAL':<30} {total_vehicles:<12} {total_plates:<12} {total_ocr:<12}")
    print("=" * 60)
    
    print("\n" + "=" * 50)
    print("  IMAGE TEST COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
