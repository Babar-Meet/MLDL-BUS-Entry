#!/usr/bin/env python3
"""
Automated Evaluation System for Number Plate Detection
Runs detection on test videos and generates evaluation report.
"""

import os
import sys
import time
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


# Configuration
SHOW_PREVIEW = False  # Set to True to show live preview window


# Global device
DEVICE = get_device()


class EvaluationDetector:
    """Detector for evaluation mode - no webcam, no display"""
    
    def __init__(self):
        self.model = None
        self.reader = None
        self.use_gpu = (DEVICE == "cuda")
        self.device = DEVICE
        
        self.output_dir = Path(os.path.join(BASE_DIR, "output"))
        self.snapshot_dir = Path(os.path.join(BASE_DIR, "snapshots"))
        self.output_dir.mkdir(exist_ok=True)
        self.snapshot_dir.mkdir(exist_ok=True)
        
        # Metrics
        self.total_frames = 0
        self.vehicle_detections = 0
        self.plate_detections = 0
        self.ocr_success = 0
        self.ocr_attempts = 0
        
        # Detailed OCR log
        self.detailed_ocr_log = []
        
        # Current video info
        self.current_video_name = ""
        self.current_video_fps = 30.0
        
        # Track unique vehicles to avoid multiple snapshots per vehicle
        self.captured_vehicles = {}  # {plate_number: {'frame': frame_count, 'confidence': confidence, 'clarity': clarity}}
        
        # Multi-frame consensus tracking
        self.frame_consensus = {}
        self.min_consensus_frames = 2
    
    def normalize_plate(self, plate_text: str) -> str:
        """Normalize plate text for consistent tracking"""
        if not plate_text:
            return ""
        normalized = "".join(c.upper() for c in plate_text if c.isalnum())
        return normalized
    
    def correct_position_based(self, text: str) -> str:
        """
        Apply position-aware character corrections for common OCR mistakes.
        Indian plate format: AA00AA0000
        """
        if not text or len(text) < 4:
            return text
        
        text = text.upper().strip()
        result = []
        
        for i, char in enumerate(text):
            if i < 2 or i > 3 and i < 6:  # Letter positions
                if char == '2':
                    result.append('Z')
                elif char == '0':
                    result.append('O')
                elif char == '1':
                    result.append('I')
                elif char == '5':
                    result.append('S')
                elif char == '8':
                    result.append('B')
                else:
                    result.append(char)
            else:  # Digit positions
                if char == 'Z':
                    result.append('2')
                elif char == 'O':
                    result.append('0')
                elif char == 'I':
                    result.append('1')
                elif char == 'S':
                    result.append('5')
                elif char == 'B':
                    result.append('8')
                else:
                    result.append(char)
        
        return ''.join(result)
    
    def validate_indian_plate(self, text: str) -> str:
        """Validate and format Indian license plate"""
        if not text:
            return ""
        
        cleaned = "".join(c for c in text.upper() if c.isalnum())
        
        if len(cleaned) < 4:
            return ""
        
        corrected = self.correct_position_based(cleaned)
        
        # Validate format
        if len(corrected) == 10:
            if corrected[:2].isalpha() and corrected[2:4].isdigit() and corrected[4:6].isalpha() and corrected[6:].isdigit():
                return corrected
        
        if len(corrected) == 11:
            if corrected[:2].isalpha() and corrected[2:4].isdigit() and corrected[4:7].isalpha() and corrected[7:].isdigit():
                return corrected
        
        if len(corrected) == 6:
            if corrected[:2].isalpha() and corrected[2:].isdigit():
                return corrected
        
        return corrected if corrected.isalnum() else ""
    
    def calculate_image_clarity(self, image) -> float:
        """Calculate image clarity using Laplacian variance (higher = sharper)"""
        if image is None or image.size == 0:
            return 0.0
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()
            return variance
        except:
            return 0.0
    
    def should_capture_vehicle(self, plate_number: str, image, ocr_confidence: float, frame_count: int) -> tuple:
        """
        Determine if we should capture this vehicle.
        Returns (should_capture, reason):
        - (True, "new") - new vehicle, capture it
        - (True, "better") - same vehicle but better quality
        - (False, reason) - don't capture
        """
        if not plate_number:
            return False, "no_plate"
        
        # Clean the plate number for consistent tracking
        clean_plate = plate_number.upper().strip()
        
        # If this is a new vehicle, capture it
        if clean_plate not in self.captured_vehicles:
            return True, "new"
        
        # Vehicle already captured - check if this is a better quality
        existing = self.captured_vehicles[clean_plate]
        
        # Calculate clarity for current image
        current_clarity = self.calculate_image_clarity(image)
        existing_clarity = existing.get('clarity', 0)
        
        # Consider OCR confidence and clarity
        current_score = (ocr_confidence * 0.6) + (current_clarity / 1000 * 0.4)
        existing_score = (existing.get('confidence', 0) * 0.6) + (existing_clarity / 1000 * 0.4)
        
        # If current is significantly better, update
        if current_score > existing_score * 1.2:  # 20% better
            return True, "better"
        
        return False, "already_captured"
    
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
    
    def detect_vehicles(self, frame, conf_threshold: float = 0.5):
        """Detect vehicles in frame"""
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
                    
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img_width, x2), min(img_height, y2)
                    
                    detections.append({
                        'box': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': confidence,
                        'class': cls
                    })
        
        return detections
    
    def detect_plates_in_vehicles(self, frame, vehicle_detections, frame_count):
        """Detect plates within vehicle detections"""
        plate_detections = []
        
        for veh in vehicle_detections:
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
            
            self.ocr_attempts += 1
            
            # Try OCR on the lower region
            try:
                results = self.reader.readtext(lower_region)
                
                if results:
                    for (bbox, text, prob) in results:
                        # Clean and validate text with position-aware corrections
                        text = text.strip()
                        text_clean = self.validate_indian_plate(text)
                        
                        if prob > 0.3 and len(text_clean) >= 4:
                            # Normalize for tracking
                            normalized = self.normalize_plate(text_clean)
                            
                            # Check if already captured
                            if normalized in self.captured_vehicles:
                                continue
                            
                            # Add to consensus
                            if normalized not in self.frame_consensus:
                                self.frame_consensus[normalized] = {
                                    'readings': [],
                                    'best_confidence': 0,
                                    'best_text': normalized
                                }
                            
                            self.frame_consensus[normalized]['readings'].append(prob)
                            if prob > self.frame_consensus[normalized]['best_confidence']:
                                self.frame_consensus[normalized]['best_confidence'] = prob
                            
                            # Only capture if we have consensus
                            if len(self.frame_consensus[normalized]['readings']) >= self.min_consensus_frames:
                                # Calculate clarity
                                clarity = self.calculate_image_clarity(lower_region)
                                
                                # Save cropped plate image
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                safe_video_name = "".join(c for c in self.current_video_name if c.isalnum())
                                plate_filename = f"ocr_{safe_video_name}_frame_{frame_count}_{timestamp}.jpg"
                                plate_path = self.snapshot_dir / plate_filename
                                cv2.imwrite(str(plate_path), lower_region)
                                
                                # Calculate timestamp
                                timestamp_sec = frame_count / self.current_video_fps
                                
                                plate_detections.append({
                                    'text': text,
                                    'text_clean': text_clean,
                                    'confidence': prob,
                                    'vehicle_box': veh['box'],
                                    'plate_image_path': str(plate_path),
                                    'frame_number': frame_count,
                                    'timestamp_sec': timestamp_sec
                                })
                                
                                # Track this vehicle
                                self.captured_vehicles[normalized] = {
                                    'frame': frame_count,
                                    'confidence': prob,
                                    'clarity': clarity,
                                    'plate_image_path': str(plate_path)
                                }
                            
                            # Log detailed OCR
                            log_entry = {
                                'video': self.current_video_name,
                                'frame': frame_count,
                                'time_sec': frame_count / self.current_video_fps,
                                'bbox': veh['box'],
                                'raw_ocr': text,
                                'cleaned_ocr': text_clean,
                                'confidence': prob,
                                'valid': len(text_clean) >= 6,
                                'captured': normalized in self.captured_vehicles
                            }
                            self.detailed_ocr_log.append(log_entry)
                else:
                    # OCR failed - log it
                    timestamp_sec = frame_count / self.current_video_fps
                    log_entry = {
                        'video': self.current_video_name,
                        'frame': frame_count,
                        'time_sec': timestamp_sec,
                        'bbox': veh['box'],
                        'raw_ocr': '',
                        'cleaned_ocr': '',
                        'confidence': 0.0,
                        'valid': False,
                        'failed': True,
                        'captured': False
                    }
                    self.detailed_ocr_log.append(log_entry)
                            
            except Exception as e:
                # OCR failed - log it
                timestamp_sec = frame_count / self.current_video_fps
                log_entry = {
                    'video': self.current_video_name,
                    'frame': frame_count,
                    'time_sec': timestamp_sec,
                    'bbox': veh['box'],
                    'raw_ocr': '',
                    'cleaned_ocr': '',
                    'confidence': 0.0,
                    'valid': False,
                    'failed': True,
                    'captured': False
                }
                self.detailed_ocr_log.append(log_entry)
        
        return plate_detections
    
    def process_video(self, video_path: str, show_preview: bool = False) -> dict:
        """Process a video and return metrics"""
        video_name = os.path.basename(video_path)
        self.current_video_name = video_name
        print(f"\n[PROCESSING] {video_name}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return None
        
        # Get total frame count and fps
        total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_video_fps = cap.get(cv2.CAP_PROP_FPS)
        if self.current_video_fps <= 0:
            self.current_video_fps = 30.0
        
        # Reset metrics
        self.total_frames = 0
        self.vehicle_detections = 0
        self.plate_detections = 0
        self.ocr_success = 0
        self.ocr_attempts = 0
        self.detailed_ocr_log = []
        
        # Reset captured vehicles and consensus for this video session
        self.captured_vehicles = {}
        self.frame_consensus = {}
        
        frame_skip = 5  # Process every 5th frame for speed
        
        # Create preview window if enabled
        if show_preview:
            cv2.namedWindow("Evaluation Preview", cv2.WINDOW_NORMAL)
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            self.total_frames += 1
            
            # Progress display
            if self.total_frames % 10 == 0:
                progress_pct = (self.total_frames / total_frame_count * 100) if total_frame_count > 0 else 0
                progress_bar = "=" * int(progress_pct // 5) + ">" + " " * (20 - int(progress_pct // 5))
                
                print(f"\r[Video: {video_name}] Frame {self.total_frames}/{total_frame_count} ({progress_pct:.1f}%) | "
                      f"Vehicles: {self.vehicle_detections} | Plates: {self.plate_detections} | OCR: {self.ocr_success} | "
                      f"[{progress_bar}]", end="", flush=True)
            
            if self.total_frames % frame_skip != 0:
                continue
            
            try:
                # Detect vehicles
                vehicles = self.detect_vehicles(frame)
                self.vehicle_detections += len(vehicles)
                
                # Detect plates within vehicles
                if vehicles:
                    plates = self.detect_plates_in_vehicles(frame, vehicles, self.total_frames)
                    self.plate_detections += len(plates)
                    
                    # Count OCR success (text length >= 6)
                    for plate in plates:
                        if len(plate['text_clean']) >= 6:
                            self.ocr_success += 1
                
                # Show preview if enabled
                if show_preview and vehicles:
                    preview_frame = frame.copy()
                    for veh in vehicles:
                        x1, y1, x2, y2 = veh['box']
                        cv2.rectangle(preview_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    preview_frame = cv2.resize(preview_frame, (640, 360))
                    cv2.imshow("Evaluation Preview", preview_frame)
                    cv2.waitKey(1)
            
            except Exception as e:
                # Continue on frame error
                pass
        
        cap.release()
        
        if show_preview:
            cv2.destroyAllWindows()
        
        # Print final progress
        progress_pct = (self.total_frames / total_frame_count * 100) if total_frame_count > 0 else 100
        progress_bar = "=" * 20
        print(f"\r[Video: {video_name}] Frame {self.total_frames}/{total_frame_count} (100.0%) | "
              f"Vehicles: {self.vehicle_detections} | Plates: {self.plate_detections} | OCR: {self.ocr_success} | "
              f"[{progress_bar}] DONE", flush=True)
        
        return {
            'video': video_name,
            'total_frames': self.total_frames,
            'vehicle_detections': self.vehicle_detections,
            'plate_detections': self.plate_detections,
            'ocr_success': self.ocr_success,
            'ocr_attempts': self.ocr_attempts,
            'unique_vehicles_captured': len(self.captured_vehicles)
        }


def download_sample_videos(test_videos_folder: str):
    """Download sample traffic videos"""
    import urllib.request
    import ssl
    
    # Public domain traffic videos (direct MP4 URLs)
    sample_urls = [
        # Small traffic sample videos
        ("https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4", "traffic1.mp4"),
        ("https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4", "traffic2.mp4"),
        ("https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4", "traffic3.mp4"),
    ]
    
    # Create SSL context that doesn't verify certificates (for some URLs)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    downloaded = []
    
    for url, filename in sample_urls:
        output_path = os.path.join(test_videos_folder, filename)
        
        if os.path.exists(output_path):
            print(f"[SKIP] Already exists: {filename}")
            downloaded.append(output_path)
            continue
        
        print(f"[DOWNLOADING] {filename}...")
        
        try:
            # Try with custom SSL context
            request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            response = urllib.request.urlopen(request, timeout=60, context=ssl_context)
            
            with open(output_path, 'wb') as f:
                f.write(response.read())
            
            # Verify file is valid
            if os.path.getsize(output_path) > 10000:
                print(f"[OK] Downloaded: {filename} ({os.path.getsize(output_path)} bytes)")
                downloaded.append(output_path)
            else:
                print(f"[ERROR] File too small: {filename}")
                if os.path.exists(output_path):
                    os.remove(output_path)
        except Exception as e:
            print(f"[ERROR] Download failed for {filename}: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
            continue
    
    return downloaded


def ensure_test_videos():
    """Ensure test_videos folder exists and has content"""
    test_videos_folder = os.path.join(BASE_DIR, "test_videos")
    os.makedirs(test_videos_folder, exist_ok=True)
    
    # Check for existing videos
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    existing_videos = []
    
    for file in os.listdir(test_videos_folder):
        ext = os.path.splitext(file)[1].lower()
        if ext in video_extensions:
            filepath = os.path.join(test_videos_folder, file)
            if os.path.getsize(filepath) > 10000:  # Must be > 10KB
                existing_videos.append(filepath)
    
    if existing_videos:
        print(f"[INFO] Found {len(existing_videos)} test video(s)")
        return test_videos_folder, existing_videos
    
    # Try to download sample videos
    print("[INFO] No test videos found, downloading sample videos...")
    downloaded = download_sample_videos(test_videos_folder)
    
    if not downloaded:
        print("\n[ERROR] FAILED TO DOWNLOAD ANY TEST VIDEOS")
        print("[ERROR] Please manually add .mp4 videos to test_videos folder")
        print("[ERROR] Or check internet connection and try again")
        sys.exit(1)
    
    print(f"[OK] Downloaded {len(downloaded)} video(s)")
    return test_videos_folder, downloaded


def save_evaluation_log(results: list, log_path: str):
    """Save evaluation results to log file"""
    with open(log_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("  NUMBER PLATE DETECTION EVALUATION REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write("=" * 50 + "\n\n")
        
        for r in results:
            f.write("=" * 50 + "\n")
            f.write(f"Video: {r['video']}\n")
            f.write(f"Total Frames: {r['total_frames']}\n")
            f.write(f"Vehicles Detected: {r['vehicle_detections']}\n")
            f.write(f"Plates Detected: {r['plate_detections']}\n")
            f.write(f"OCR Attempts: {r['ocr_attempts']}\n")
            f.write(f"Valid Plates (len>=6): {r['ocr_success']}\n")
            
            pdr = 0.0
            if r['vehicle_detections'] > 0:
                pdr = (r['plate_detections'] / r['vehicle_detections']) * 100
            f.write(f"Plate Detection Rate: {pdr:.1f}%\n")
            
            osr = 0.0
            if r['plate_detections'] > 0:
                osr = (r['ocr_success'] / r['plate_detections']) * 100
            f.write(f"Valid Plate Rate: {osr:.1f}%\n")
            
            if r['vehicle_detections'] == 0:
                f.write("WARNING: No vehicles detected in this video.\n")
            
            f.write("=" * 50 + "\n\n")
        
        # Summary
        if results:
            total_frames = sum(r['total_frames'] for r in results)
            total_vehicles = sum(r['vehicle_detections'] for r in results)
            total_plates = sum(r['plate_detections'] for r in results)
            total_ocr_attempts = sum(r['ocr_attempts'] for r in results)
            total_valid = sum(r['ocr_success'] for r in results)
            
            avg_pdr = sum((r['plate_detections']/r['vehicle_detections']*100 if r['vehicle_detections']>0 else 0) for r in results) / len(results)
            avg_osr = sum((r['ocr_success']/r['plate_detections']*100 if r['plate_detections']>0 else 0) for r in results) / len(results)
            
            f.write("=" * 50 + "\n")
            f.write("SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Videos Processed: {len(results)}\n")
            f.write(f"Total Frames: {total_frames}\n")
            f.write(f"Total Vehicles: {total_vehicles}\n")
            f.write(f"Total Plates: {total_plates}\n")
            f.write(f"Total OCR Attempts: {total_ocr_attempts}\n")
            f.write(f"Total Valid Plates (len>=6): {total_valid}\n")
            f.write(f"Average Plate Detection Rate: {avg_pdr:.1f}%\n")
            f.write(f"Average Valid Plate Rate: {avg_osr:.1f}%\n")
            f.write("=" * 50 + "\n")


def save_detailed_ocr_log(all_ocr_logs: list, log_path: str):
    """Save detailed OCR results to log file"""
    with open(log_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("  DETAILED OCR LOG\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write("=" * 60 + "\n\n")
        
        for entry in all_ocr_logs:
            f.write("-" * 60 + "\n")
            f.write(f"Video: {entry['video']}\n")
            f.write(f"Frame: {entry['frame']}\n")
            f.write(f"Time: {entry['time_sec']:.2f} sec\n")
            
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
            
            if entry['valid']:
                f.write("Valid Plate: YES\n")
            else:
                f.write("Valid Plate: NO\n")
            
            f.write("-" * 60 + "\n\n")
        
        # Summary
        total_entries = len(all_ocr_logs)
        valid_entries = sum(1 for e in all_ocr_logs if e.get('valid', False))
        failed_entries = sum(1 for e in all_ocr_logs if e.get('failed', False))
        
        f.write("=" * 60 + "\n")
        f.write("OCR SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total OCR Entries: {total_entries}\n")
        f.write(f"Valid Plates (len>=6): {valid_entries}\n")
        f.write(f"Failed OCR: {failed_entries}\n")
        f.write(f"Invalid/Short: {total_entries - valid_entries - failed_entries}\n")
        
        if total_entries > 0:
            valid_rate = (valid_entries / total_entries) * 100
            f.write(f"Valid Rate: {valid_rate:.1f}%\n")
        
        f.write("=" * 60 + "\n")


def print_summary(results: list):
    """Print summary table to console"""
    print("\n" + "=" * 90)
    print("  EVALUATION RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Video':<25} {'Frames':<10} {'Vehicles':<10} {'Plates':<10} {'Valid':<10} {'PDR%':<10} {'VPR%':<10}")
    print("-" * 90)
    
    for r in results:
        pdr = 0.0
        if r['vehicle_detections'] > 0:
            pdr = (r['plate_detections'] / r['vehicle_detections']) * 100
        
        vpr = 0.0
        if r['plate_detections'] > 0:
            vpr = (r['ocr_success'] / r['plate_detections']) * 100
        
        warning = ""
        if r['vehicle_detections'] == 0:
            warning = " [NO DETECTIONS]"
        
        print(f"{r['video'][:24] + warning:<25} {r['total_frames']:<10} {r['vehicle_detections']:<10} "
              f"{r['plate_detections']:<10} {r['ocr_success']:<10} "
              f"{pdr:<10.1f} {vpr:<10.1f}")
        
        if r['vehicle_detections'] == 0:
            print(f"  WARNING: No vehicles detected in this video.")
    
    print("-" * 90)
    
    if results:
        total_frames = sum(r['total_frames'] for r in results)
        total_vehicles = sum(r['vehicle_detections'] for r in results)
        total_plates = sum(r['plate_detections'] for r in results)
        total_valid = sum(r['ocr_success'] for r in results)
        
        avg_pdr = sum((r['plate_detections']/r['vehicle_detections']*100 if r['vehicle_detections']>0 else 0) for r in results) / len(results)
        avg_vpr = sum((r['ocr_success']/r['plate_detections']*100 if r['plate_detections']>0 else 0) for r in results) / len(results)
        
        print(f"{'TOTAL/AVERAGE':<25} {total_frames:<10} {total_vehicles:<10} "
              f"{total_plates:<10} {total_valid:<10} "
              f"{avg_pdr:<10.1f} {avg_vpr:<10.1f}")
    
    print("=" * 90)


def main():
    """Main evaluation function"""
    print("\n" + "=" * 50)
    print("  AUTOMATED EVALUATION SYSTEM")
    print("  Number Plate Detection Testing")
    print("=" * 50)
    print(f"Running from: {BASE_DIR}")
    print(f"Preview: {'Enabled' if SHOW_PREVIEW else 'Disabled'}")
    print()
    
    # Ensure test videos exist
    test_folder, test_videos = ensure_test_videos()
    
    if not test_videos:
        print("[ERROR] No test videos available")
        sys.exit(1)
    
    # Create logs folder
    logs_folder = os.path.join(BASE_DIR, "logs")
    os.makedirs(logs_folder, exist_ok=True)
    
    # Initialize detector
    detector = EvaluationDetector()
    
    try:
        detector.load_models()
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        sys.exit(1)
    
    # Process each video
    results = []
    all_ocr_logs = []
    
    for video_path in test_videos:
        try:
            result = detector.process_video(video_path, show_preview=SHOW_PREVIEW)
            if result:
                results.append(result)
                all_ocr_logs.extend(detector.detailed_ocr_log)
                
                if result['vehicle_detections'] == 0:
                    print(f"[WARNING] {result['video']}: No vehicles detected")
                else:
                    pdr = (result['plate_detections'] / result['vehicle_detections']) * 100
                    vpr = (result['ocr_success'] / result['plate_detections'] * 100) if result['plate_detections'] > 0 else 0
                    print(f"[OK] {result['video']}: {result['vehicle_detections']} vehicles, "
                          f"{result['plate_detections']} plates, {result['ocr_success']} valid "
                          f"(PDR: {pdr:.1f}%, VPR: {vpr:.1f}%)")
        except Exception as e:
            print(f"[ERROR] Failed to process {video_path}: {e}")
            continue
    
    if not results:
        print("[ERROR] No videos processed successfully")
        sys.exit(1)
    
    # Save evaluation log (summary)
    log_path = os.path.join(logs_folder, "evaluation_log.txt")
    save_evaluation_log(results, log_path)
    print(f"\n[OK] Summary log saved to: {log_path}")
    
    # Save detailed OCR log
    ocr_log_path = os.path.join(logs_folder, "ocr_detailed_log.txt")
    save_detailed_ocr_log(all_ocr_logs, ocr_log_path)
    print(f"[OK] Detailed OCR log saved to: {ocr_log_path}")
    
    # Print summary
    print_summary(results)
    
    print("\n" + "=" * 50)
    print("  EVALUATION COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
