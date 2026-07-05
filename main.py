#!/usr/bin/env python3
"""
Indian Number Plate Detection System - Main Entry Point
Fully automated with pretrained models, auto demo generation, and proper path handling.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import time
import sqlite3

# Get BASE_DIR - the project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add utils to path for imports
sys.path.insert(0, os.path.join(BASE_DIR, "utils"))

import cv2
import numpy as np
from PIL import Image, ImageDraw

# Import detection modules
import torch
from ultralytics import YOLO
import easyocr

# Import our modules
from database.init_db import init_database
from utils.demo_generator import (
    check_input_folder,
    get_first_input_file,
    create_demo_image
)


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


class DatabaseManager:
    """SQLite database manager for vehicle logs with proper path handling"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(BASE_DIR, "database", "vehicles.db")
        self.db_path = db_path
        self.ensure_database()
    
    def ensure_database(self):
        """Create database and tables if they don't exist"""
        db_folder = os.path.dirname(self.db_path)
        os.makedirs(db_folder, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicles_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_number TEXT NOT NULL,
                vehicle_type TEXT NOT NULL,
                detection_date DATE NOT NULL,
                detection_time TIME NOT NULL,
                snapshot_path TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_vehicle_number ON vehicles_log(vehicle_number)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detection_date ON vehicles_log(detection_date)')
        
        conn.commit()
        conn.close()
    
    def log_detection(self, vehicle_number: str, vehicle_type: str, 
                      snapshot_path: str, confidence: float) -> bool:
        """Log a vehicle detection to database"""
        try:
            now = datetime.now()
            date = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO vehicles_log 
                (vehicle_number, vehicle_type, detection_date, detection_time, 
                 snapshot_path, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (vehicle_number, vehicle_type, date, time_str, snapshot_path, confidence))
            
            conn.commit()
            conn.close()
            
            print(f"[DATABASE] Logged: {vehicle_number} at {date} {time_str}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Database error: {e}")
            return False
    
    def get_all_logs(self) -> list:
        """Get all vehicle logs"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM vehicles_log ORDER BY created_at DESC')
        rows = cursor.fetchall()
        
        conn.close()
        return [dict(row) for row in rows]


class OCREngine:
    """EasyOCR text extraction engine with enhanced post-processing"""
    
    def __init__(self, use_gpu: bool = None):
        self.reader = None
        # Use global DEVICE if not specified
        if use_gpu is None:
            use_gpu = (DEVICE == "cuda")
        self.use_gpu = use_gpu
    
    def load(self):
        if self.reader is None:
            print("[INFO] Loading EasyOCR model...")
            self.reader = easyocr.Reader(['en'], gpu=self.use_gpu)
            print("[INFO] OCR model loaded!")
        return self.reader
    
    def correct_position_based(self, text: str) -> str:
        """
        Apply position-aware character corrections for common OCR mistakes.
        Indian plate format: AA 00 AA 0000 (2 letters, 2 digits, 2 letters, 4 digits)
        """
        if not text or len(text) < 4:
            return text
        
        text = text.upper().strip()
        result = []
        
        for i, char in enumerate(text):
            if i < 2 or i > 3 and i < 6:  # Letter positions (0-1, 4-5)
                # Letter positions - fix common confusions
                if char == '2':
                    result.append('Z')  # Z commonly confused with 2
                elif char == '0':
                    result.append('O')  # O commonly confused with 0
                elif char == '1':
                    result.append('I')  # I commonly confused with 1
                elif char == '5':
                    result.append('S')  # S commonly confused with 5
                elif char == '8':
                    result.append('B')  # B commonly confused with 8
                else:
                    result.append(char)
            else:  # Digit positions (2-3, 6-9)
                # Digit positions - fix common confusions
                if char == 'Z':
                    result.append('2')  # 2 confused as Z
                elif char == 'O':
                    result.append('0')  # 0 confused as O
                elif char == 'I':
                    result.append('1')  # 1 confused as I
                elif char == 'S':
                    result.append('5')  # 5 confused as S
                elif char == 'B':
                    result.append('8')  # 8 confused as B
                else:
                    result.append(char)
        
        return ''.join(result)
    
    def validate_indian_plate(self, text: str) -> str:
        """
        Validate and format Indian license plate.
        Formats: AA00AA0000, AA00AAA0000, AA00000000
        """
        if not text:
            return ""
        
        # Remove all spaces and special characters
        cleaned = "".join(c for c in text.upper() if c.isalnum())
        
        if len(cleaned) < 4:
            return ""
        
        # Apply position-based corrections
        corrected = self.correct_position_based(cleaned)
        
        # Try to match Indian plate patterns
        # Pattern 1: AA00AA0000 (10 chars) - Most common
        if len(corrected) == 10:
            # Verify format: AA 00 AA 0000
            if corrected[:2].isalpha() and corrected[2:4].isdigit() and corrected[4:6].isalpha() and corrected[6:].isdigit():
                return corrected
        
        # Pattern 2: AA00AAA0000 (11 chars)
        if len(corrected) == 11:
            if corrected[:2].isalpha() and corrected[2:4].isdigit() and corrected[4:7].isalpha() and corrected[7:].isdigit():
                return corrected
        
        # Pattern 3: AA0000 (6 chars) - Old format
        if len(corrected) == 6:
            if corrected[:2].isalpha() and corrected[2:].isdigit():
                return corrected
        
        # Try to fix common issues and re-validate
        # If we have close to valid length, try to fix
        if 9 <= len(corrected) <= 11:
            # Try removing extra chars
            if len(corrected) == 11:
                # Remove middle character if it doesn't fit
                test = corrected[:4] + corrected[7:]
                if test.isalnum():
                    return self.validate_indian_plate(test)
            elif len(corrected) == 9:
                # Try adding a digit or letter
                test = corrected[:6] + '0' + corrected[6:]
                return self.validate_indian_plate(test)
        
        # Return the best effort cleaned version
        return corrected if corrected.isalnum() else ""
    
    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        
        text = text.replace(" ", "")
        cleaned = "".join(char for char in text if char.isalnum())
        
        # Validate and format as Indian plate
        validated = self.validate_indian_plate(cleaned)
        
        return validated
    
    def format_indian_plate(self, text: str) -> str:
        if len(text) < 4:
            return text
        
        state_codes = ['AP', 'AR', 'AS', 'BR', 'CG', 'DL', 'GA', 'GJ', 'HR', 'HP',
                       'JH', 'JK', 'KA', 'KL', 'LD', 'MH', 'ML', 'MN', 'MP', 'MZ',
                       'NL', 'OD', 'PB', 'PY', 'RJ', 'SK', 'TN', 'TR', 'TS', 'UP', 'WB']
        
        text = text.upper()
        
        if len(text) >= 2:
            first_two = text[:2]
            if first_two in state_codes:
                remaining = text[2:]
                if len(remaining) >= 2 and remaining[:2].isdigit():
                    digits = remaining[:2]
                    rest = remaining[2:]
                    if len(rest) >= 2:
                        letters = rest[:2]
                        numbers = rest[2:]
                        return f"{first_two}{digits}{letters}{numbers}".strip()
                    return f"{first_two}{digits}{rest}".strip()
        
        return text
    
    def extract_text(self, image) -> dict:
        if not self.reader:
            self.load()
        
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                return {'text': '', 'confidence': 0.0, 'raw_results': []}
        else:
            img = image
        
        results = self.reader.readtext(img)
        
        if not results:
            return {'text': '', 'confidence': 0.0, 'raw_results': []}
        
        raw_texts = []
        confidences = []
        
        for (bbox, text, prob) in results:
            raw_texts.append(text)
            confidences.append(prob)
        
        raw_combined = ' '.join(raw_texts)
        cleaned_text = self.clean_text(raw_combined)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            'text': cleaned_text,
            'raw_text': raw_combined,
            'confidence': avg_confidence,
            'raw_results': results
        }


class PlateDetector:
    """License plate detection and OCR system"""
    
    def __init__(self, model_path: str = None, use_gpu: bool = None):
        self.model = None
        self.model_path = model_path
        # Use global DEVICE if not specified
        if use_gpu is None:
            use_gpu = (DEVICE == "cuda")
        self.use_gpu = use_gpu
        self.device = DEVICE
        
        self.output_dir = Path(os.path.join(BASE_DIR, "output"))
        self.snapshot_dir = Path(os.path.join(BASE_DIR, "snapshots"))
        
        self.output_dir.mkdir(exist_ok=True)
        self.snapshot_dir.mkdir(exist_ok=True)
        
        self.ocr_engine = OCREngine(use_gpu=use_gpu)
        self.db_manager = DatabaseManager()
        
        self.recent_detections = {}
        self.duplicate_window = 10
        
        # Multi-frame consensus tracking
        self.frame_consensus = {}  # {normalized_plate: {'readings': [], 'best_confidence': 0, 'best_text': ''}}
        self.min_consensus_frames = 2  # Require at least 2 matching readings
        
        # Track unique vehicles to avoid multiple snapshots
        self.captured_vehicles = {}  # {plate_number: {'frame': frame_count, 'confidence': ocr_conf, 'image': img, 'box': box}}
        self.session_frame_count = 0
    
    def normalize_plate(self, plate_text: str) -> str:
        """Normalize plate text for consistent tracking"""
        if not plate_text:
            return ""
        # Remove spaces, convert to uppercase, keep only alphanumeric
        normalized = "".join(c.upper() for c in plate_text if c.isalnum())
        return normalized
    
    def add_to_consensus(self, plate_text: str, confidence: float) -> str:
        """
        Add a reading to multi-frame consensus.
        Returns the consensus plate if stable, otherwise empty string.
        """
        if not plate_text:
            return ""
        
        normalized = self.normalize_plate(plate_text)
        if not normalized or len(normalized) < 4:
            return ""
        
        if normalized not in self.frame_consensus:
            self.frame_consensus[normalized] = {
                'readings': [],
                'best_confidence': 0,
                'best_text': ''
            }
        
        # Add this reading
        self.frame_consensus[normalized]['readings'].append(confidence)
        
        # Update best if this is better
        if confidence > self.frame_consensus[normalized]['best_confidence']:
            self.frame_consensus[normalized]['best_confidence'] = confidence
            self.frame_consensus[normalized]['best_text'] = normalized
        
        # Check if we have consensus (multiple consistent readings)
        if len(self.frame_consensus[normalized]['readings']) >= self.min_consensus_frames:
            return normalized
        
        return ""
    
    def get_consensus_plate(self) -> tuple:
        """Get the best plate from consensus that has stable readings"""
        best_plate = ""
        best_confidence = 0
        
        for plate, data in self.frame_consensus.items():
            if len(data['readings']) >= self.min_consensus_frames:
                avg_conf = sum(data['readings']) / len(data['readings'])
                if avg_conf > best_confidence:
                    best_confidence = avg_conf
                    best_plate = data['best_text']
        
        return best_plate, best_confidence
    
    def clear_consensus(self):
        """Clear consensus tracking"""
        self.frame_consensus = {}
    
    def load_model(self, model_name: str = "yolov8n.pt"):
        """Load YOLOv8 model"""
        print(f"[INFO] Loading YOLOv8 model: {model_name}")
        
        models_folder = os.path.join(BASE_DIR, "models")
        
        if self.model_path and os.path.exists(self.model_path):
            self.model = YOLO(self.model_path)
        elif os.path.exists(os.path.join(models_folder, model_name)):
            self.model = YOLO(os.path.join(models_folder, model_name))
        else:
            print(f"[INFO] Downloading YOLOv8n model...")
            self.model = YOLO('yolov8n.pt')
            
            try:
                os.makedirs(models_folder, exist_ok=True)
                print(f"[INFO] Model loaded from Ultralytics")
            except Exception as e:
                print(f"[WARNING] Could not save model locally: {e}")
        
        # Move model to GPU if available
        if self.device == "cuda":
            self.model.to(self.device)
            print(f"[INFO] YOLOv8 model moved to GPU")
        else:
            print(f"[INFO] YOLOv8 model on CPU")
        
        print("[INFO] YOLOv8 model loaded!")
        
        self.ocr_engine.load()
        
        return self.model
    
    def detect_plates(self, image, conf_threshold: float = 0.5):
        """Detect plates in an image or frame"""
        results = self.model(image, conf=conf_threshold, iou=0.45, verbose=False)
        
        detections = []
        img = None
        
        for result in results:
            boxes = result.boxes
            
            if len(boxes) == 0:
                continue
            
            if isinstance(image, str):
                img = cv2.imread(image)
                if img is None:
                    continue
                img_height, img_width = img.shape[:2]
            else:
                img = image.copy()
                img_height, img_width = img.shape[:2]
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_width, x2), min(img_height, y2)
                
                detections.append({
                    'box': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': confidence
                })
        
        return detections, img
    
    def process_plate(self, image, box: tuple, frame_count: int = 0) -> dict:
        """Process a single plate: crop, OCR, save"""
        x1, y1, x2, y2 = box
        
        cropped = image[y1:y2, x1:x2]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plate_filename = f"plate_{timestamp}_{frame_count}.jpg"
        plate_path = self.snapshot_dir / plate_filename
        
        cv2.imwrite(str(plate_path), cropped)
        
        ocr_result = self.ocr_engine.extract_text(cropped)
        
        return {
            'plate_image': cropped,
            'plate_path': str(plate_path),
            'ocr_result': ocr_result
        }
    
    def draw_results(self, image, detections: list, ocr_results: list = None) -> np.ndarray:
        """Draw bounding boxes and text on image"""
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        box_color = (0, 255, 0)
        text_color = (0, 0, 0)
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['box']
            conf = det['confidence']
            
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
            
            label = f"Plate: {conf:.2f}"
            if ocr_results and i < len(ocr_results):
                text = ocr_results[i].get('text', '')
                if text:
                    label = f"{text}"
            
            text_bbox = draw.textbbox((x1, y1), label)
            draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill=box_color)
            draw.text((x1, y1 - 18), label, fill=text_color)
        
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    def save_snapshot(self, image, vehicle_number: str = "", ocr_confidence: float = 0.0) -> str:
        """Save detection snapshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if vehicle_number:
            safe_number = "".join(c for c in vehicle_number if c.isalnum())
            filename = f"snapshot_{safe_number}_{timestamp}.jpg"
        else:
            filename = f"snapshot_{timestamp}.jpg"
        
        snapshot_path = self.output_dir / filename
        cv2.imwrite(str(snapshot_path), image)
        
        return str(snapshot_path)
    
    def is_duplicate(self, plate_number: str) -> bool:
        """Check if plate number was detected within duplicate window"""
        if not plate_number:
            return False
        
        current_time = time.time()
        
        self.recent_detections = {
            k: v for k, v in self.recent_detections.items()
            if current_time - v < self.duplicate_window
        }
        
        if plate_number in self.recent_detections:
            return True
        
        self.recent_detections[plate_number] = current_time
        return False
    
    def log_vehicle(self, vehicle_number: str, snapshot_path: str, confidence: float):
        """Log vehicle to database with duplicate prevention"""
        if self.is_duplicate(vehicle_number):
            print(f"[SKIP] Duplicate detection: {vehicle_number} (within {self.duplicate_window}s)")
            return False
        
        vehicle_type = "unknown"
        
        success = self.db_manager.log_detection(
            vehicle_number=vehicle_number,
            vehicle_type=vehicle_type,
            snapshot_path=snapshot_path,
            confidence=confidence
        )
        
        return success
    
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
    
    def should_capture_vehicle(self, plate_number: str, image, ocr_confidence: float, detection_confidence: float, frame_count: int) -> tuple:
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
        
        # Also consider OCR confidence
        current_score = (ocr_confidence * 0.5) + (detection_confidence * 0.3) + (current_clarity / 1000 * 0.2)
        existing_score = (existing.get('ocr_confidence', 0) * 0.5) + (existing.get('detection_confidence', 0) * 0.3) + (existing_clarity / 1000 * 0.2)
        
        # If current is significantly better, update
        if current_score > existing_score * 1.2:  # 20% better
            return True, "better"
        
        return False, "already_captured"
    
    def capture_vehicle(self, plate_number: str, image, box: tuple, ocr_confidence: float, detection_confidence: float, frame_count: int):
        """Capture a vehicle - save snapshot and update tracking"""
        clean_plate = plate_number.upper().strip()
        
        # Calculate clarity
        clarity = self.calculate_image_clarity(image)
        
        # Save snapshot
        snapshot_path = self.save_snapshot(image, clean_plate, ocr_confidence)
        
        # Log to database
        self.log_vehicle(clean_plate, snapshot_path, ocr_confidence)
        
        # Track this vehicle
        self.captured_vehicles[clean_plate] = {
            'frame': frame_count,
            'confidence': ocr_confidence,
            'ocr_confidence': ocr_confidence,
            'detection_confidence': detection_confidence,
            'clarity': clarity,
            'snapshot_path': snapshot_path
        }
        
        return snapshot_path
    
    def update_vehicle_capture(self, plate_number: str, image, box: tuple, ocr_confidence: float, detection_confidence: float, frame_count: int):
        """Update captured vehicle with better quality image"""
        clean_plate = plate_number.upper().strip()
        
        # Calculate clarity
        clarity = self.calculate_image_clarity(image)
        
        # Save new snapshot (overwrites old one)
        snapshot_path = self.save_snapshot(image, clean_plate, ocr_confidence)
        
        # Update database with new path
        old_path = self.captured_vehicles[clean_plate].get('snapshot_path', '')
        self.db_manager.log_detection(clean_plate, "unknown", snapshot_path, ocr_confidence)
        
        # Update tracking
        self.captured_vehicles[clean_plate] = {
            'frame': frame_count,
            'confidence': ocr_confidence,
            'ocr_confidence': ocr_confidence,
            'detection_confidence': detection_confidence,
            'clarity': clarity,
            'snapshot_path': snapshot_path
        }
        
        return snapshot_path


def process_image_mode(detector: PlateDetector, image_path: str):
    """Process a single image file"""
    print(f"\n[MODE] Image Processing")
    print(f"[FILE] {image_path}")
    print("=" * 50)
    
    detections, img = detector.detect_plates(image_path, conf_threshold=0.5)
    
    print(f"\n[RESULT] Detected {len(detections)} plate(s)\n")
    
    ocr_results = []
    
    for i, det in enumerate(detections):
        print(f"Plate {i+1}:")
        print(f"  Bounding Box: {det['box']}")
        print(f"  Confidence: {det['confidence']:.2%}")
        
        result = detector.process_plate(img, det['box'])
        ocr_result = result['ocr_result']
        ocr_results.append(ocr_result)
        
        print(f"  Raw Text: {ocr_result['raw_text']}")
        print(f"  Cleaned Text: {ocr_result['text']}")
        print(f"  OCR Confidence: {ocr_result['confidence']:.2%}")
        
        if ocr_result['text']:
            print(f"\n  *** DETECTED VEHICLE NUMBER: {ocr_result['text']} ***")
            
            snapshot_path = detector.save_snapshot(img, ocr_result['text'], ocr_result['confidence'])
            detector.log_vehicle(ocr_result['text'], snapshot_path, ocr_result['confidence'])
        
        print()
    
    if detections:
        annotated = detector.draw_results(img, detections, ocr_results)
        output_path = detector.output_dir / f"result_{Path(image_path).name}"
        cv2.imwrite(str(output_path), annotated)
        print(f"[OUTPUT] Annotated image saved to: {output_path}")
    
    return len(detections) > 0


def process_video_mode(detector: PlateDetector, video_path: str):
    """Process video file"""
    print(f"\n[MODE] Video Processing")
    print(f"[FILE] {video_path}")
    print("=" * 50)
    print("[INFO] Press 'q' or Ctrl+C to stop")
    print("=" * 50 + "\n")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    detections_logged = 0
    frame_skip = 5
    
    # Reset captured vehicles and consensus for this session
    detector.captured_vehicles = {}
    detector.frame_consensus = {}
    detector.session_frame_count = 0
    
    # Track pending captures (waiting for consensus)
    pending_captures = {}  # {normalized_plate: {'image': img, 'box': box, 'ocr_confidence': conf, 'detection_confidence': det_conf, 'frame_count': fc}}
    
    window_name = "Number Plate Detection - Video Mode"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("[INFO] End of video file")
            break
        
        frame_count += 1
        detector.session_frame_count = frame_count
        
        if frame_count % frame_skip != 0:
            continue
        
        detections, img = detector.detect_plates(frame, conf_threshold=0.5)
        
        if detections:
            ocr_results = []
            
            for det in detections:
                result = detector.process_plate(img, det['box'], frame_count)
                ocr_result = result['ocr_result']
                ocr_results.append(ocr_result)
                
                if ocr_result['text'] and len(ocr_result['text']) >= 4:
                    # Normalize plate for tracking
                    normalized_plate = detector.normalize_plate(ocr_result['text'])
                    
                    # Skip if already captured
                    if normalized_plate in detector.captured_vehicles:
                        continue
                    
                    # Add to consensus
                    consensus_result = detector.add_to_consensus(normalized_plate, ocr_result['confidence'])
                    
                    # Store pending capture with best quality
                    if normalized_plate not in pending_captures or ocr_result['confidence'] > pending_captures[normalized_plate]['ocr_confidence']:
                        pending_captures[normalized_plate] = {
                            'image': img.copy(),
                            'box': det['box'],
                            'ocr_confidence': ocr_result['confidence'],
                            'detection_confidence': det['confidence'],
                            'frame_count': frame_count,
                            'plate_text': ocr_result['text']
                        }
                    
                    # Check if we have consensus - require multiple consistent readings
                    if consensus_result and consensus_result not in detector.captured_vehicles:
                        # Check if this plate has appeared multiple times
                        if len(detector.frame_consensus[consensus_result]['readings']) >= 2:
                            # Get the best pending capture
                            pending = pending_captures.get(consensus_result)
                            if pending:
                                print(f"[CAPTURE] Frame {frame_count}: {consensus_result} (conf: {pending['ocr_confidence']:.2%}, {len(detector.frame_consensus[consensus_result]['readings'])} readings)")
                                
                                # Capture the vehicle
                                snapshot_path = detector.save_snapshot(pending['image'], consensus_result, pending['ocr_confidence'])
                                detector.log_vehicle(consensus_result, snapshot_path, pending['ocr_confidence'])
                                
                                # Mark as captured
                                detector.captured_vehicles[consensus_result] = {
                                    'frame': pending['frame_count'],
                                    'confidence': pending['ocr_confidence'],
                                    'ocr_confidence': pending['ocr_confidence'],
                                    'detection_confidence': pending['detection_confidence'],
                                    'clarity': detector.calculate_image_clarity(pending['image']),
                                    'snapshot_path': snapshot_path
                                }
                                
                                detections_logged += 1
                                
                                # Remove from pending
                                if consensus_result in pending_captures:
                                    del pending_captures[consensus_result]
            
            annotated = detector.draw_results(img, detections, ocr_results)
            cv2.imshow(window_name, annotated)
        else:
            cv2.imshow(window_name, img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[INFO] User stopped video processing")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n[RESULT] Video processing complete")
    print(f"  Total frames processed: {frame_count}")
    print(f"  Unique vehicles captured: {len(detector.captured_vehicles)}")
    print(f"  Detections logged: {detections_logged}")


def process_webcam_mode(detector: PlateDetector, camera_index: int = 0):
    """Process webcam stream"""
    print(f"\n[MODE] Webcam Processing")
    print(f"[CAMERA] Camera index: {camera_index}")
    print("=" * 50)
    print("[INFO] Press 'q' or Ctrl+C to stop")
    print("=" * 50 + "\n")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_index}")
        print("[INFO] Make sure camera is connected and not in use by another application")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    frame_count = 0
    detections_logged = 0
    frame_skip = 10
    
    # Reset captured vehicles and consensus for this session
    detector.captured_vehicles = {}
    detector.frame_consensus = {}
    detector.session_frame_count = 0
    
    # Track pending captures (waiting for consensus)
    pending_captures = {}
    
    window_name = "Number Plate Detection - Webcam Mode"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    print("[INFO] Starting webcam... Press 'q' to quit\n")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("[ERROR] Failed to grab frame from camera")
            break
        
        frame_count += 1
        detector.session_frame_count = frame_count
        
        display_frame = frame.copy()
        
        if frame_count % frame_skip == 0:
            detections, img = detector.detect_plates(frame, conf_threshold=0.5)
            
            if detections:
                ocr_results = []
                
                for det in detections:
                    result = detector.process_plate(img, det['box'], frame_count)
                    ocr_result = result['ocr_result']
                    ocr_results.append(ocr_result)
                    
                    if ocr_result['text'] and len(ocr_result['text']) >= 4:
                        # Normalize plate for tracking
                        normalized_plate = detector.normalize_plate(ocr_result['text'])
                        
                        # Skip if already captured
                        if normalized_plate in detector.captured_vehicles:
                            continue
                        
                        # Add to consensus
                        consensus_result = detector.add_to_consensus(normalized_plate, ocr_result['confidence'])
                        
                        # Store pending capture with best quality
                        if normalized_plate not in pending_captures or ocr_result['confidence'] > pending_captures[normalized_plate]['ocr_confidence']:
                            pending_captures[normalized_plate] = {
                                'image': img.copy(),
                                'box': det['box'],
                                'ocr_confidence': ocr_result['confidence'],
                                'detection_confidence': det['confidence'],
                                'frame_count': frame_count,
                                'plate_text': ocr_result['text']
                            }
                        
                        # Check for consensus
                        if consensus_result and consensus_result not in detector.captured_vehicles:
                            if len(detector.frame_consensus[consensus_result]['readings']) >= 2:
                                pending = pending_captures.get(consensus_result)
                                if pending:
                                    print(f"[CAPTURE] {consensus_result} (conf: {pending['ocr_confidence']:.2%})")
                                    
                                    snapshot_path = detector.save_snapshot(pending['image'], consensus_result, pending['ocr_confidence'])
                                    detector.log_vehicle(consensus_result, snapshot_path, pending['ocr_confidence'])
                                    
                                    detector.captured_vehicles[consensus_result] = {
                                        'frame': pending['frame_count'],
                                        'confidence': pending['ocr_confidence'],
                                        'ocr_confidence': pending['ocr_confidence'],
                                        'detection_confidence': pending['detection_confidence'],
                                        'clarity': detector.calculate_image_clarity(pending['image']),
                                        'snapshot_path': snapshot_path
                                    }
                                    
                                    detections_logged += 1
                                    
                                    if consensus_result in pending_captures:
                                        del pending_captures[consensus_result]
                
                display_frame = detector.draw_results(img, detections, ocr_results)
        
        cv2.putText(display_frame, f"Frame: {frame_count} | Captured: {len(detector.captured_vehicles)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press 'q' to quit", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(window_name, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[INFO] User stopped webcam")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n[RESULT] Webcam session complete")
    print(f"  Total frames: {frame_count}")
    print(f"  Unique vehicles captured: {len(detector.captured_vehicles)}")
    print(f"  Detections logged: {detections_logged}")


def select_camera():
    """Interactive camera selection"""
    print("\n[SELECT] Testing available cameras...")
    print("=" * 50)
    
    available_cameras = []
    
    for i in range(4):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    available_cameras.append(i)
                    print(f"  Camera {i}: Available")
        except:
            pass
    
    if not available_cameras:
        print("[ERROR] No cameras found!")
        return None
    
    print("\nAvailable cameras:", available_cameras)
    
    while True:
        try:
            choice = input("\nSelect camera index (number): ").strip()
            camera_idx = int(choice)
            
            if camera_idx in available_cameras:
                print(f"[OK] Selected camera {camera_idx}")
                return camera_idx
            else:
                print(f"[ERROR] Camera {camera_idx} not available. Choose from: {available_cameras}")
        except ValueError:
            print("[ERROR] Please enter a valid number")


def auto_detect_input():
    """Auto detect input source"""
    print("\n[AUTO] Determining input source...")
    print("=" * 50)
    
    print("[STEP 1] Testing webcam...")
    test_cap = cv2.VideoCapture(0)
    if test_cap.isOpened():
        test_cap.release()
        print("[RESULT] Webcam available")
        return 'webcam', 0
    test_cap.release()
    print("[RESULT] Webcam not available")
    
    print("\n[STEP 2] Checking input folder...")
    first_file = get_first_input_file()
    if first_file:
        ext = os.path.splitext(first_file)[1].lower()
        video_exts = ['.mp4', '.avi', '.mov', '.mkv']
        if ext in video_exts:
            print(f"[RESULT] Found video: {first_file}")
            return 'video', first_file
        else:
            print(f"[RESULT] Found image: {first_file}")
            return 'image', first_file
    
    print("\n[STEP 3] Generating demo image...")
    demo_path = create_demo_image()
    print(f"[RESULT] Demo image created: {demo_path}")
    
    return 'image', demo_path


def interactive_mode():
    """Interactive mode - ask user for input selection"""
    print("\n" + "=" * 50)
    print("  SELECT INPUT SOURCE")
    print("=" * 50)
    print("  1) Webcam")
    print("  2) Video file")
    print("  3) Image file")
    print("  4) Auto demo")
    print("=" * 50)
    
    while True:
        try:
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == '1':
                camera_idx = select_camera()
                if camera_idx is not None:
                    return 'webcam', camera_idx
            elif choice == '2':
                video_path = input("Enter video file path: ").strip().strip('"')
                if os.path.exists(video_path):
                    return 'video', video_path
                else:
                    print(f"[ERROR] File not found: {video_path}")
            elif choice == '3':
                image_path = input("Enter image file path: ").strip().strip('"')
                if os.path.exists(image_path):
                    return 'image', image_path
                else:
                    print(f"[ERROR] File not found: {image_path}")
            elif choice == '4':
                print("[MODE] Auto demo mode")
                demo_path = create_demo_image()
                return 'image', demo_path
            else:
                print("[ERROR] Invalid choice. Enter 1, 2, 3, or 4")
        except KeyboardInterrupt:
            print("\n[INFO] Cancelled by user")
            sys.exit(0)
        except Exception as e:
            print(f"[ERROR] {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Indian Number Plate Detection System - Fully Automated",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Interactive mode
  python main.py --auto                   # Auto mode
  python main.py --webcam                 # Webcam with selection
  python main.py --camera 1               # Specific camera
  python main.py --image image.jpg        # Image mode
  python main.py --video video.mp4        # Video mode
        """
    )
    
    parser.add_argument('--auto', action='store_true', help='Auto detect input')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--webcam', action='store_true', help='Use webcam with selection')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--gpu', action='store_true', help='Force GPU usage')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 50)
    print("  INDIAN NUMBER PLATE DETECTION SYSTEM")
    print("  Fully Automated - Pretrained Models Only")
    print("=" * 50)
    print(f"Running from: {BASE_DIR}")
    print()
    
    # Override device if specified
    global DEVICE
    if args.gpu and torch.cuda.is_available():
        DEVICE = "cuda"
    elif args.cpu:
        DEVICE = "cpu"
    
    print("[INFO] Initializing database...")
    db_path = os.path.join(BASE_DIR, "database", "vehicles.db")
    db_manager = DatabaseManager(db_path)
    print(f"[OK] Database ready: {db_path}")
    
    detector = PlateDetector(use_gpu=(DEVICE == "cuda"))
    detector.load_model("yolov8n.pt")
    
    mode = None
    input_source = None
    
    if args.image:
        mode = 'image'
        input_source = args.image
    elif args.video:
        mode = 'video'
        input_source = args.video
    elif args.webcam:
        mode = 'webcam'
        input_source = select_camera()
        if input_source is None:
            print("[ERROR] No camera selected")
            sys.exit(1)
    elif args.camera != 0:
        mode = 'webcam'
        input_source = args.camera
    else:
        mode, input_source = interactive_mode()
    
    print(f"\n[MODE] {mode.upper()}")
    if input_source is not None:
        print(f"[INPUT] {input_source}")
    print("=" * 50 + "\n")
    
    try:
        if mode == 'image':
            if not input_source or not os.path.exists(input_source):
                print(f"[ERROR] Image file not found: {input_source}")
                sys.exit(1)
            process_image_mode(detector, input_source)
            
        elif mode == 'video':
            if not input_source or not os.path.exists(input_source):
                print(f"[ERROR] Video file not found: {input_source}")
                sys.exit(1)
            process_video_mode(detector, input_source)
            
        else:
            process_webcam_mode(detector, input_source)
    
    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("  SYSTEM SHUTDOWN")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
