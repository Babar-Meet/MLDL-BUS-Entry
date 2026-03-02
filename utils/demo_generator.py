#!/usr/bin/env python3
"""
Auto demo generator - creates synthetic demo image when no input is available.
Generates an image with a vehicle-like rectangle and a license plate.
"""

import os
import sys
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

# Get the base directory (project root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level from utils/ to project root
BASE_DIR = os.path.dirname(BASE_DIR)


def create_demo_image(plate_text="GJ05AB1234"):
    """
    Create a synthetic demo image with vehicle and license plate.
    
    Args:
        plate_text: Text to display on the license plate
        
    Returns:
        Path to the generated demo image
    """
    
    # Create input folder path
    input_folder = os.path.join(BASE_DIR, "input")
    
    # Ensure input folder exists
    os.makedirs(input_folder, exist_ok=True)
    
    # Output path
    demo_path = os.path.join(input_folder, "demo.jpg")
    
    # Image dimensions
    width, height = 1280, 720
    
    # Create white background
    img = Image.new('RGB', (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Draw road (gray rectangle at bottom)
    road_y = height - 150
    draw.rectangle([0, road_y, width, height], fill=(80, 80, 80))
    
    # Draw road markings (dashed white lines)
    for i in range(0, width, 100):
        draw.rectangle([i + 20, road_y + 60, i + 60, road_y + 70], fill=(255, 255, 255))
    
    # Draw vehicle (large rectangle - car body)
    car_x1, car_y1 = 300, 200
    car_x2, car_y2 = 900, 500
    
    # Car body
    draw.rectangle([car_x1, car_y1, car_x2, car_y2], fill=(200, 50, 50), outline=(0, 0, 0), width=3)
    
    # Car roof
    roof_x1, roof_y1 = 380, 120
    roof_x2, roof_y2 = 820, 200
    draw.rectangle([roof_x1, roof_y1, roof_x2, roof_y2], fill=(180, 40, 40), outline=(0, 0, 0), width=3)
    
    # Windows (darker rectangles)
    draw.rectangle([roof_x1 + 10, roof_y1 + 10, roof_x1 + 100, roof_y2 - 10], fill=(50, 50, 100))
    draw.rectangle([roof_x1 + 110, roof_y1 + 10, roof_x2 - 110, roof_y2 - 10], fill=(50, 50, 100))
    draw.rectangle([roof_x2 - 100, roof_y1 + 10, roof_x2 - 10, roof_y2 - 10], fill=(50, 50, 100))
    
    # Wheels (black circles - approximated as rectangles)
    wheel_color = (30, 30, 30)
    draw.ellipse([car_x1 + 50, car_y2 - 20, car_x1 + 120, car_y2 + 30], fill=wheel_color)
    draw.ellipse([car_x2 - 120, car_y2 - 20, car_x2 - 50, car_y2 + 30], fill=wheel_color)
    
    # Headlights
    draw.ellipse([car_x2 - 30, car_y1 + 100, car_x2 - 5, car_y1 + 140], fill=(255, 255, 200))
    draw.ellipse([car_x1 + 5, car_y1 + 100, car_x1 + 30, car_y1 + 140], fill=(255, 255, 200))
    
    # Draw license plate (small rectangle at back of car)
    plate_x1, plate_y1 = car_x1 + 10, car_y1 + 250
    plate_x2, plate_y2 = car_x1 + 150, car_y1 + 320
    
    # Plate background (white)
    draw.rectangle([plate_x1, plate_y1, plate_x2, plate_y2], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    
    # Plate border (black)
    draw.rectangle([plate_x1 + 5, plate_y1 + 5, plate_x2 - 5, plate_y2 - 5], outline=(0, 0, 0), width=1)
    
    # Try to use a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # Draw plate text
    text_bbox = draw.textbbox((plate_x1, plate_y1), plate_text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    text_x = plate_x1 + (plate_x2 - plate_x1 - text_width) // 2
    text_y = plate_y1 + (plate_y2 - plate_y1 - text_height) // 2
    
    if font:
        draw.text((text_x, text_y), plate_text, fill=(0, 0, 0), font=font)
    else:
        draw.text((text_x, text_y), plate_text, fill=(0, 0, 0))
    
    # Add some random cars in background for realism
    for i in range(2):
        bg_car_x = 100 + i * 400
        bg_car_y = road_y - 100
        bg_car_w, bg_car_h = 200, 120
        draw.rectangle([bg_car_x, bg_car_y, bg_car_x + bg_car_w, bg_car_y + bg_car_h], 
                     fill=(100, 100, 150), outline=(0, 0, 0), width=1)
    
    # Add text label
    try:
        label_font = ImageFont.truetype("arial.ttf", 20)
    except:
        label_font = None
    
    label = "Demo: Synthetic Vehicle Image"
    if label_font:
        draw.text((20, 20), label, fill=(0, 0, 0), font=label_font)
    else:
        draw.text((20, 20), label, fill=(0, 0, 0))
    
    # Save the image
    img.save(demo_path, 'JPEG', quality=95)
    
    print(f"[AUTO] Generated demo image: {demo_path}")
    
    return demo_path


def check_input_folder():
    """
    Check if input folder has any valid files.
    Returns True if files exist, False otherwise.
    """
    input_folder = os.path.join(BASE_DIR, "input")
    
    if not os.path.exists(input_folder):
        return False
    
    # Check for valid image/video files
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.mp4', '.avi', '.mov', '.mkv']
    
    for file in os.listdir(input_folder):
        ext = os.path.splitext(file)[1].lower()
        if ext in valid_extensions:
            return True
    
    return False


def get_first_input_file():
    """
    Get the first valid input file from input folder.
    Returns full path or None if no file found.
    """
    input_folder = os.path.join(BASE_DIR, "input")
    
    if not os.path.exists(input_folder):
        return None
    
    # Check for valid image/video files
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.mp4', '.avi', '.mov', '.mkv']
    valid_extensions_img = ['.jpg', '.jpeg', '.png', '.bmp']
    valid_extensions_vid = ['.mp4', '.avi', '.mov', '.mkv']
    
    for file in os.listdir(input_folder):
        ext = os.path.splitext(file)[1].lower()
        if ext in valid_extensions:
            file_path = os.path.join(input_folder, file)
            # Verify file exists and is readable
            if os.path.isfile(file_path):
                # For images, verify it's a valid image
                if ext in valid_extensions_img:
                    try:
                        img = Image.open(file_path)
                        img.verify()
                        return file_path
                    except:
                        continue
                # For videos, just return the path
                elif ext in valid_extensions_vid:
                    return file_path
    
    return None


if __name__ == "__main__":
    print("=" * 50)
    print("  DEMO IMAGE GENERATOR")
    print("=" * 50)
    print(f"Running from: {BASE_DIR}")
    print()
    
    # Check if input folder has files
    has_input = check_input_folder()
    
    if has_input:
        print("[INFO] Input folder already has files")
        first_file = get_first_input_file()
        if first_file:
            print(f"[INFO] First input file: {first_file}")
    else:
        print("[INFO] Input folder empty - generating demo image")
        demo_path = create_demo_image()
        print(f"[SUCCESS] Demo image created: {demo_path}")
    
    print()
