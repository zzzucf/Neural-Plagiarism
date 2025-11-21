import os
import sys
import cv2
import argparse
from imwatermark import WatermarkEncoder, WatermarkDecoder
from pathlib import Path

def load_watermark(watermark_file):
    """Loads the watermark from a binary file."""
    with open(watermark_file, 'rb') as f:
        watermark = f.read()
    print(f"Watermark loaded from: {watermark_file}")
    return watermark

def calculate_bit_accuracy(original, extracted):
    """Calculates the bit accuracy between the original and extracted watermark."""
    original_bits = ''.join(format(byte, '08b') for byte in original)
    extracted_bits = ''.join(format(byte, '08b') for byte in extracted)
    matching_bits = sum(1 for o, e in zip(original_bits, extracted_bits) if o == e)
    accuracy = matching_bits / len(original_bits) * 100
    return accuracy

def detect_watermark(image_path, watermark, method):
    """Detects the watermark from an image using the specified method."""
    bgr = cv2.imread(str(image_path))

    # Initialize the decoder
    decoder = WatermarkDecoder('bytes', len(watermark) * 8)
    if method == 'rivaGan':
        decoder.loadModel()

    # Decode the watermark using the specified method
    extracted_watermark = decoder.decode(bgr, method)

    # Calculate bit accuracy
    bit_accuracy = calculate_bit_accuracy(watermark, extracted_watermark)
    return bit_accuracy

def process_images(image_folder, watermark, method):
    """Processes all images in the image folder and detects the watermark."""
    total_accuracy = 0
    image_count = 0

    # Process each image in the folder
    for image_path in image_folder.glob('*'):
        if image_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
            bit_accuracy = detect_watermark(image_path, watermark, method)
            total_accuracy += bit_accuracy
            image_count += 1

    # Calculate the average bit accuracy
    if image_count > 0:
        average_accuracy = total_accuracy / image_count
        print(f"\nAverage bit accuracy across all images: {average_accuracy:.2f}%")
    else:
        print("No valid images found in the specified folder.")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Detect watermarks in images using a specified method.")
    parser.add_argument('--watermark_file', type=str, help="Path to the watermark binary file.")
    parser.add_argument('--image_folder', type=str, help="Path to the folder containing images to check for the watermark.")
    parser.add_argument('--method', type=str, help="Watermark detection method to use (e.g., 'dwtDct', 'dct').")
    parser.add_argument('--bits', default=32, type=int, help="Watermark detection method to use (e.g., 'dwtDct', 'dct').")

    # Parse the arguments
    args = parser.parse_args()

    watermark_file = Path(args.watermark_file)
    image_folder = Path(args.image_folder)
    method = args.method
    bits = args.bits

    # Check if the watermark file exists
    if not watermark_file.exists() or not watermark_file.is_file():
        print(f"Error: The watermark file '{watermark_file}' does not exist.")
        sys.exit(1)

    # Check if the image folder exists
    if not image_folder.exists() or not image_folder.is_dir():
        print(f"Error: The image folder '{image_folder}' does not exist or is not a directory.")
        sys.exit(1)

    # Load the watermark
    watermark = load_watermark(watermark_file)

    # Process images
    process_images(image_folder, watermark, method)

if __name__ == "__main__":
    main()
