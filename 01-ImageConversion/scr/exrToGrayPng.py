
import cv2
import OpenEXR
import Imath
import numpy as np
import os
import argparse


# Function to read an EXR file and return it as a numpy array
def read_exr(exr_file):
    exr_file = OpenEXR.InputFile(exr_file)
    header = exr_file.header()

    # Get the image size
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Extract the data for each color channel
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    rgb_str = [exr_file.channel(c, pt) for c in ("R", "G", "B")]

    # Convert data to numpy array and reshape to image dimensions
    rgb = [np.frombuffer(c, dtype=np.float32) for c in rgb_str]
    rgb = [c.reshape(height, width) for c in rgb]

    # Stack channels to create a 3D array
    img = np.stack(rgb, axis=-1)

    return img


# Function to apply tone mapping and convert to grayscale
def tone_map_and_convert_to_grayscale(img):
    # Apply simple Reinhard tone mapping to handle HDR to LDR conversion
    tonemap = cv2.createTonemapReinhard(gamma=0.8)  # Gamma correction
    ldr_img = tonemap.process(img)  # Apply tone mapping to HDR image

    # Normalize to [0, 255] and convert to uint8 for saving as PNG
    ldr_img = np.clip(ldr_img * 255, 0, 255).astype(np.uint8)

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(ldr_img, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_img = clahe.apply(gray_img)

    return gray_img


# Function to convert the EXR image to grayscale and save as PNG
def convert_exr_to_png(exr_path, output_path):
    # Read the EXR file
    img = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
    clipped_img = np.clip(img, 0.0, 0.85)
    # Apply tone mapping and convert to grayscale
    gray_img = tone_map_and_convert_to_grayscale(clipped_img)

    # Save the image as PNG
    cv2.imwrite(output_path, gray_img)
    print(f"Saved {output_path}")


# Convert all EXR files in the input folder and save as PNG in the output folder
def convert_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".exr"):
            exr_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + ".png"
            output_path = os.path.join(output_dir, output_filename)
            convert_exr_to_png(exr_path, output_path)


# Main function to handle command-line arguments
def main():
    parser = argparse.ArgumentParser(description="Convert EXR images to grayscale PNG")
    parser.add_argument('--file', type=str, required=True, help="Path to the folder containing on_centre or off_centre data")

    args = parser.parse_args()
    input_folder = os.path.join(args.file, r"im\rgb")
    output_folder = os.path.join(args.file, r"im\gray")

    convert_directory(input_folder, output_folder)


if __name__ == "__main__":
    main()

