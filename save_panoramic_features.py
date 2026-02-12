"""
Script to save a panoramic image from 360Loc dataset along with its
processed versions: grayscale, magnitude, angle, and hue.

Usage:
    python save_panoramic_features.py --input <path_to_image> --output <output_folder>

Example:
    python save_panoramic_features.py --input /path/to/image.jpg --output ./output_features/
"""

import os
import argparse
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage


def load_image(image_path):
    """Load an image from file."""
    if image_path.endswith((".jpeg", ".jpg", ".png")):
        return np.array(Image.open(image_path).convert('RGB')).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported image format: {image_path}")


def generate_grayscale(image):
    """Convert RGB image to grayscale.
    
    Args:
        image: RGB image as numpy array
        
    Returns:
        Grayscale image normalized to [0, 1]
    """
    # Convert numpy array to PIL Image, then to grayscale
    pil_img = Image.fromarray(image)
    gray_img = pil_img.convert('L')
    gray = np.array(gray_img).astype(np.float32) / 255.0
    return gray


def generate_hue(image):
    """Extract hue channel from RGB image.
    
    Args:
        image: RGB image as numpy array
        
    Returns:
        Hue channel normalized to [0, 1]
    """
    # Convert numpy array to PIL Image, then to HSV
    pil_img = Image.fromarray(image)
    hsv_img = pil_img.convert('HSV')
    hsv_array = np.array(hsv_img)
    hue = hsv_array[:, :, 0].astype(np.float32) / 255.0  # PIL hue is in [0, 255]
    return hue


def generate_magnitude_angle(image):
    """Compute gradient magnitude and angle from RGB image.
    
    Args:
        image: RGB image as numpy array
        
    Returns:
        magnitude: Gradient magnitude normalized to [0, 1]
        angle: Gradient angle normalized to [0, 1]
    """
    # Convert to grayscale using PIL
    pil_img = Image.fromarray(image)
    gray_img = pil_img.convert('L')
    
    # Apply Gaussian blur using PIL
    blurred_img = gray_img.filter(ImageFilter.GaussianBlur(radius=2))
    gray = np.array(blurred_img).astype(np.float64)
    
    # Calculate gradients using Sobel operators with scipy
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    grad_x = ndimage.convolve(gray, sobel_x)
    grad_y = ndimage.convolve(gray, sobel_y)
    
    # Compute magnitude and angle
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x)
    
    # Normalize magnitude to [0, 1]
    max_mag = np.max(magnitude)
    if max_mag > 0:
        magnitude = (magnitude / max_mag).astype(np.float32)
    else:
        magnitude = magnitude.astype(np.float32)
    
    # Normalize angle from [-pi, pi] to [0, 1]
    angle = ((angle + np.pi) / (2 * np.pi)).astype(np.float32)
    
    return magnitude, angle


def convert_features_from_image_path(image_path, output_folder):
    """Convert feature .npy files to .jpg based on image path.
    
    Given an image path like:
    /path/to/dataset/image_resized/0295.jpg
    
    This function will find and convert:
    /path/to/dataset/GRAYSCALE/0295.npy -> output_folder/0295_GRAYSCALE.jpg
    /path/to/dataset/HUE/0295.npy -> output_folder/0295_HUE.jpg
    /path/to/dataset/MAGNITUDE/0295.npy -> output_folder/0295_MAGNITUDE.jpg
    /path/to/dataset/ANGLE/0295.npy -> output_folder/0295_ANGLE.jpg
    
    Args:
        image_path: Path to the original image
        output_folder: Directory to save converted .jpg files
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get the directory and filename from image path
    image_dir = os.path.dirname(image_path)
    image_filename = os.path.basename(image_path)
    base_name = os.path.splitext(image_filename)[0]
    image_ext = os.path.splitext(image_filename)[1]
    
    print(f"Processing features for image: {image_path}")
    print(f"Base name: {base_name}")
    
    # Features to process
    features = ['GRAYSCALE', 'HUE', 'MAGNITUDE', 'ANGLE']
    
    for feature in features:
        # Replace 'image_resized' with feature name and extension with .npy
        npy_path = image_path.replace('image_resized', feature).replace(image_ext, '.npy')
        
        if not os.path.exists(npy_path):
            print(f"Warning: {feature} file not found at {npy_path}")
            continue
        
        # Load the .npy file
        print(f"Loading: {npy_path}")
        data = np.load(npy_path)
        
        # Check the data range
        data_min = np.min(data)
        data_max = np.max(data)
        print(f"  Data range: [{data_min:.4f}, {data_max:.4f}], dtype: {data.dtype}")
        
        # Normalize to [0, 255] range
        print(f"  Normalizing from [{data_min:.4f}, {data_max:.4f}] to [0, 255]")
        if data_max > data_min:
            data_normalized = (data - data_min) / (data_max - data_min)
            data_uint8 = (data_normalized * 255).astype(np.uint8)
        else:
            # All values are the same, just use zeros
            data_uint8 = np.zeros_like(data, dtype=np.uint8)
        
        # Save as JPG
        jpg_filename = f"{base_name}_{feature}.jpg"
        jpg_path = os.path.join(output_folder, jpg_filename)
        Image.fromarray(data_uint8).save(jpg_path)
        print(f"Saved: {jpg_path}")
    
    print(f"\nConversion complete! JPG files saved in: {output_folder}")


def save_features(input_path, output_folder):
    """Process and save all feature representations of an image.
    
    Args:
        input_path: Path to input panoramic image
        output_folder: Directory to save output images
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get base filename (without extension)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # Load original image
    print(f"Loading image: {input_path}")
    image = load_image(input_path)
    
    # Save original RGB image
    rgb_path = os.path.join(output_folder, f"{base_name}_RGB.jpg")
    Image.fromarray(image).save(rgb_path)
    print(f"Saved RGB: {rgb_path}")
    
    # Generate and save grayscale
    grayscale = generate_grayscale(image)
    grayscale_path = os.path.join(output_folder, f"{base_name}_GRAYSCALE.npy")
    np.save(grayscale_path, grayscale)
    # Also save as image for visualization
    grayscale_img_path = os.path.join(output_folder, f"{base_name}_GRAYSCALE.jpg")
    Image.fromarray((grayscale * 255).astype(np.uint8)).save(grayscale_img_path)
    print(f"Saved GRAYSCALE: {grayscale_path} and {grayscale_img_path}")
    
    # Generate and save hue
    hue = generate_hue(image)
    hue_path = os.path.join(output_folder, f"{base_name}_HUE.npy")
    np.save(hue_path, hue)
    # Also save as image for visualization
    hue_img_path = os.path.join(output_folder, f"{base_name}_HUE.jpg")
    Image.fromarray((hue * 255).astype(np.uint8)).save(hue_img_path)
    print(f"Saved HUE: {hue_path} and {hue_img_path}")
    
    # Generate and save magnitude and angle
    magnitude, angle = generate_magnitude_angle(image)
    
    magnitude_path = os.path.join(output_folder, f"{base_name}_MAGNITUDE.npy")
    np.save(magnitude_path, magnitude)
    # Also save as image for visualization
    magnitude_img_path = os.path.join(output_folder, f"{base_name}_MAGNITUDE.jpg")
    Image.fromarray((magnitude * 255).astype(np.uint8)).save(magnitude_img_path)
    print(f"Saved MAGNITUDE: {magnitude_path} and {magnitude_img_path}")
    
    angle_path = os.path.join(output_folder, f"{base_name}_ANGLE.npy")
    np.save(angle_path, angle)
    # Also save as image for visualization
    angle_img_path = os.path.join(output_folder, f"{base_name}_ANGLE.jpg")
    Image.fromarray((angle * 255).astype(np.uint8)).save(angle_img_path)
    print(f"Saved ANGLE: {angle_path} and {angle_img_path}")
    
    print(f"\nAll features saved successfully to: {output_folder}")
    print(f"Files created:")
    print(f"  - {base_name}_RGB.jpg (original panoramic image)")
    print(f"  - {base_name}_GRAYSCALE.npy/.jpg (grayscale representation)")
    print(f"  - {base_name}_HUE.npy/.jpg (hue channel)")
    print(f"  - {base_name}_MAGNITUDE.npy/.jpg (gradient magnitude)")
    print(f"  - {base_name}_ANGLE.npy/.jpg (gradient angle)")


def main():
    parser = argparse.ArgumentParser(
        description="Save panoramic image and its feature representations (grayscale, magnitude, angle, hue)"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to input panoramic image (jpg, jpeg, or png)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output folder where features will be saved"
    )
    parser.add_argument(
        "--convert-from-image",
        action="store_true",
        help="Convert feature .npy files to .jpg based on image path (use with --input and --output)"
    )
    
    args = parser.parse_args()
    
    if args.convert_from_image:
        # Convert features based on image path
        if not args.input or not args.output:
            parser.error("--input and --output are required when using --convert-from-image")
        
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input file not found: {args.input}")
        
        convert_features_from_image_path(args.input, args.output)
    else:
        # Original feature generation mode
        if not args.input or not args.output:
            parser.error("--input and --output are required")
        
        # Validate input file exists
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input file not found: {args.input}")
        
        # Process and save features
        save_features(args.input, args.output)


if __name__ == "__main__":
    main()
