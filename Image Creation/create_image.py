import cv2
import numpy as np
import os

# Create an original image which has 2 objects and a background (3 pixel values)
def create_original_image():
    """
    Creates a synthetic image with 2 objects and a background (3 pixel values).
    
    Returns:
        image (numpy.ndarray): noisy_image
    """
    
    # Apply grayscale values for the three classes
    background_val = 40
    object1_val = 160
    object2_val = 200
    
    # 1. Create a black image (300x400)
    # Create a black image
    original_image = np.full((300, 400), background_val, dtype=np.uint8)
    
    # 2. Add the two objects
    # Object 1: A rectangle
    cv2.rectangle(original_image, (50, 50), (150, 250), object1_val, -1)
    # Object 2: A circle
    cv2.circle(original_image, (300, 150), 70, object2_val, -1)
    
    return original_image

if __name__ == "__main__":
    
    # Output directory
    OUTPUT_DIR = "Image Creation/output"

    # Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Create a synthetic image with 2 objects and a background
    # and add Gaussian noise to it
    original_3class_img= create_original_image()
    
   
    # Save images to the output directory
    cv2.imwrite(os.path.join(OUTPUT_DIR, "original_3class_image.png"), original_3class_img)
