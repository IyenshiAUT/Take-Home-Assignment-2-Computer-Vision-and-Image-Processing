import cv2
import numpy as np
import os

### Task 1: Otsu's Algorithm for Thresholding ###

# Create a noise image to test the algorithm
def create_noisy_test_image(original_image):
    """
    Adds Gaussian noise to an provided image.
    
    Returns:
        image (numpy.ndarray): noisy_image
    """
    
    # 1. Add Gaussian noise.
    mean = 0    # mean value
    sigma = 25  # Standard deviation - control the amount of noise, high sigma => difficult segmentation
    # Define the noise distribution
    noise = np.random.normal(mean, sigma, original_image.shape)
    # Create noisy image
    noisy_image = original_image + noise
    # Clip the values to be in the valid 0-255 range and convert to uint8
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image


# Define Otsu's Algorithm
def implement_otsu_algorithm(image):
    """
    Implements Otsu's algorithm from scratch to find the optimal global threshold.

    Args:
        image (numpy.ndarray): The input grayscale image.

    Returns:
        int: The optimal threshold value found by the algorithm.
    """

    # 1. Calculate the normalized histogram of the image
    pixel_counts = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
    # Normalize the histogram to get pixel probabilities
    total_pixels = image.shape[0] * image.shape[1]
    pixel_probabilities = pixel_counts / total_pixels

    # Initialize the maximum variance and the optimal threshold
    max_variance = 0
    optimal_threshold = 0

    # 2. Iterate through all possible thresholds (t) from 1 to 255
    for t in range(1, 256):
        # Split the pixels into two classes based on the current threshold 't'
        
        # Background Class (t <= pixel intensity) => Probability of a pixel being in the background class
        w_b = np.sum(pixel_probabilities[:t])
        if w_b == 0: continue
        # Mean intensity of the background class
        mean_b = np.sum(np.arange(t) * pixel_probabilities[:t]) / w_b

        # Foreground Class (t > pixel intensity) => Probability of a pixel being in the foreground class
        w_f = np.sum(pixel_probabilities[t:])
        if w_f == 0: continue
        # Mean intensity of the foreground class
        mean_f = np.sum(np.arange(t, 256) * pixel_probabilities[t:]) / w_f
        
        # 3. Calculate the Between-Class Variance => measure of separation between the two classes.
        between_class_variance = w_b * w_f * (mean_b - mean_f)**2
        
        # 4. Check if this threshold gives a better separation
        if between_class_variance > max_variance:
            # Update the maximum variance and optimal threshold
            max_variance = between_class_variance
            optimal_threshold = t
            
    return optimal_threshold

if __name__ == "__main__":
    
    # Input image path 
    INPUT_IMAGE_PATH = "Image Creation/output/original_3class_image.png"
    
    # Output directory
    OUTPUT_DIR = "1 Otsu Algorithm/output"

    # Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Display error message if image does not exist
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"Error: Input image '{INPUT_IMAGE_PATH}' does not exist.")
        exit(1)
        
    # Load original image
    original_3class_img = cv2.imread(INPUT_IMAGE_PATH)
        
    # Add Gaussian noise to the image
    noisy_3class_img = create_noisy_test_image(original_3class_img)
    
    # Find the optimal threshold using our implementation
    otsu_threshold = implement_otsu_algorithm(noisy_3class_img)
    # Print the optimal threshold found by Otsu's algorithm
    print(f"Otsu's algorithm found an optimal threshold at: {otsu_threshold}")
    
    # Apply the found threshold to segment the image
    _, otsu_segmented_img = cv2.threshold(noisy_3class_img, otsu_threshold, 255, cv2.THRESH_BINARY)
    
    
    # Save images to the output directory
    cv2.imwrite(os.path.join(OUTPUT_DIR, "original_3class_image.png"), original_3class_img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "noisy_3class_image.png"), noisy_3class_img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "otsu_segmented_image.png"), otsu_segmented_img)
    
    # Display the original image, noisy image, and segmented image
    cv2.imshow("Original Image (3 Classes)", original_3class_img)
    cv2.imshow("Image with Gaussian Noise", noisy_3class_img)
    cv2.imshow("Segmented with Otsu's Method", otsu_segmented_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()