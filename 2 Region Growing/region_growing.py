import cv2
import numpy as np
import os
import collections


### Task 2: Region Growing for Segmentation ###

# Define region growing function
def region_growing(image, seeds, similarity_threshold):
    """
    Implements a region growing algorithm for image segmentation.

    Args:
        image (numpy.ndarray): The input grayscale image.
        seeds (list of tuples): A list of starting seed points, e.g., [(y1, x1), (y2, x2)].
        similarity_threshold (int): The maximum intensity difference a pixel can have
                                    from its neighbor to be included in the region.

    Returns:
        numpy.ndarray: A binary mask of the segmented region.
    """

    # Height and width
    h, w = image.shape
    output_mask = np.zeros((h, w), dtype=np.uint8)
    
    # A queue to hold the pixels to be checked
    pixel_queue = collections.deque()
    
    # Add initial seeds to the queue and mark them on the output mask
    for seed in seeds:
        y, x = seed
        pixel_queue.append((y, x))
        output_mask[y, x] = 255

    # Define the 8 directions for neighbors (N, NE, E, SE, S, SW, W, NW)
    directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

    while pixel_queue:
        # Get the current pixel from the queue
        y_curr, x_curr = pixel_queue.popleft()
        
        # Check all 8 neighbors
        for dy, dx in directions:
            y_next, x_next = y_curr + dy, x_curr + dx
            
            # Check if the neighbor is within the image bounds
            if 0 <= y_next < h and 0 <= x_next < w:
                # Check if we have already visited this neighbor
                if output_mask[y_next, x_next] == 0:
                    # Check for intensity similarity
                    intensity_diff = abs(int(image[y_next, x_next]) - int(image[y_curr, x_curr]))
                    
                    if intensity_diff <= similarity_threshold:
                        # This neighbor is part of the region
                        output_mask[y_next, x_next] = 255
                        pixel_queue.append((y_next, x_next))

    return output_mask

if __name__ == "__main__":
    
    # Input image path 
    INPUT_IMAGE_PATH = "Image Creation/output/original_3class_image.png"
    
    # Output directory
    OUTPUT_DIR = "2 Region Growing/output"

    # Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Display error message if image does not exist
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"Error: Input image '{INPUT_IMAGE_PATH}' does not exist.")
        exit(1)
    
    # Load original image
    original_3class_img = cv2.imread(INPUT_IMAGE_PATH, cv2.COLOR_BGR2GRAY)
    
    # Make a copy of original image for region growing
    rg_input_image = original_3class_img.copy()

    # Define seed points inside the rectangle
    seed_points = [(150, 100)] 
    
    # Create a visual representation of the seed points on the image
    seeds_visual = cv2.cvtColor(rg_input_image, cv2.COLOR_GRAY2BGR)
    for y, x in seed_points:
        cv2.circle(seeds_visual, (x, y), 5, (0, 0, 255), -1) # Draw a red dot for the seed

    # Set the similarity threshold
    tolerance = 10
    
    # Run the region growing algorithm
    region_grown_mask = region_growing(rg_input_image, seed_points, tolerance)
    
    # Print the results
    print(f"Region growing started from seed(s) at {seed_points} with a tolerance of {tolerance}.")
    
    # Save the output mask to the output directory
    cv2.imwrite(os.path.join(OUTPUT_DIR, "original_image.png"), rg_input_image)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "seeds_visual.png"), seeds_visual)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "region_grown_mask.png"), region_grown_mask)
    
    # Display original image, seed point located image, segmented image
    cv2.imshow("Input for Region Growing", rg_input_image)
    cv2.imshow("Seed Point Location", seeds_visual)
    cv2.imshow("Segmented Region Mask", region_grown_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()