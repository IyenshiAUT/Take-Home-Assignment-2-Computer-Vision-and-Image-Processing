# EC7212 – Computer Vision and Image Processing  
# Take Home Assignment 2

Reg. No: EG/2020/3975 <br/>
Name   : Iyenshi A.U.T.

This repository consists of the source codes and outputs of the Take Home Assignment 02 of EC7212 – Computer Vision and Image Processing.

---
It demonstrates two fundamental image segmentation techniques:

1.  **Otsu's Algorithm**: An adaptive, global thresholding method used to separate an image into two classes (foreground and background).
2.  **Region Growing**: An iterative segmentation approach that groups pixels into larger regions based on a predefined similarity criterion.

---

## Project Structure

The repository is organized into separate modules for each implemented algorithm, making it easy to navigate and understand.

```
.
├── 1 Otsu Algorithm/
│   ├── output/
│   │   ├── noisy_3class_image.png
│   │   ├── original_3class_image.png
│   │   └── otsu_segmented_image.png
│   └── otsu_algorithm.py
│
├── 2 Region Growing/
│   ├── output/
│   │   ├── original_image.png
│   │   ├── region_grown_mask.png
│   │   └── seeds_visual.png
│   └── region_growing.py
│
├── Image Creation/
│   ├── output/
│   │   └── original_3class_image.png
│   └── create_images.py
│
├── requirements.txt
└── README.md
```

---

## Outputs

### 1. Otsu's Algorithm


| Original Image | Noisy Image | Otsu's Segmentation |
| :---: | :---: | :---: |
| ![Original 3-Class Image](./1%20Otsu%20Algorithm/output/original_3class_image.png) | ![Noisy 3-Class Image](./1%20Otsu%20Algorithm/output/noisy_3class_image.png) | ![Otsu Segmented Image](./1%20Otsu%20Algorithm/output/otsu_segmented_image.png) |



### 2. Region Growing

| Input Image | Seed Point | Segmented Mask |
| :---: | :---: | :---: |
| ![Original Image for Region Growing](./2%20Region%20Growing/output/original_image.png) | ![Seed Point Visualization](./2%20Region%20Growing/output/seeds_visual.png) | ![Region Grown Mask](./2%20Region%20Growing/output/region_grown_mask.png) |

---

## Setup and Usage

Follow these steps to set up and run the project.

### 1. Prerequisites

- Python 3.6+
- `pip` package manager

### 2. Installation

First, clone the repository to your local machine:
```bash
git clone https://github.com/IyenshiAUT/Take-Home-Assignment-2-Computer-Vision-and-Image-Processing.git
```

Create and activate a virtual environment (recommended):
```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

Install the required dependencies from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 3. Running the Scripts

Each part of the assignment can be run independently.
**To run the Original Image Creation:**
```bash
python "Image Creation/create_image.py"
```


**To run the Otsu's Algorithm demonstration:**
```bash
python "1 Otsu Algorithm/otsu_algorithm.py"
```

**To run the Region Growing demonstration:**
```bash
python "2 Region Growing/region_growing.py"
```

The scripts will display the resulting images on your screen and save them to the corresponding `output` directory.

---

## Dependencies

This project relies on the following Python libraries:

- **OpenCV-Python**: For all core computer vision and image processing tasks.
- **NumPy**: For efficient numerical operations and array manipulation.

```text
# requirements.txt
opencv-python
numpy
```

---
## 📄 License
This project is licensed under the MIT License.
```
