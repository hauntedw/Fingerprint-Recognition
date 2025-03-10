# Fingerprint Recognition System

A Python-based fingerprint recognition system that implements a complete pipeline for fingerprint processing, feature extraction, and matching.

## Overview

This script provides a complete fingerprint biometric system with the following capabilities:
- Image acquisition from files
- Fingerprint enhancement and preprocessing
- Minutiae detection (ridge endings and bifurcations)
- Fingerprint matching using Hough transform

## Requirements

The script requires the following Python packages:
- Python 3.x
- NumPy
- OpenCV (cv2)
- SciPy
- scikit-image

You can install the dependencies using pip:
```
pip install numpy opencv-python scipy scikit-image
```

## Dataset Structure

The dataset folder should contain fingerprint images in BMP or other common image formats. he script is currently configured to look for images in the `dataset/` directory.

## Usage

1. Place your fingerprint images in a `dataset` folder next to the script or use the images provided.
2. Edit the image paths at the bottom of the script to point to your fingerprint images:
   ```python
   fingeprint_filepath_1 = 'dataset/10__M_Left_index_finger_Zcut.BMP'
   fingeprint_filepath_2 = 'dataset/10__M_Left_index_finger_Zcut.BMP'
   ```
3. Run the script directly:
   ```
   python fingerprint_recognition.py
   ```

## Visualization Options

The script contains several visualization flags that can be enabled to see the intermediate processing steps:

- In the `acquire_from_file()` function, set `view=True` to display the original image
- In the `enhance()` function, set `view=True` to visualize each enhancement step
- In the `describe()` function, set `view=True` to visualize detected minutiae
- In the `match()` function, set `view=True` to visualize the matching results (enabled by default)

## Main Components

### Section A: Acquire
Functions for loading fingerprint images from files.

### Section B: Enhance
Implements preprocessing techniques to improve fingerprint image quality:
- Image resizing and normalization
- Segmentation to isolate the fingerprint from the background
- Ridge orientation computation
- Gabor filtering to enhance ridge structure
- Skeletonization to thin the ridges

### Section C: Describe
Extracts minutiae features from the enhanced fingerprint:
- Detection of ridge endings and bifurcations
- False minutiae elimination
- Minutiae angle computation

### Section D: Match
Implements the fingerprint matching algorithm:
- Hough transform for alignment
- Minutiae pairing
- Match score calculation

## Customization

Key parameters that can be adjusted include:
- `FINGERPRINT_HEIGHT`: Target height for resizing the fingerprint (default: 352)
- `FINGERPRINT_BLOCK`: Block size for processing (default: 16)
- `FINGERPRINT_MASK_TRSH`: Threshold for fingerprint segmentation (default: 0.25)
- `DIST_TRSH`: Distance threshold for matching minutiae (default: 10)
- `ANGLE_TRSH`: Angle threshold for matching minutiae (default: π/8)

## Output

The script displays:
- Visual representation of the matching process (if visualization is enabled)
- Number of minutiae in each fingerprint
- Match score in the console

## Example

```python
fingeprint_filepath_1 = 'dataset/5__M_Left_index_finger_Zcut.BMP'
fingeprint_filepath_2 = 'dataset/5__M_Left_index_finger_Zcut.BMP'

fingerprint_1 = acquire_from_file(fingeprint_filepath_1, view=True)
fingerprint_2 = acquire_from_file(fingeprint_filepath_2, view=True)

pp_fingerprint_1, en_fingerprint_1, mask_1 = enhance(fingerprint_1, dark_ridges=False, view=True)
pp_fingerprint_2, en_fingerprint_2, mask_2 = enhance(fingerprint_2, dark_ridges=False, view=True)

ridge_endings_1, bifurcations_1 = describe(en_fingerprint_1, mask_1, view=True)
ridge_endings_2, bifurcations_2 = describe(en_fingerprint_2, mask_2, view=True)

match = match(en_fingerprint_1, ridge_endings_1, bifurcations_1, en_fingerprint_2, ridge_endings_2, bifurcations_2, view=True)
```

## License

This code is provided for educational and research purposes.