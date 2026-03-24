# Automatic Document Scanner
#### By Phillip Roos

## Description
An automatic document scanning pipeline that detects an 8.5 x 11 inch piece of paper in a photograph and rectifies it into a clean, frontal view. No manual interaction is required — the script processes all images in a given folder and outputs rectified results.

## Usage

```bash
py rectify.py                  # defaults to synthetic_data folder
py rectify.py <folder_name>    # specify a custom input folder
```

Output is saved to `output_samples/`. Each input image `input (N).jpg` produces `output (N).jpg`. 

#### WARNING: 
I put all the data sample outputs in the output_samples folder as well as debugging outputs which show how each image looks / changes during every step of the pipeline to show how robust this scanner is

---

## Pipeline Overview

The pipeline performs the following sequence of operations on each input image:

1. **Grayscale Conversion**
2. **Gaussian Blur**
3. **Binary Threshold (Binarization)**
4. **Morphological Close / Open for Text Removal and Noise Cleanup**
5. **Largest Blob Isolation**
6. **Contour Finding & Corner Localization**
7. **Homography Warp (Rectification)**

---

### Step 0 — Original Image

The raw input photograph of a document on a background surface.

![Original](output_samples/debug_input%20(1)/01_00_original.png)

---

### Step 1 — Grayscale

Convert the BGR image to a single-channel grayscale image using `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`.

![Grayscale](output_samples/debug_input%20(1)/02_01_gray.png)

---

### Step 2 — Gaussian Blur

I then apply `cv2.GaussianBlur` to smooth out fine texture and noise before thresholding. Enhances edges of paper compared to background

**Method:** `cv2.GaussianBlur(gray, GAUSS_KSIZE, 0)`

**Key Parameter:**
- `GAUSS_KSIZE = (11, 11)` — An 11x11 kernel provides enough smoothing to suppress background texture (grass, wood grain, etc.) without blurring the paper's edges into the background. I started around (5,5) and slowly increased the amount as lower values left too much texture noise. Additionally, larger kernels risked merging the paper edge with the background.

![Blurred](output_samples/debug_input%20(1)/03_02_blurred.png)

---

### Step 3 — Binary Threshold (Binarization)

Next I apply a constant binary threshold to separate the bright paper from the darker background.

**Method:** `cv2.threshold(blurred, BINARY_THRESH, 255, cv2.THRESH_BINARY)`

**Key Parameter:**
- `BINARY_THRESH = 160` — Chosen because the paper is consistently brighter than typical backgrounds like the wood, carpet, or grass. The binary threshold was used to determine what was the background and what was the paper. A value of 127 (midpoint) included too much background; I steadily increased the threshold to 160 until it worked across the 72 sample images with varied backgrounds. 

**Why I chose this fixed threshold over Otsu?** A simple fixed threshold proved more robust across the full sample set. I actually originally tried using Otsu's method but sometimes it picked a threshold that was too low for images with super bright backgrounds and adaptive thresholding introduced too much internal detail highling text and lines that complicated contour detection. The fixed threshold thus worked the best and handled the full range of samples reliably.

![Binary](output_samples/debug_input%20(1)/04_03_binary.png)

---

### Steps 4 & 5 — Morphological Cleanup (Close + Open)

Two morphological operations are applied back-to-back to clean up the binary mask:

1. **Close** — fills the black holes inside the white paper region caused by text and lines. Turns the paper into a white mesh
2. **Open** — removes small bright specks in the background that survived the binary threshold and smooths the corners from any stray pieces caused by grass stems, reflections, etc.

**Methods:**
I used mainly `cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=)` and `cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=)` 

| After Close | After Open |
|---|---|
| ![Closed](output_samples/debug_input%20(1)/05_04_closed.png) | ![Opened](output_samples/debug_input%20(1)/06_05_opened.png) |

---

### Step 5 — Largest Blob Isolation

I used connected component analysis (`cv2.connectedComponentsWithStats`) to identify all separate white regions and keep only the largest one. This guarantees that only the paper remains and discards any residual background noise. A little redudant but necessary as a few images failed without it. 

**Method:** `cv2.connectedComponentsWithStats(binary, connectivity=)`

![Blob](output_samples/debug_input%20(1)/07_06_blob.png)

---

### Step 6 — Contour Finding & Corner Localization

Find the external contour of the isolated paper blob and approximate it to a 4-sided polygon (quadrilateral) representing the document's four corners.

**Contour Finding Method:** `cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`
- `RETR_EXTERNAL` retrieves only the outermost contour, ignoring any internal holes.

**Corner Localization Method:** `cv2.approxPolyDP(contour, epsilon * perimeter, True)`
- Had to do a little research on the corner detection but I found that an algorithm that tries multiple epsilon values such as mine which progressively choose: `[0.02, 0.04, 0.06, 0.08, 0.10]` had a higher success rate than those with constant values. The algorithm works by starting at a tight 2% of perimeter and relaxes up to 10%, accepting the first approximation that yields exactly 4 vertices.
- **Fallback:** If direct approximation fails, the contour is first simplified via `cv2.convexHull` and then re-approximated with the same epsilon progression.

**Corner ordering:** The 4 corners are sorted into a consistent order (top-left, top-right, bottom-right, bottom-left) using the sum and difference of coordinates — required for the homography to produce a correctly oriented output.

![Corner Selection](output_samples/debug_input%20(1)/08_07_corner_selection.png)

---

### Step 7 — Homography Warp (final stage and output)

Finally, I compute a homography matrix mapping the 4 detected corners to the corners of the output rectangle and warp the original image to produce the final frontal view of the document.

**Method:** I mainly relied on `cv2.findHomography(corners, destination_points)` followed by `cv2.warpPerspective(image, H, (width, height))`

**Output dimensions:**
- **Width:** `1000 px`
- **Height:** `Approx ~1294 px` (derived from the 8.5 x 11 inch letter aspect ratio: `1000 * 11 / 8.5`)

![Rectified Output](output_samples/debug_input%20(1)/09_09_output_image.png)

---

## Summary of Key Parameters

| Parameter | Value | Purpose |
|---|---|---|
| `GAUSS_KSIZE` | `(11, 11)` | Gaussian blur kernel size |
| `BINARY_THRESH` | `160` | Fixed binary threshold value |
| approxPolyDP epsilons | `[0.02, 0.04, 0.06, 0.08, 0.10]` | Progressive polygon approximation |
| Output size | `1000 x 1294 px` | Derived from 8.5 x 11 letter aspect ratio |

