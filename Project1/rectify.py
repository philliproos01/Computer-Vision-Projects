import cv2
import numpy as np
import sys
import re
from pathlib import Path
from typing import Optional


#Automatic Document Scanner
#--------------------------
#Fully automatic version of document_scanner.py.  Instead of manual GrabCut
#interaction, this script detects page edges automatically using:

#Grayscale to Gaussian blur to Canny edge detection to contour search
#to approximate polygon (4 corners) to order corners to homography warp


OUTPUT_WIDTH = 1000
OUTPUT_HEIGHT = int(OUTPUT_WIDTH * 11 / 8.5)


#Gaussian blur kernel.. Larger meansmore smoothing.
GAUSS_KSIZE = (11, 11)
#Binary threshold value (0-255).  Pixels above this become white (paper).
#Raise if background is being included; lower if paper edges are lost.
BINARY_THRESH = 160

def synthetic_data(data_dir: Path):
    if not data_dir.exists():
        return []
    paths = list(data_dir.iterdir())
    jpgs = [p for p in paths if p.suffix.lower() in (".jpg", ".jpeg", ".png")]

    def sort_file(p: Path):
        m = re.search(r"\((\d+)\)", p.name)
        return int(m.group(1)) if m else 10**9

    return sorted(jpgs, key=sort_file)


def sort_corner(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)

    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]

    return ordered


def image_write(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image is None:
        return
    if image.dtype != np.uint8:
        image_to_write = image
        if image.dtype in (np.float32, np.float64):
            image_to_write = np.clip(image_to_write, 0, 1)
            image_to_write = (image_to_write * 255).astype(np.uint8)
        else:
            image_to_write = np.clip(image_to_write, 0, 255).astype(np.uint8)
    else:
        image_to_write = image
    cv2.imwrite(str(path), image_to_write)


class DebugWriter:
    def __init__(self, out_dir: Path, base_name: str):
        self.dir = out_dir / f"debug_{base_name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self._log_lines = []

    def write(self, name: str, image) -> None:
        self.step += 1
        safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")
        path = self.dir / f"{self.step:02d}_{safe_name}.png"
        image_write(path, image)

    def log(self, line: str) -> None:
        self._log_lines.append(str(line))

    def flush_log(self) -> None:
        if not self._log_lines:
            return
        path = self.dir / "debug_log.txt"
        path.write_text("\n".join(self._log_lines) + "\n", encoding="utf-8")


def counter_find(binary, image, min_area_ratio=0.05, dbg: Optional[DebugWriter] = None, label: str = ""):
    """
    Given a binary (single-channel uint8) image, find the largest 4-sided
    polygon.  Tries multiple approxPolyDP epsilon values and falls back to
    convex-hull approximation.  Returns (approx_contour, idx, area, peri) or None.
    """
    counts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not counts:
        return None
    image_area = image.shape[0] * image.shape[1]
    
    counts_sorted = sorted(counts, key=cv2.contourArea, reverse=True)[:10]

    #direct approxPolyDP with more laxed epsilon values
    epsilons = [0.02, 0.04, 0.06, 0.08, 0.10]
    for eps in epsilons:
        for i, cnt in enumerate(counts_sorted):
            area = float(cv2.contourArea(cnt))
            if area < min_area_ratio * image_area:
                continue
            peri = float(cv2.arcLength(cnt, True))
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) == 4:
                if dbg:
                    dbg.log(f"{label} quad found: eps={eps} cnt[{i}] area={area:.1f}")
                return approx, i, area, peri

    #convex hull fallback
    for i, cnt in enumerate(counts_sorted):
        area = float(cv2.contourArea(cnt))
        if area < min_area_ratio * image_area:
            continue
        hull = cv2.convexHull(cnt)
        peri = float(cv2.arcLength(hull, True))
        for eps in epsilons:
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                if dbg:
                    dbg.log(f"{label} quad via hull: eps={eps} cnt[{i}] area={area:.1f}")
                return approx, i, area, peri

    return None


def blob_mask(binary):
    """
    Keep only the largest connected white region in a binary image.
    This suppresses small background noise blobs (e.g. grass stems).
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary
    
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.zeros_like(binary)
    mask[labels == largest] = 255
    return mask


def page_detect(image, dbg: Optional[DebugWriter] = None):
    #Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if dbg:
        dbg.write("01_gray", gray)

    #Gaussian blur
    #Adjust GAUSS_KSIZE at top of file (line ~33)
    blurred = cv2.GaussianBlur(gray, GAUSS_KSIZE, 0)
    if dbg:
        dbg.write("02_blurred", blurred)

    #Binary threshold
    #Adjust BINARY_THRESH at top of file 
    _, binary = cv2.threshold(blurred, BINARY_THRESH, 255, cv2.THRESH_BINARY)
    if dbg:
        dbg.write("03_binary", binary)
        dbg.log(f"GAUSS_KSIZE={GAUSS_KSIZE}  BINARY_THRESH={BINARY_THRESH}")

    #Remove internal text with morphological close
    #A large close kernel fills the black text holes inside the white
    #paper region, turning it into a solid white blob.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=3)
    if dbg:
        dbg.write("04_closed", closed)

    #Clean up background noise with morphological open
    #Open removes small bright specks in the background that survived
    #the threshold (e.g. grass stems, reflections).
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel, iterations=2)
    if dbg:
        dbg.write("05_opened", opened)

    #Keep only the largest white blob (the paper)
    blob = blob_mask(opened)
    if dbg:
        dbg.write("06_blob", blob)

    #Find the 4-corner quad
    result = counter_find(blob, image, dbg=dbg, label="blob")
    if result is None:
        if dbg:
            dbg.log("selected=None (no quad found)")
        return None

    page_contour, idx, area, peri = result
    poly_corner = sort_corner(page_contour)

    if dbg:
        dbg.log(f"selected=candidate[{idx}] area={area:.1f} peri={peri:.1f}")
        dbg.log(f"corners_ordered={poly_corner.tolist()}")

        corners_viz = image.copy()
        pts_int = poly_corner.astype(int)
        cv2.polylines(corners_viz, [pts_int], True, (255, 0, 0), 3)
        for i, pt in enumerate(pts_int):
            cv2.circle(corners_viz, tuple(pt), 8, (0, 0, 255), -1)
            cv2.putText(
                corners_viz,
                str(i),
                (int(pt[0]) + 10, int(pt[1]) + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        dbg.write("07_corner_selection", corners_viz)

    return poly_corner


def final_output(image, corners, dbg: Optional[DebugWriter] = None):
    poly_corner = np.array(
        [
            [0, 0],
            [OUTPUT_WIDTH - 1, 0],
            [OUTPUT_WIDTH - 1, OUTPUT_HEIGHT - 1],
            [0, OUTPUT_HEIGHT - 1],
        ],
        dtype=np.float32,
    )

    H, mask = cv2.findHomography(corners, poly_corner)
    if dbg:
        if H is None:
            dbg.log("homography=None")
        else:
            dbg.log("homography=" + np.array2string(H, precision=4, suppress_small=True))
        if mask is not None:
            dbg.log(f"homography_inliers={int(mask.sum())}/{mask.size}")

    if H is None:
        return None

    output_image = cv2.warpPerspective(image, H, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    if dbg:
        dbg.write("09_output_image", output_image)
    return output_image



project_dir = Path(__file__).resolve().parent
out_dir = project_dir / "output_samples"
out_dir.mkdir(parents=True, exist_ok=True)

folder_name = sys.argv[1] if len(sys.argv) > 1 else "synthetic_data"
data_dir = project_dir / folder_name

image_paths = synthetic_data(data_dir)
if not image_paths:
    print(f"No image samples found in {data_dir}")
    raise SystemExit(1)

print(f"Processing {len(image_paths)} image(s)\n")

successes = 0
failures = 0
failed_images = []

for image_path in image_paths:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  [SKIP] Could not read: {image_path}")
        failures += 1
        failed_images.append(str(image_path))
        continue

    dbg = DebugWriter(out_dir, image_path.stem)
    dbg.write("00_original", image)
    dbg.log(f"image_path={image_path}")
    dbg.log(f"image_shape={image.shape}")

    poly_corner = page_detect(image, dbg=dbg)
    if poly_corner is None:
        dbg.flush_log()
        print(f"  [FAIL] Could not detect 4-corner page contour: {image_path.name}")
        failures += 1
        failed_images.append(image_path.name)
        continue

    output_image = final_output(image, poly_corner, dbg=dbg)
    if output_image is None:
        dbg.flush_log()
        print(f"  [FAIL] Homography/warp failed: {image_path.name}")
        failures += 1
        failed_images.append(image_path.name)
        continue

    m = re.search(r"\((\d+)\)", image_path.stem)
    num = m.group(1) if m else image_path.stem
    out_name = f"output ({num}).jpg"
    out_path = out_dir / out_name
    cv2.imwrite(str(out_path), output_image)
    dbg.log(f"output_image_path={out_path}")
    dbg.flush_log()

    print(f"  [OK]   {image_path.name}  to  {out_path}")
    successes += 1

print(f"\nDone.  Successes: {successes}  Failures: {failures}")
if failed_images:
    print("Failed image scans:")
    for failed in failed_images:
        print(f"  - {failed}")
print(f"Outputs saved to: {out_dir}")
