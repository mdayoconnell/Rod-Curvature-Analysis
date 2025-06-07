import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label

def extract_skeleton_from_image(path, debug=False, threshhold = 99):
    """
    Extracts a clean skeleton from a grayscale image with a clearly visible white rod on black background.
    Avoids cv2.threshold and uses percentile-based binarization to isolate the rod.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at path: {path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Normalize to full 0-255 range
    normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)

    # Threshold for top X% brightest pixels (white rod)
    threshold_val = np.percentile(normalized, threshhold)  # keep top 1% brightest
    binary = (normalized >= threshold_val).astype(np.uint8)

    # Clean up using morphology (optional, depending on noise)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Skeletonize and clean
    skeleton = skeletonize(binary > 0)
    cleaned = remove_small_objects(label(skeleton), min_size=300)
    cleaned = cleaned > 0

    if debug:
        vis = np.ones_like(cleaned, dtype=np.uint8) * 255  # white bg
        vis[cleaned] = 0  # black skeleton

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(blurred, cmap='gray')
        plt.title('Blurred Image (to reduce noise)')

        plt.subplot(1, 3, 2)
        plt.imshow(binary, cmap='gray')
        plt.title(f'Binary Mask for Skeletonizing(>{threshold_val:.0f})')

        plt.subplot(1, 3, 3)
        plt.imshow(vis, cmap='gray')
        plt.title('Skeleton on White')

        plt.tight_layout()
        plt.show()

    return cleaned

# === TEST ===
if __name__ == "__main__":
    test_image_path = "/Users/micah/Desktop/Frame4.png"  # Update path as needed
    extract_skeleton_from_image(test_image_path, debug=True)