import cv2
import numpy as np
from skimage.morphology import skeletonize

from config import SMOOTHING_ITERATIONS


def find_red_point(image_path):
    """Ищет красную точку в изображении, возвращает (pixel_x, pixel_z) или None."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
    pts = np.where(cv2.bitwise_or(m1, m2) > 0)
    if len(pts[0]) == 0:
        return None
    return int(np.mean(pts[1])), int(np.mean(pts[0]))


def get_mask(image_path, iterations=SMOOTHING_ITERATIONS):
    """Загружает изображение и применяет сглаживание/бинаризацию."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    cur = img.copy()
    for _ in range(iterations):
        cur = cv2.GaussianBlur(cur, (5, 5), 0)
        _, cur = cv2.threshold(cur, 50, 255, cv2.THRESH_BINARY)
        cur = cv2.morphologyEx(cur, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return cur


def build_skeleton(mask):
    """Строит тонкий скелет (1px) маски трассы."""
    return skeletonize((mask > 128).astype(np.uint8)).astype(np.uint8)