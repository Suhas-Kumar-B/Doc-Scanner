import cv2
import numpy as np
import rect

def scan_document(file_path):
    image = cv2.imread(file_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image from {file_path}.")
    image = cv2.resize(image, (1500, 880))
    orig = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(blurred, 0, 50)
    orig_edged = edged.copy()

    (contours, _) = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    target = None
    for c in contours:
        p = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * p, True)
        if len(approx) == 4:
            target = approx
            break
    if target is None:
        raise ValueError("Could not find a quadrilateral contour in the image.")

    approx = rect.rectify(target)
    pts2 = np.float32([[0, 0], [800, 0], [800, 800], [0, 800]])
    M = cv2.getPerspectiveTransform(approx, pts2)
    dst = cv2.warpPerspective(orig, M, (800, 800))

    cv2.drawContours(image, [target], -1, (0, 255, 0), 2)

    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    ret, th1 = cv2.threshold(dst, 127, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    th3 = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    ret2, th4 = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return {
        "original": orig,
        "gray": gray,
        "blurred": blurred,
        "edged": orig_edged,
        "outline": image,
        "scanned": dst,
        "binary_thresh": th1,
        "mean_thresh": th2,
        "gaussian_thresh": th3,
        "otsu_thresh": th4
    }