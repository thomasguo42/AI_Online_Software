import cv2
import os
import sys
import numpy as np

print("Python version:", sys.version)
print("OpenCV version:", cv2.__version__)
print("DISPLAY environment variable:", os.environ.get("DISPLAY"))

# Create a blank image
img = cv2.imread("non_existent_image.jpg")  # This will fail gracefully if needed
if img is None:
    img = np.zeros((480, 640, 3), dtype="uint8")
    print("Created blank image due to missing input.")

# Try to display a window
cv2.imshow("Test Window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Window displayed successfully.")