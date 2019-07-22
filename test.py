import numpy as np
import cv2
import cvui
from decimal import Decimal

WINDOW_NAME = 'CVUI Test'

# Initialize cvui and create/open a OpenCV window.
cvui.init(WINDOW_NAME)
# Create a frame to render components to.
frame = np.zeros((200, 400, 3), np.uint8)

while True:
    # Clear the frame.
    frame[:] = (49, 52, 49)
    # Render a message in the frame at position (10, 15)
    cvui.text(frame, 10, 15, 'Hello world!')
    value = 12.0
    cvui.trackbar(frame, 40, 30, 220, value, 10.0, 15.0)
    # Show frame in a window.
    cvui.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(20) == 27:
        break