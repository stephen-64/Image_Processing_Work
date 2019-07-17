import cv2 as cv
import argparse
import numpy as np
import sys
import time
from threading import Thread
if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue