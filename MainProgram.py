# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 09:23:28 2026

@author: rtan5
"""

import cv2
import numpy as np
from matplotlib import pyplot as pt

def pageToBinary(image_path): ##take a page from the folder and converts it to grayscale, then thresholds it to become a binary image for processing the text
    page = cv2.imread(image_path)
    pageGrayscale = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    
    _, pageBinary = cv2.threshold(pageGrayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return page, pageBinary




