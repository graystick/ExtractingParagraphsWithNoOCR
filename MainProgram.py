# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 09:23:28 2026

@author: rtan5
"""

import cv2
import numpy as np
from matplotlib import pyplot as pt


def main():
    return True

##func for detecting tables

##func for detecting images in papers

##mask for tables and images


def pageToBinary(image_path): ##take a page from the folder and converts it to grayscale, then thresholds it to become a binary image for processing the text
    page = cv2.imread(image_path)
    pageGrayscale = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    
    _, pageBinary = cv2.threshold(pageGrayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return page, pageBinary
##page to Binary is gonna be used quite a few times so we made it a function for easy access
    


##hist_TextBlock = reads histogram projections where alphabetical characters show up
def findTextInPage(hist_TextBlock, minGap = 1, minLengthofText = 1):
    hasText = hist_TextBlock > 0 ##alphabetical characters would be 255, empty space is 0
    
    textBlocks_SE = [] ##store the starting point and ending points of text blocks
    start = None
    gapCounter = 0 
    
    for i, val in enumerate(hasText): #checks the image for lines of text based on its histogram projection and does this through a loop
        if val:
            if start is None:
                start = i ##start of new run
                gapCounter = 0
        else:
            if start is not None:
                gapCounter += 1     ##close each run only if the gap is long enough
                if gapCounter >= minGap: 
                    end = i - gapCounter + 1
                    if end - start >= minLengthofText:
                        textBlocks_SE.append((start, end))
                    start = None
                    gapCounter = 0
                    
    if start is not None:       #close run at the end of the array
        end = len(hasText) - gapCounter
        if end - start >= minLengthofText:
            textBlocks_SE.append((start, end))

    return textBlocks_SE

def colDetection(img_binary, minGap_col = 15, minLengthofText_col = 30):
    imageProjection_col = np.sum(img_binary, axis=0) ##sum of pixels in each column
    
    textColumns = findTextInPage(imageProjection_col, minGap =minGap_col, minLengthofText = minLengthofText_col) ##polymorph the columns width and gap into the findTextinPage function
    
    return textColumns ##uses the findTextinPage function to get the column width and gap



    