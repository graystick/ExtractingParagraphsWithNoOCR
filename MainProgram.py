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

##tables in the papers have long border lines so we dilate these so that all the lines merge into a single blob of black pixels in regardless of how far apart they are, and since paragraph text never has these long lines we can detect the tables from that
def tableDetection(img_binary, minWidth_table = 150, minHeight_table = 40, minIntersect_table = 4):
    
    img_binaryY, img_binaryX = img_binary.shape ##take the shape of the X(width) and Y(height) of the table
    kernelX_length = max(img_binaryX // 8, 60) ##find the largest values of both the width and height of the table
    kernelY_length = max(img_binaryY // 40, 25)
    
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelX_length), 1)    ##create kernel for both width and height using the max() values that we got before
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, 1 ,(kernelY_length))
    
    linesX = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernelX)  ##remove the noise from the kernel for both width and height
    linesY = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernelY)
    
    intersect = cv2.bitwise_and(linesX, linesY) 
    tableline_mask = cv2.bitwise_or(linesX, linesY)
    kernelmerge = cv2.getStructuringElement(cv2.MORPH_RECT, 15,15)
    tableline_mask = cv2.dilate(tableline_mask, kernelmerge, iterations=2)
    
    tablecontours = cv2.findCountours(tableline_mask, cv2.RETR.EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    tableBoxes = []
    for i in tablecontours:
        x,y,w,h = cv2.boundingRect(i)
        if w < minWidth_table or h < minHeight_table:
            continue
        region = intersect[y:y + h, x:x + w]
        nPoints = cv2.connectedComponentsWithStats(region, connectingSections=8)[0] - 1
        
        if nPoints >= minIntersect_table:
            tableBoxes.append((x,y,x + w,y + h))
        
    return tableBoxes


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
    imageProjection_col = np.sum(img_binary, axis=0) ##sum of pixels in each column across a whole page
    
    textColumns = findTextInPage(imageProjection_col, minGap =minGap_col, minLengthofText = minLengthofText_col) ##polymorph the columns width and gap into the findTextinPage function
    
    return textColumns ##uses the findTextinPage function to get the column width and gap

def rowDetection(coldetection_result, minGap_row = 2, minHeight_row = 4):
    imageProjection_row = np.sum(coldetection_result, axis = 1)
    textRows = findTextInPage(imageProjection_row, minGap = minGap_row, minLengthofText = minHeight_row)
    
    return textRows