# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 09:23:28 2026

@author: rtan5
"""

import cv2
import numpy as np
from matplotlib import pyplot as pt
import os


def main():
    if __name__ == '__main__':
        output_dir = 'paragraphs'

    for i in range(1, 9):
        filename = f"{i:03d}.png"
        path = os.path.join(filename)

        if not os.path.exists(path):
            print(f"Skipping {filename} (not found)")
            continue
        
        original, binary = pageToBinary(path)
        tableBoxes = tableDetection(binary)
        images = imageDetection(binary)
        
        masked = maskTablesandImages(
            binary,
            tableBoxes,
            images
        )
        
        print("Detected tables:", tableBoxes)

        pt.figure(figsize=(10, 12))
        pt.imshow(binary, cmap='gray')

        for x1, y1, x2, y2 in tableBoxes:
            pt.gca().add_patch(
                pt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor='red',
                    linewidth=2
                )
            )

        pt.title('Detected Tables')
        pt.axis('off')
        pt.show()

        print("Detected images:", images)

        pt.figure(figsize=(10, 12))
        pt.imshow(binary, cmap='gray')

        for x1, y1, x2, y2 in images:
            pt.gca().add_patch(
                pt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor='blue',
                    linewidth=2
                )
            )

        pt.title('Detected Images')
        pt.axis('off')
        pt.show()
        
        pt.figure(figsize=(10, 12))
        pt.imshow(masked, cmap='gray')
        pt.title('Binary Image After Table and Image Masking')
        pt.axis('off')
        pt.show()
        
        ##column detection after masking the image
        columns = colDetection(masked)
        print("Detected columns:", columns)
        
        ##row detection after masking the image
        lines = rowDetection(masked, columns)
        print("Detected lines:", lines)
        
        
        
        ##to be implemented (paragraph detection)
        paragraphs = paragraphDetection(lines)
        print("Detected paragraphs:", paragraphs)
        
        ##export paragraphs as individual images
        page_name = os.path.splitext(filename)[0]

        for paragraph_number, (x1, y1, x2, y2) in enumerate(paragraphs, start=1):
    
             ##add small padding around the paragraph
             padding = 5
    
             x1 = max(x1 - padding, 0)
             y1 = max(y1 - padding, 0)
    
             x2 = min(x2 + padding, original.shape[1])
             y2 = min(y2 + padding, original.shape[0])
    
             ##Crop from original image
             paragraph_image = original[y1:y2, x1:x2]
    
             ##Filename:
             ##001_paragraph_01.png
             output_path = os.path.join(output_dir, f"{page_name}_paragraph_{paragraph_number:02d}.png")
    
             cv2.imwrite(output_path, paragraph_image)
    
             print(
                 f"Saved paragraph {paragraph_number}: "
                 f"{output_path}"
             ) 
             
        ##visualise final paragraph detection
        pt.figure(figsize=(10, 12))

        ##convert BGR to RGB for Matplotlib
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        pt.imshow(original_rgb)
        
        for paragraph_number, (x1, y1, x2, y2) in enumerate(paragraphs, start=1):
        
            pt.gca().add_patch(
                pt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='blue', linewidth=2))
        
            pt.text(x1, y1, str(paragraph_number), color='blue',fontsize=10)
        
        pt.title(
            f"{filename} - Detected Paragraphs "
            f"({len(paragraphs)})"
        )
        
        pt.axis("off")
        pt.show()
        

##tables in the papers have long border lines so we dilate these so that all the lines merge into a single blob of black pixels in regardless of how far apart they are, and since paragraph text never has these long lines we can detect the tables from that
def tableDetection(img_binary, minWidth_table = 150, minHeight_table = 40, minIntersect_table = 4):
    
    img_binaryY, img_binaryX = img_binary.shape ##take the shape of the X(width) and Y(height) of the table
    kernelX_length = max(img_binaryX // 8, 60) ##find the largest values of both the width and height of the table
    kernelY_length = max(img_binaryY // 40, 25)
    
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelX_length, 1))    ##create kernel for both width and height using the max() values that we got before
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1 ,kernelY_length))
    
    linesX = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernelX)  ##remove the noise from the kernel for both width and height
    linesY = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernelY)
    
    intersect = cv2.bitwise_and(linesX, linesY) ##where the lines overlap
    
    ##dilates all the table border lines so that they all merge into one region creating an intersection
    tableline_mask = cv2.bitwise_or(linesX, linesY)
    kernelmerge = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    tableline_mask = cv2.dilate(tableline_mask, kernelmerge, iterations=2)
    
    tablecontours, _ = cv2.findContours(tableline_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    tableBoxes = []
    for i in tablecontours:
        x,y,w,h = cv2.boundingRect(i)
        if w < minWidth_table or h < minHeight_table:
            continue
        ## confirmation that the lines actually do intersect so that paragraph text isnt accidentally removed in the final process
        region = intersect[y:y + h, x:x + w]
        nPoints = cv2.connectedComponentsWithStats(region, connectivity=8)[0] - 1
        
        if nPoints >= minIntersect_table:
            tableBoxes.append((x,y,x + w,y + h))
        
    return tableBoxes

def boxOverlap(box_a, box_b, thresh=0.3):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = max(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    intersectedArea = (ix2 - ix1) * (iy2 - iy1)
    smallerArea = min((ax2 - ax1) * (ay2-ay1), (bx2-bx1) * (by2-by1))
    return smallerArea >0 and (intersectedArea / smallerArea) > thresh


##func for detecting images in papers
def imageDetection(img_binary, minWidth_image=200, minHeight_image=100, minArea_image=30000):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (15, 15)
    )

    merged = cv2.dilate(
        img_binary,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        merged,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    imageBoxes = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        if w < minWidth_image or h < minHeight_image:
            continue

        if w * h < minArea_image:
            continue

        region = img_binary[y:y+h, x:x+w]

        row_projection = np.sum(region, axis=1)

        row_runs = findTextInPage(
            row_projection,
            minGap = 2, 
            minLengthofText = 2
        )
        
        print("paragraphed lines" , groupLinestoParagraphs(row_runs))

        # A very small number of row groups suggests
        # the region is not ordinary paragraph text.
        if len(row_runs) <= 2:
            imageBoxes.append(
                (x, y, x + w, y + h)
            )
            continue

        gaps = []

        for i in range(len(row_runs) - 1):
            gap = (
                row_runs[i + 1][0]
                - row_runs[i][1]
            )

            gaps.append(gap)

        if len(gaps) == 0:
            continue

        median_gap = np.median(gaps)

        # Measure how consistently the gaps repeat
        deviations = [
            abs(gap - median_gap)
            for gap in gaps
        ]

        mean_deviation = np.mean(deviations)

        if median_gap > 0:
            irregularity = (
                mean_deviation / median_gap
            )
        else:
            irregularity = 0

        if (
            len(row_runs) >= 3
            and irregularity > 0.8
        ):
            imageBoxes.append(
                (x, y, x + w, y + h)
            )

    return imageBoxes


##mask for tables and images
def maskTablesandImages(binary, tableBoxes, imageBoxes):
    
    masked = binary.copy()
    
    ##mask tables
    for x1, y1, x2, y2 in tableBoxes:
        masked[y1:y2, x1:x2] = 0
        
    ##mask images
    for x1, y1, x2, y2 in imageBoxes:
        masked[y1:y2, x1:x2] = 0

    return masked

def pageToBinary(image_path): ##take a page from the folder and converts it to grayscale, then thresholds it to become a binary image for processing the text
    page = cv2.imread(image_path)
    pageGrayscale = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    
    _, pageBinary = cv2.threshold(pageGrayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return page, pageBinary


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


    ##rowDetection() function works the same in theory with the colDetection() function where we take the sum of pixels in each row across a whole page and then put the values of the minimum gap length and height and put it in the findTextinPage() function to get our rows
def rowDetection(coldetection_result, minGap_row = 2, minHeight_row = 4):
    imageProjection_row = np.sum(coldetection_result, axis = 1)
    textRows = findTextInPage(imageProjection_row, minHeight_row, minGap_row)
    
    return textRows


def groupLinestoParagraphs(textinpage, paragraph_gapF = 1.8): ##returns the start and end point of paragraphs
    if len(textinpage) == 0:
        return[]
    if len(textinpage) == 1:
        return[textinpage[0]]
    
    ##the gaps betwewen end of one line and the start of the next line
    gaps = [textinpage[i + 1][0] - textinpage[i][1] for i in range(len(textinpage) -1)]
    gapMedian = np.median(gaps) if len(gaps) > 0 else 0

    paragraphs = []
    para_start = textinpage[0][0]
    para_end = textinpage[0][1]
    
    for i in range(1, len(textinpage)):
        gap = textinpage[i][0] - textinpage[i-1][1]
        if gap > paragraph_gapF * max(gapMedian, 1):
            paragraphs.append((para_start, para_end))
            ##if gap is unusually large the new paragraph starts here
            para_start = textinpage[i][0]
        para_end = textinpage[i][1]
        
    paragraphs.append((para_start, para_end)) ##append the last paragraph to the list
    return paragraphs 

    


main()