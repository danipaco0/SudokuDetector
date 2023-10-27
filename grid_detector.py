import numpy as np
import cv2 as cv
import sys
import os

def preProcessImage(image_grid):
    #changing color to gray for better image processing
    gray = cv.cvtColor(image_grid, cv.COLOR_BGR2GRAY)

    blur = cv.GaussianBlur(gray,(5,5),0)

    #function to create a binary image
    threshold = cv.threshold(blur,0,255,cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]

    #function to return the connected components from the image
    #the second argument is the connectivity (4 or 8) to determine nbr of neighbors
    #the third is the threshold type
    return cv.connectedComponentsWithStats(threshold,4,cv.CV_32S)

def extractNumbers(image_grid):
    gray = cv.cvtColor(image_grid, cv.COLOR_BGR2GRAY)
    height, width = gray.shape
    box_height = height // 9
    box_width = width // 9
    boxes = []
    for row in range(9):
        for col in range(9):
            # Calculate the coordinates for each box
            y1 = row * box_height
            y2 = (row + 1) * box_height
            x1 = col * box_width
            x2 = (col + 1) * box_width
            box = gray[y1:y2, x1:x2]
            boxes.append(box)
    nbrs = np.zeros((9,9))
    templates = "digits_templates"
    files = os.listdir(templates)
    for i, b in enumerate(boxes):
        start = 1
        for f in files:
            template = cv.imread(os.path.join(templates,f),cv.IMREAD_GRAYSCALE)
            template = cv.resize(template, (box_width, box_height))
            result = cv.matchTemplate(b, template, cv.TM_CCOEFF)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
            if max_val > 0.9:
                nbrs[i//9][i-(i//9)*9] = start
                break
            if start == 9: start = 1
            else:start+=1
    return nbrs

#importing file
file = ["sudoku_grid_test.png","sudoku_rotated_grid.png"]
img = cv.imread(file[0])

out = preProcessImage(img)

#numLabels = number of components
#labels = position of pixel
#stats = stat of component (coordinates and area)
#centroid = center of component
(numLabels, labels, stats, centroid) = out
component = (labels==1).astype("uint8")*255

#get largest area of the image
corners, _ = cv.findContours(component,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
sudoku = max(corners, key=cv.contourArea)

#sorting of corners
sudoku_corners = np.zeros((4, 2), dtype=np.float32)
points = np.squeeze(sudoku)
sudoku_corners[0] = points[np.argmin(points.sum(1))]
sudoku_corners[2] = points[np.argmax(points.sum(1))]
sudoku_corners[1] = points[np.argmin(np.diff(points, axis=1))]
sudoku_corners[3] = points[np.argmax(np.diff(points, axis=1))]

#rotating and warping grid
output_size = (500,500)
perspective_matrix = cv.getPerspectiveTransform(sudoku_corners, np.array([[0, 0], [output_size[0], 0], [output_size[0], output_size[1]], [0, output_size[1]]], dtype=np.float32))
warped_sudoku = cv.warpPerspective(img, perspective_matrix, output_size)

numbers = extractNumbers(warped_sudoku)
print(numbers)

cv.imshow("Sudoku grid", warped_sudoku)
k = cv.waitKey(0)

if k == ord("q"):
    sys.exit()
