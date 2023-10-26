import numpy as np
import cv2 as cv
import sys

#importing file
file = "sudoku_grid_test.png"
img = cv.imread(file)

#changing color to gray for better image processing
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray,(5,5),0)

#function to create a binary image
threshold = cv.threshold(blur,0,255,cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]

#function to return the connected components from the image
#the second argument is the connectivity (4 or 8) to determine nbr of neighbors
#the third is the threshold type
out = cv.connectedComponentsWithStats(threshold,4,cv.CV_32S)

#numLabels = number of components
#labels = position of pixel
#stats = stat of component (coordinates and area)
#centroid = center of component
(numLabels, labels, stats, centroid) = out

component = (labels==1).astype("uint8")*255
cv.imshow("Sudoku grid", component)

k = cv.waitKey(0) 
if k == ord("q"):
    sys.exit()
