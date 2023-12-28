import numpy as np
import cv2 as cv
import tensorflow as tf
from keras.models import load_model

model = load_model('ocr_model.h5')

def preProcessImage(image_grid, block_size, C):
    #creation of 2x2 rectangle
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    #changing color of image to gray if it's not already
    grayscale = image_grid if len(image_grid.shape) == 2 else cv.cvtColor(image_grid, cv.COLOR_BGR2GRAY)
    #removing noise from image
    noise = cv.GaussianBlur(grayscale, (9, 9), 0)
    #converting the image to binary
    thresh = cv.adaptiveThreshold(noise, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block_size, C)
    #inverting black and white
    inverted = cv.bitwise_not(thresh, 0)
    #removal of extra white noise
    morph = cv.morphologyEx(inverted, cv.MORPH_OPEN, kernel)
    #dilation of the lines and numbers
    dilated = cv.dilate(morph, kernel, iterations=1)
    return dilated

def get_corners(image_grid):
    #getting the contours of the grid in order to find corners
    contours, _ = cv.findContours(image_grid, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #sorting by size and getting the biggest contour
    contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)
    largest_contour = np.squeeze(contours[0])

    #getting the max and min of each line
    sums = [sum(i) for i in largest_contour]
    differences = [i[0] - i[1] for i in largest_contour]
    top_left = np.argmin(sums)
    top_right = np.argmax(differences)
    bottom_left = np.argmax(sums)
    bottom_right = np.argmin(differences)

    corners = [largest_contour[top_left], largest_contour[top_right], largest_contour[bottom_left],
               largest_contour[bottom_right]]
    return corners

#calculating height and width
#using it to transform the grid if rotated
def transform(pts, image_grid): 
    pts = np.float32(pts)
    top_l, top_r, bot_l, bot_r = pts[0], pts[1], pts[2], pts[3]

    def pythagoras(pt1, pt2):
        return np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)

    width = int(max(pythagoras(bot_r, bot_l), pythagoras(top_r, top_l)))
    height = int(max(pythagoras(top_r, bot_r), pythagoras(top_l, bot_l)))
    square = max(width, height) // 9 * 9

    dim = np.array(([0, 0], [square - 1, 0], [square - 1, square - 1], [0, square - 1]), dtype='float32')
    matrix = cv.getPerspectiveTransform(pts, dim)
    warped = cv.warpPerspective(image_grid, matrix, (square, square))
    return warped

def get_grid_lines(image_grid, length=12):
    horizontal = np.copy(image_grid)
    cols = horizontal.shape[1]
    horizontal_size = cols // length
    horizontal_structure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv.erode(horizontal, horizontal_structure)
    horizontal = cv.dilate(horizontal, horizontal_structure)

    vertical = np.copy(image_grid)
    rows = vertical.shape[0]
    vertical_size = rows // length
    vertical_structure = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_size))
    vertical = cv.erode(vertical, vertical_structure)
    vertical = cv.dilate(vertical, vertical_structure)

    return vertical, horizontal

def create_grid_mask(vertical, horizontal):
    grid = cv.add(horizontal, vertical)
    grid = cv.adaptiveThreshold(grid, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 135, 2)
    grid = cv.dilate(grid, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)), iterations=2)
    pts = cv.HoughLines(grid, .3, np.pi / 90, 200)

    def draw_lines(image_grid, pts):
        image_grid = np.copy(image_grid)
        pts = np.squeeze(pts)
        for r, theta in pts:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * r
            y0 = b * r
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            cv.line(image_grid, (x1, y1), (x2, y2), (255, 255, 255), 2)
        return image_grid

    lines = draw_lines(grid, pts)
    return lines

def extractNumbers(image_grid):
    grid = cv.bitwise_not(image_grid, 0)

    cv.imshow("Numbers", grid)
    k = cv.waitKey(0)

    height, width = grid.shape
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
            box = grid[y1:y2, x1:x2]
             
            resized = cv.resize(box, (14, 28), interpolation=cv.INTER_LANCZOS4)
            if np.sum(resized) == 0:
                boxes.append(0)
                print(len(boxes))
                continue
            array = np.array([resized])
            flt = array.astype('float32')
            flt /= 255
            flt = flt.reshape((1,28,14,1))

            pred = np.argmax(model.predict(flt), axis=1)
            boxes.append(pred[0]+1)
    # Reshape the results to match the 9x9 Sudoku grid
    nbrs=np.array(boxes).reshape((9, 9))
    print(nbrs)
    return nbrs

#importing file
file = ["sudoku_grid_test.png","sudoku_rotated_grid.png"]
img = cv.imread(file[0])

grid = preProcessImage(img,3,2)
corners = get_corners(grid)
warped_grid = transform(corners, grid)
vertical, horizontal = get_grid_lines(warped_grid)
mask = create_grid_mask(vertical, horizontal)
numbers = cv.subtract(warped_grid, mask)
extractNumbers(numbers)