import numpy as np
import cv2 as cv
import tensorflow as tf
from keras.models import load_model
from sudoku_solve import solve_sudoku, check_if_solvable
from py2cuda import apply_gaussian_blur
import threading
import queue
import time


model = load_model('ocr_model_retrained.h5') # ocr model

'''
Function used for grid detection.
It processes the image to get the contours of each element in it.
Each contour is analyzed to try and find the grid
Returns None or coordinates of grid contour
'''
def detect_grid(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (7, 7), 0)
    edged = cv.Canny(blurred, 50, 150, apertureSize=3) # Binary image with contours drawn

    contours, _ = cv.findContours(edged, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) # All the contours in the binary image

    img_area = image.shape[0] * image.shape[1]

    for cnt in contours:
        epsilon = 0.1 * cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4 and cv.isContourConvex(approx):
            x, y, w, h = cv.boundingRect(approx)
            aspect_ratio = float(w) / h
            contour_area = cv.contourArea(approx)

            # Check if the contour is square-like and large enough
            if 0.8 < aspect_ratio < 1.2 and contour_area > img_area * 0.1:
                # Check the uniformity of sides
                side_lengths = [cv.norm(approx[i % 4] - approx[(i + 1) % 4]) for i in range(4)]
                if max(side_lengths) < 1.5 * min(side_lengths):
                    return approx  # Contours of grid
    return None

'''
Preprocessing of the frames.
Gray -> Blurred -> Binary -> Inverted -> White noise removal -> Dilated
The Gaussian blur is done with a custom script using a CUDA kernel for GPU processing
'''
def preProcessImage(image_grid, block_size, C):
    
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2)) # Creation of 2x2 rectangle

    grayscale = image_grid if len(image_grid.shape) == 2 else cv.cvtColor(image_grid, cv.COLOR_BGR2GRAY)
    
    blur = apply_gaussian_blur(grayscale) # Removing noise from image
    cv.imshow("Noise", blur)
    
    thresh = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block_size, C) # Converting the image to binary
    cv.imshow("Thresh", thresh)
    
    inverted = cv.bitwise_not(thresh, 0) # Inverting black and white
    
    morph = cv.morphologyEx(inverted, cv.MORPH_OPEN, kernel) # Removal of extra white noise
    
    dilated = cv.dilate(morph, kernel, iterations=1) # Dilation of the lines and numbers
    cv.imshow("Dilated", dilated)

    return dilated

'''
Function returning an array of size 4 containing the 4 corners of the grid
'''
def get_corners(image_grid):
    contours, _ = cv.findContours(image_grid, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True) # Sorting by size and getting the biggest contour
    largest_contour = np.squeeze(contours[0])

    '''Getting the max and min of each line'''
    sums = [sum(i) for i in largest_contour]
    differences = [i[0] - i[1] for i in largest_contour]
    top_left = np.argmin(sums)
    top_right = np.argmax(differences)
    bottom_left = np.argmax(sums)
    bottom_right = np.argmin(differences)

    corners = [largest_contour[top_left], largest_contour[top_right], largest_contour[bottom_left],
               largest_contour[bottom_right]]
    return corners

'''
Function changing the perspective of the image in case of rotation.
Calculating height, width of grid and also new coordinates before warping.
'''
def transform(pts, image_grid): 
    pts = np.float32(pts)
    top_l, top_r, bot_l, bot_r = pts[0], pts[1], pts[2], pts[3] # Corners

    def pythagoras(pt1, pt2):
        return np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)

    width = int(max(pythagoras(bot_r, bot_l), pythagoras(top_r, top_l)))
    height = int(max(pythagoras(top_r, bot_r), pythagoras(top_l, bot_l)))
    square = max(width, height) // 9 * 9

    dim = np.array(([0, 0], [square - 1, 0], [square - 1, square - 1], [0, square - 1]), dtype='float32')
    matrix = cv.getPerspectiveTransform(pts, dim)

    warped = cv.warpPerspective(image_grid, matrix, (square, square)) # Warping of the image with all the coordinates calculated
    return warped

'''
Function returning all the lines of the grid, horizontal and vertical
Applying erosion and then dilation in order to avoid empty spaces in lines.
'''
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

'''
Creating grid mask based on its vertical and horizontal lines.
Used afterwards for comparison with template
'''
def create_grid_mask(vertical, horizontal):
    grid = cv.add(horizontal, vertical)
    grid = cv.adaptiveThreshold(grid, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 135, 2)
    grid = cv.dilate(grid, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)), iterations=2)
    pts = cv.HoughLines(grid, .3, np.pi / 90, 200)

    def draw_lines(image_grid, pts):
        if pts is None:
            return image_grid
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

'''
Extraction of each cell from the image of the sudoku grid.
Each cell is resized to match the size of the model and analyzed to see if empty or not.
Border is also removed because the training data used for ocr model is without border.

'''
def extractNumbers(image_grid):
    grid = cv.bitwise_not(image_grid, 0)
    height, width = image_grid.shape
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
            
            resized = cv.resize(box, (20, 20), interpolation=cv.INTER_LANCZOS4)
            if np.sum(resized) >= 99960: # Check if empty
                boxes.append(0)
                continue

            box_no_border = remove_border(box)
            resized = cv.resize(box_no_border, (20, 20), interpolation=cv.INTER_LANCZOS4)

            array = np.array([resized]) # Image converted in numpy array
            flt = array.astype('float32') # Array as float
            flt /= 255 # Normalization of pixel values from [0,255] to [0,1]
            flt = flt.reshape((1,20,20,1)) # (batch_size=1 item at a time, height, width, channel=grayscale)
            pred = np.argmax(model.predict(flt), axis=1) # Prediction using the OCR model
            #print("Predicted = ",pred[0]+1)
            boxes.append(pred[0]+1)
    
    nbrs=np.array(boxes).reshape((9, 9)) # Reshape the results to match the 9x9 Sudoku grid
    print(nbrs)
    return nbrs

'''
Function used in the number extraction method to prepare images for the ocr.
'''
def remove_border(img):
    blurred = cv.GaussianBlur(img, (5, 5), 0)

    # Apply adaptive thresholding and find contours
    thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour and its bounding box
        c = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(c)

        # Crop the image to the bounding box
        cropped = img[y:y+h, x:x+w]
        return cropped
    return img

'''
Function used to display solution on camera feed.
An inverse matrix is calculated as opposed to the warping technique.
Transform the given frame and put the solution on it.
'''
def overlay_solution(frame, corners, solution, original):
    if not isinstance(solution, (list, np.ndarray)) or not isinstance(original, (list, np.ndarray)):
        print("Solution or original is not a list or array.")
        return frame

    # Perspective transform for the Sudoku grid
    pts1 = np.float32(corners)
    pts2 = np.float32([[0, 0], [450, 0], [450, 450], [0, 450]])
    try:
        inv_matrix = cv.getPerspectiveTransform(pts2, pts1)
    except Exception as e:
        print(f"Error in perspective transform calculation: {e}")
        return frame 

    # Process each cell in the Sudoku grid
    for i in range(9):
        for j in range(9):
            if original[i][j] == 0 and solution[i][j] != 0:  # Only fill empty cells
                text = str(solution[i][j])
                (text_width, text_height), baseline = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 1, 3)
                origin = np.array([[j * 50 + 25 - text_width // 2, i * 50 + 25 + text_height // 2]], dtype=np.float32)
                try:
                    dst = cv.perspectiveTransform(origin[None, :, :], inv_matrix)
                except cv.error as e:
                    print(f"Error in perspectiveTransform: {e}")
                    return frame
                text_position = tuple(int(val) for val in dst[0, 0])
                cv.putText(frame, text, text_position, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    return frame

'''
OCR thread used to avoid freezing the script while predicting numbers.
Once the numbers are predicted, it solves the puzzle or returns an error.
'''
def ocr_processing(frame_queue, result_queue):
    while True:
        frame = frame_queue.get()
        if frame is None:
            result_queue.put(None)
            break
        extracted_numbers = extractNumbers(frame)
        if check_if_solvable(extracted_numbers):
            solved_board = solve_sudoku(np.copy(extracted_numbers))
            result_queue.put((solved_board, extracted_numbers))
        else:
            print("OCR incorrect")
            result_queue.put(None)

img = cv.imread("template.png")
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

solution = None # Array with puzzle solution
detected = None # Array with predicted numbers after OCR
start_time = None

# Settings of separate OCR thread 
frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue()
ocr_thread = threading.Thread(target=ocr_processing, args=(frame_queue, result_queue))
ocr_thread.start()

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break 

    grid_contour = detect_grid(frame)
    frame_copy = frame.copy()

    if grid_contour is not None:
            # If a grid is detected, draw its contour on the frame
            cv.drawContours(frame, [grid_contour], -1, (0, 255, 0), 3)  # Draw in green

            grid = preProcessImage(frame_copy, 11, 2)
            corners = get_corners(grid)

            if len(corners) == 4:  # Check if corners are correctly detected
                if solution is not None: # Always display solution if possible
                    frame = overlay_solution(frame, corners, solution, detected)
                if start_time is None or time.time() - start_time > 5: # If a solution is found, wait for 5s before analyzing frame

                    warped_grid = transform(corners, grid)
                    cv.imshow("Warped", warped_grid)

                    vertical, horizontal = get_grid_lines(warped_grid)
                    mask = create_grid_mask(vertical, horizontal)
                    
                    mask = cv.normalize(mask, None, 0, 255, cv.NORM_MINMAX).astype('uint8') # Ensure both images are of type uint8
                    res = cv.matchTemplate(mask, img, cv.TM_CCOEFF_NORMED) # Compare current grid with template

                    if (np.array(np.where(res >= 0.5))).size != 0: # Check if at least 50% similarity
                        numbers = cv.subtract(warped_grid, mask) # Remove grid and leave just numbers
                        if frame_queue.empty():
                            frame_queue.put(numbers) # OCR in parallel thread
                    if not result_queue.empty():
                        result = result_queue.get()
                        if result is not None:
                            solution, detected = result
                            if solution is not None:
                                start_time = time.time() # Start the 5s
        
    cv.imshow("Sudoku Solver", frame) # Display the frame

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

frame_queue.put(None)  # Stop OCR thread
ocr_thread.join()

cap.release()
cv.destroyAllWindows()