def is_valid(board, row, col, num):
    # Check if the number is not repeated in the row
    for x in range(9):
        if board[row][x] == num:
            return False

    # Check if the number is not repeated in the column
    for x in range(9):
        if board[x][col] == num:
            return False

    # Check if the number is not repeated in the 3x3 square
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            if board[i + start_row][j + start_col] == num:
                return False
    return True

def get_row(puzzle, row_num):
    return puzzle[row_num]


def get_column(puzzle, col_num):
    return [puzzle[i][col_num] for i, _ in enumerate(puzzle[0])]


def get_square(puzzle, row_num, col_num):
    square_x = row_num // 3
    square_y = col_num // 3
    coords = []
    for i in range(3):
        for j in range(3):
            coords.append((square_x * 3 + j, square_y * 3 + i))
    return [puzzle[i[0]][i[1]] for i in coords]

def find_empty(board):
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 0:
                return i, j  # row, col
    return None

def print_board(board):
    for row in board:
        print(" ".join(str(num) for num in row))

def check_if_solvable(unsolved_puzzle):
    for i in range(9):
        if sum(set(get_row(unsolved_puzzle, i))) != sum(get_row(unsolved_puzzle, i)):
            return False
        if sum(set(get_column(unsolved_puzzle, i))) != sum(get_column(unsolved_puzzle, i)):
            return False
        if sum(set(get_square(unsolved_puzzle, i, i))) != sum(get_square(unsolved_puzzle, i, i)):
            return False
    return True

def solve_sudoku(board, depth=0, max_depth=81):
    if depth > max_depth:
        return False # Unsolvable

    empty = find_empty(board)
    if not empty:
        print_board(board)
        return board  # Puzzle solved

    row, col = empty
    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num
            solved_board = solve_sudoku(board, depth + 1, max_depth)
            if solved_board is not False:
                return solved_board  # Return the solved board
            board[row][col] = 0  # Reset the cell and backtrack

    return False