#!/usr/bin/python3
# B351/Q351 Fall 2020
# Do not share these assignments or their solutions outside of this class.

import csv
import itertools

class Board():

    ##########################################
    ####   Constructor
    ##########################################
    def __init__(self, filename):

        # initialize all of the variables
        self.n2 = 0
        self.n = 0
        self.spaces = 0
        self.board = None
        self.valsInRows = None
        self.valsInCols = None
        self.valsInBoxes = None
        self.unsolvedSpaces = None

        # load the file and initialize the in-memory board with the data
        self.loadSudoku(filename)


    # loads the sudoku board from the given file
    def loadSudoku(self, filename):

        with open(filename) as csvFile:
            self.n = -1
            reader = csv.reader(csvFile)
            for row in reader:

                # Assign the n value and construct the approriately sized dependent data
                if self.n == -1:
                    self.n = int(len(row) ** (1/2))
                    if not self.n ** 2 == len(row):
                        raise Exception('Each row must have n^2 values! (See row 0)')
                    else:
                        self.n2 = len(row)
                        self.spaces = self.n ** 4
                        self.board = {}
                        self.valsInRows = [set() for _ in range(self.n2)]
                        self.valsInCols = [set() for _ in range(self.n2)]
                        self.valsInBoxes = [set() for _ in range(self.n2)]
                        self.unsolvedSpaces = set(itertools.product(range(self.n2), range(self.n2)))

                # check if each row has the correct number of values
                else:
                    if len(row) != self.n2:
                        raise Exception('Each row must have the same number of values. (See row ' + str(reader.line_num - 1) + ')')

                # add each value to the correct place in the board; record that the row, col, and box contains value
                for index, item in enumerate(row):
                    if not item == '':
                        self.board[(reader.line_num-1, index)] = int(item)
                        self.valsInRows[reader.line_num-1].add(int(item))
                        self.valsInCols[index].add(int(item))
                        self.valsInBoxes[self.spaceToBox(reader.line_num-1, index)].add(int(item))
                        self.unsolvedSpaces.remove((reader.line_num-1, index))


    ##########################################
    ####   Utility Functions
    ##########################################

    # converts a given row and column to its inner box number
    def spaceToBox(self, row, col):
        return self.n * (row // self.n) + col // self.n

    # prints out a command line representation of the board
    def print(self):
        for r in range(self.n2):
            # add row divider
            if r % self.n == 0 and not r == 0:
                if self.n2 > 9:
                    print("  " + "----" * self.n2)
                else:
                    print("  " + "---" * self.n2)

            row = ""

            for c in range(self.n2):

                if (r,c) in self.board:
                    val = self.board[(r,c)]
                else:
                    val = None

                # add column divider
                if c % self.n == 0 and not c == 0:
                    row += " | "
                else:
                    row += "  "

                # add value placeholder
                if self.n2 > 9:
                    if val is None: row += "__"
                    else: row += "%2i" % val
                else:
                    if val is None: row += "_"
                    else: row += str(val)
            print(row)


    ##########################################
    ####   Move Functions - YOUR IMPLEMENTATIONS GO HERE
    ##########################################

    # makes a move, records it in its row, col, and box, and removes the space from unsolvedSpaces
    def makeMove(self, space, value):
        self.board[space] = value
        self.valsInRows[space[0]].add(value)
        self.valsInCols[space[1]].add(value)
        self.valsInBoxes[self.spaceToBox(space[0], space[1])].add(value)
        self.unsolvedSpaces.remove(space)

    # removes the move, its record in its row, col, and box, and adds the space back to unsolvedSpaces
    def undoMove(self, space, value):
        del self.board[space]
        self.valsInRows[space[0]].remove(value)
        self.valsInCols[space[1]].remove(value)
        self.valsInBoxes[self.spaceToBox(space[0], space[1])].remove(value)
        self.unsolvedSpaces.add(space)

    # returns True if the space is empty and on the board,
    # and assigning value to it if not blocked by any constraints
    def isValidMove(self, space, value):
        spaces = (space[0] <= self.n2) & (space[0] >= 0) & (space[1] <= self.n2) & (space[1] >= 0)
        col = value not in self.valsInCols[space[1]]
        row = value not in self.valsInRows[space[0]]
        box = value not in self.valsInBoxes[self.spaceToBox(space[0], space[1])]
        unsolved = space in self.unsolvedSpaces
        return unsolved and box and row and col and spaces

    # optional helper function for use by getMostConstrainedUnsolvedSpace
    def evaluateSpace(self, space):
        doms = 0
        for i in range(1, self.n2+1):
            if self.isValidMove(space, i):
                doms += 1
        return doms

    # gets the unsolved space with the most current constraints
    # returns None if unsolvedSpaces is empty
    def getMostConstrainedUnsolvedSpace(self):
        if len(self.unsolvedSpaces) > 0:
            move = None
            num = self.n2
            for space in self.unsolvedSpaces:
                nums = self.evaluateSpace(space)
                if nums < num:
                    move = space
                    num = nums
            return move
        else:
            return None


class Solver:
    ##########################################
    ####   Constructor
    ##########################################
    def __init__(self):
        pass

    ##########################################
    ####   Solver
    ##########################################

    # recursively selects the most constrained unsolved space and attempts
    # to assign a value to it

    # upon completion, it will leave the board in the solved state (or original
    # state if a solution does not exist)

    # returns True if a solution exists and False if one does not
    def solveBoard(self, board):
        move = board.getMostConstrainedUnsolvedSpace()
        if len(board.unsolvedSpaces) == 0:
            return True
        else:
            for i in range(1, board.n2+1):
                if board.isValidMove(move, i):
                    board.makeMove(move, i)
                    if self.solveBoard(board):
                        return True
                    board.undoMove(move, i)
            return False
            

if __name__ == "__main__":
    # change this to the input file that you'd like to test
    board = Board('/Users/benjamin/school/B351/distribution/a2/tests/example.csv')
    s = Solver()
    print(s.solveBoard(board))
    board.print()