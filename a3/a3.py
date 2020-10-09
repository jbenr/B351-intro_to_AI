#!/usr/bin/python3

# B351/Q351 Spring 2020
# Professor Saúl Blanco
# Do not share these assignments or their solutions outside of this class.

###################################
#                                 #
# Assignment 3: Search Algorithms #
#                                 #
###################################

import State
import Board
import heapq

STOP = -1
CONTINUE = 0


#################################
# Problem 1 - Fringe Expansion
#################################
# Objective:
# 1) Write a function that adds the possible states that we can get to
#    from the current state to the end of the fringe.
#
# Notes:
# (1) This function should not return or yield anything but just update the contents of the fringe
# (2) board_object.slide_blank is error-safe. It will return None if it is impossible to slide the blank

def expand_fringe(current_state, fringe):
    b = current_state.board
    valid_moves = [(1,0), (0,1), (-1,0), (0,-1)]
    for move in valid_moves:
        if b.slide_blank(move) is not None:
            fringe.append(State.State(b.slide_blank(move), current_state, current_state.depth+1, 1))
    

########################################
# Problem 2 - BFS (Breadth First Search)
########################################
# Objectives:
# (1) Write a function that implements a single iteration of the BFS algorithm
#     by considering the first state from the fringe.
#     (Returns STOP if the fringe is empty.)
#     See the project documentation for more details.

def breadth_first_search(fringe, max_depth, goal_board):
    if len(fringe) < 1:
        return STOP
    f = fringe.pop(0)
    b = f.board
    dep = f.depth
    if dep > max_depth:
        return CONTINUE
    elif b == goal_board:
        return f
    else:
        fringe.append(expand_fringe(f, fringe))
        return CONTINUE


def uninformed_solver(start_board, max_depth, goal_board):
    """
        Looping function which calls breadth_first_search until it finds a solution (a State object) or
        until STOP has been returned. Does not consider States below max_depth.
        If the goal is reached, this function should return the Goal State,
        which includes a path to the goal. Otherwise, returns None.
    """
    fringe = [State.State(start_board, None, 0, 0)]
    found = CONTINUE
    while found == CONTINUE:
        found = breadth_first_search(fringe, max_depth, goal_board)
    if isinstance(found, State.State):
        # Found goal!
        return found
    # Max depth reached...
    return None


####################################
# Problem 3 - UCS f-value Function
####################################
# Objectives:
# (1) Write a function that takes a board and depth and returns the f-value
#     (priority) that board should have in a uniform-cost search scenario.

def ucs_f_function(board, current_depth):
    #goal_board = []
    #l = len(board.matrix)
    #for i in range(0, l):
    #    goal_board.append([])
    #for i in range(0, l):
    #    for j in range(0, l):
    #        goal_board[j].append((i+1)+(l*j))
    #goal_board[l-1][l-1] = 0
    #cur_zero = board.find_element(0)
    #goal_zero = goal_board.find_element(0)
    #start_zero = (0, l-1)
    #dep = abs(start_zero[0] - cur_zero[0]) + abs(start_zero[1] - cur_zero[1])
    return current_depth


###########################################
# Problem 4 - A* f-value Function Factory
###########################################
# Objectives:
# (1) Given a heuristic function and a goal board, returns a f-value FUNCTION
#     (like ucs_f_function) that evaluates boards and depths as in the A* algorithm.
#
# Notes:
# (1) It may be helpful to consult your solution for a1.compose here.

def a_star_f_function_factory(heuristic, goal_board):
    return lambda current_board, depth: heuristic(current_board, goal_board) + ucs_f_function(current_board, depth)


# Here is an example heuristic function.
def manhattan_distance(current_board, goal_board):
    total = 0
    goal_matrix = goal_board.matrix
    for goal_r in range(len(goal_board.matrix)):
        for goal_c in range(len(goal_board.matrix[0])):
            val = goal_matrix[goal_r][goal_c]
            if val == 0:
                continue
            current_r, current_c = current_board.find_element(val)
            total += abs(goal_r - current_r) + abs(goal_c - current_c)
    return total

#################################
# Problem 5 - Your Own Heuristic
#################################
# Objectives:
# (1) Write a function that takes current_board and goal_board as arguments and
#     returns an estimate of how many moves it will take to reach the goal board.
#     Your heuristic must be admissible (never overestimate cost to goal), but
#     it does not have to be consistent (never overestimate step costs).
#
# Notes:
# (1) This heuristic should be admissible, but greater than (closer to the real
#     value than) the manhattan distance heuristic on average. That makes it a
#     better heuristic.

def my_helper(current_board, goal_board, val):
    num = 0
    g = goal_board.find_element(val)
    c = current_board.find_element(val)
    num += (abs(g[0] - c[0]) + abs(g[1] - c[1]))
    return num

def lin_conflicts(board, goal_board):
    num = 0
    m = board.matrix
    g = goal_board.matrix
    l = len(m[0])
    for r in range(l):
        if 0 not in m[r]:
            for v in range(l):
                if m[r][v] in g[r]:
                    for i in range(l-v):
                        if (m[r][v+i] in g[r]) and (m[r][v] != g[r][v]) and (m[r][v+i] != g[r][v+i]) and (m[r][v] != 0) and (m[r][v+i] != 0):
                            num += 2
                            break
                        break
                    break
    return num

def lin_conflicts2(board, goal_board):
    num = 0
    m = board.matrix
    g = goal_board.matrix
    l = len(m[0])
    for r in range(l):
        vals = []
        for v in range(l):
            if (m[r][v] in g[r]):
                vals.append(m[r][v])
        if (len(vals) == l) and 0 not in vals:
            for j in range(l):
                if m[r][j] != g[r][j]:
                    num += 1
    return num

#def transpose(board):
    m = board
    width = len(m)
    b = [[m[j][i] for j in range(width)] for i in range(width)]
    return b

def my_heuristic(current_board, goal_board):
    num = 0
    l = len(current_board.matrix)
    for i in range((l*l)-1):
        num = num + my_helper(current_board, goal_board, i+1)
    return num + lin_conflicts2(current_board, goal_board)
    

#################################
# Problem 6 - Informed Expansion
#################################
# Objectives:
# (1) Write a function that expands the fringe using the f-value function
#     provided. Note that States automatically sort by their f-values.
#
# Notes:
# (1) This function should update the contents of the fringe using heapq.

def informed_expansion(current_state, fringe, f_function):
    heapq.heapify(fringe)
    valid_moves = [(-1,0), (0,1), (1,0), (0, -1)]
    for move in valid_moves:
        cur = current_state.board
        if cur.slide_blank(move) is not None:
            cur = cur.slide_blank(move)
            new = State.State(cur, current_state, current_state.depth+1, f_function(cur, current_state.depth+1))
            heapq.heappush(fringe, new)


#################################
# Problem 7 - Informed Search
#################################
# Objectives:
# (1) Write a function that implements a single iteration of the
#     A*/UCS algorithm by considering the top-priority state from the fringe.
#     (Returns STOP if the fringe is empty.)
#     See the project documentation for more details.

def informed_search(fringe, goal_board, f_function, explored):
    if not fringe:
        return STOP
    top = heapq.heappop(fringe)
    if (top.board in explored.keys()) and (top.fvalue >= explored[top.board]):
        return CONTINUE
    else:
        explored[top.board] = top.fvalue
    if top.board == goal_board:
        return top
    else:
        informed_expansion(top, fringe, f_function)
        return CONTINUE


def informed_solver(start_board, goal_board, f_function):
    """
        Looping function which calls informed_search until it finds a solution
        (a State object) or until STOP has been returned.
        If the goal is reached, this function should return the Goal State,
        which includes a path to the goal. Otherwise, returns None.
    """
    fringe = [State.State(start_board, None, 0, f_function(start_board, 0))]
    explored = {}
    found = CONTINUE
    while found == CONTINUE:
        found = informed_search(fringe, goal_board, f_function, explored)
    if isinstance(found, State.State):
        return found
    return None


def ucs_solver(start_board, goal_board):
    return informed_solver(start_board, goal_board, ucs_f_function)


def a_star_solver(start_board, goal_board, heuristic):
    f_function = a_star_f_function_factory(heuristic, goal_board)
    return informed_solver(start_board, goal_board, f_function)

#################################
# Bonus Problem - IDS (10pts)
#################################
# Implement IDS in any way you choose. You will probably want to write multiple
# helper functions; be sure to document these appropriately.
#
# ids should take a start board and goal board and then perform multiple
# depth-first searches, with the maximum depth increasing from 0 all the way to
# final depth.
#
# If there is a solution within final_depth moves, ids should return the board.


#def ids(start_board, goal_board, final_depth):
#    raise NotImplementedError
###########################
# Main method for testing #
###########################


def main():
    # 8-Puzzle Tests!
    goal_board = Board.Board([[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 0]])

    simple_board = Board.Board([[1, 2, 0],
                              [4, 5, 3],
                              [7, 8, 6]])

    # Simple test case for expand_fringe
    fringe1 = []
    node1 = State.State(simple_board, None, 0, 0)
    expand_fringe(node1, fringe1)
    assert State.State(simple_board.slide_blank((-1, 0)), node1, 0, 0) not in fringe1
    assert State.State(simple_board.slide_blank((0, -1)), node1, 0, 1) in fringe1


    # Simple test case for breadth_first_search
    fringe1 = []
    node1 = State.State(simple_board, None, 0, 0)
    expand_fringe(node1, fringe1)
    assert breadth_first_search(fringe1, 3, goal_board) == CONTINUE
    fringe1[0] = State.State(goal_board, node1, 0, 0)
    assert type(breadth_first_search(fringe1, 3, goal_board)) is State.State

    # Simple test case for ucs_f_function
    node1 = State.State(simple_board, None, 0, 0)
    assert ucs_f_function(node1.board, 0) == 0

    # Simple test case for a_star_f_function
    # -> This checks that the return type is correct
    assert hasattr(a_star_f_function_factory(None, goal_board), '__call__')

    # This section is for you to create tests for your own heuristic
    boardNT = Board.Board([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 0]])
    boardT = Board.Board([[1, 4, 7],
                         [2, 5, 8],
                         [3, 6, 0]])
    boardLin = Board.Board([[3, 2, 1],
                            [4, 5, 6],
                            [7, 8, 0]])
    boardTest = Board.Board([[1, 2, 3],
                            [4, 8, 6],
                            [7, 5, 0]])
    #assert Board.Board(transpose(boardNT)) == boardT
    assert lin_conflicts(boardNT, goal_board) == 0
    assert lin_conflicts(boardLin, goal_board) == 2

    boardGrader = Board.Board([[1, 2, 3],
                               [0, 5, 6],
                               [4, 7, 8]])
    assert lin_conflicts2(boardGrader, goal_board) == 0
    board_A = Board.Board([[1, 2, 3],
                            [7, 5, 6],
                            [4, 8, 0]])
    assert lin_conflicts2(board_A, goal_board) == 0

    my_board = Board.Board([[7, 3, 1],
                            [0, 6, 2],
                            [8, 5, 4]])
    assert my_helper(my_board, goal_board, 1) == 2
    assert my_helper(my_board, goal_board, 0) == 3
    
    assert my_heuristic(simple_board, goal_board) >= manhattan_distance(simple_board, goal_board)
    assert my_heuristic(my_board, goal_board) >= manhattan_distance(my_board, goal_board)
    assert my_heuristic(boardTest, goal_board) <= 2
    boardTest2 = Board.Board([[1, 2, 3],
                              [4, 5, 6],
                              [7, 0, 8]])
    assert my_heuristic(boardTest2, goal_board) <= 1

    assert my_helper(my_board, goal_board, 1) == 2
    assert my_helper(my_board, goal_board, 0) == 3

    # Simple test for Informed Expansion
    node1 = State.State(simple_board, None, 0, 0)
    fringe1 = []
    informed_expansion(node1, fringe1, ucs_f_function)
    assert State.State(simple_board.slide_blank((-1, 0)), node1, 0, 0) not in fringe1
    assert State.State(simple_board.slide_blank((0, -1)), node1, 0, 1) in fringe1

    # Simple test for Informed Search
    fringe1 = []
    explored = {}
    node1 = State.State(simple_board, None, 0, 0)
    expand_fringe(node1, fringe1)
    assert informed_search(fringe1, goal_board, ucs_f_function, explored) == CONTINUE
    fringe1[0] = State.State(goal_board, node1, 0, 0)
    assert type(informed_search(fringe1, goal_board, ucs_f_function, explored)) is State.State
    informed_expansion(node1, fringe1, ucs_f_function)
    informed_search(fringe1, goal_board, ucs_f_function, explored)
    assert len(explored) == 2

    # Simple test for IDS
    node1 = State.State(simple_board, None, 0, 0)
    #assert ids(node1.board, goal_board, 1) is None
    #result = ids(node1.board, goal_board, 2)
    #assert type(result) is Board.Board

    # 15-Puzzle Tests

    goal_board = Board.Board([[1, 2, 3, 4],
                              [5, 6, 7, 8],
                              [9, 10, 11, 12],
                              [13, 14, 15, 0]])

    simple_board = Board.Board([[1, 2, 3, 0],
                                [5, 6, 7, 4],
                                [9, 10, 11, 8],
                                [13, 14, 15, 12]])
    # print(goal_board)
    # print(simple_board)

    fringe1 = []
    node1 = State.State(simple_board, None, 0, 0)
    expand_fringe(node1, fringe1)
    assert State.State(simple_board.slide_blank((-1, 0)), node1, 0, 0) not in fringe1
    assert State.State(simple_board.slide_blank((0, -1)), node1, 0, 1) in fringe1

    # Simple test case for breadth_first_search
    fringe1 = []
    node1 = State.State(simple_board, None, 0, 0)
    expand_fringe(node1, fringe1)
    assert breadth_first_search(fringe1, 3, goal_board) == CONTINUE
    fringe1[0] = State.State(goal_board, node1, 0, 0)
    assert type(breadth_first_search(fringe1, 3, goal_board)) is State.State

    # Simple test case for ucs_f_function
    node1 = State.State(simple_board, None, 0, 0)
    assert ucs_f_function(node1.board, 0) == 0

    # Simple test case for a_star_f_function
    # -> This ONLY checks that the return type is correct
    assert hasattr(a_star_f_function_factory(None, goal_board), '__call__')

    # This section is for you to create tests for your own heuristic


    # Simple test for Informed Expansion
    node1 = State.State(simple_board, None, 0, 0)
    fringe1 = []
    informed_expansion(node1, fringe1, ucs_f_function)
    assert State.State(simple_board.slide_blank((-1, 0)), node1, 0, 0) not in fringe1
    assert State.State(simple_board.slide_blank((0, -1)), node1, 0, 1) in fringe1

    # Simple test for Informed Search
    fringe1 = []
    explored = {}
    node1 = State.State(simple_board, None, 0, 0)
    expand_fringe(node1, fringe1)
    assert informed_search(fringe1, goal_board, ucs_f_function, explored) == CONTINUE
    fringe1[0] = State.State(goal_board, node1, 0, 0)
    assert type(informed_search(fringe1, goal_board, ucs_f_function, explored)) is State.State

    # Simple test for IDS
    node1 = State.State(simple_board, None, 0, 0)
    #assert ids(node1.board, goal_board, 1) == None
    #result = ids(node1.board, goal_board, 4)
    #assert type(result) is Board.Board


if __name__ == "__main__":
    main()
