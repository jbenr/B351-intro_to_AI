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
    cur = State.State(current_state, None, 0, 0)
    b = current_state.board
    down = b.slide_blank((0, -1))
    left = b.slide_blank((1, 0))
    up = b.slide_blank((0, 1))
    right = b.slide_blank((-1, 0))
    if down is None:
        None
    else:
        fringe.append(State.State(down, cur, 0, 1))
    if up is None:
        None
    else:
        fringe.append(State.State(up, cur, 0, 1))
    if left is None:
        None
    else:
        fringe.append(State.State(left, cur, 0, 1))
    if right is None:
        None
    else:
        fringe.append(State.State(right, cur, 0, 1))
    

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
    else:
        b = fringe[0].board
        cur = fringe[0].parent_state
        dep = fringe[0].depth
        f = fringe[0].fvalue
        if dep > max_depth:
            fringe.pop()
            return breadth_first_search(fringe, max_depth, goal_board)
        elif b == goal_board:
            return fringe[0]
        else:
            fringe.pop()
            fringe.append(expand_fringe(State.State(b, cur, dep, f), fringe))
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
    l = len(board.matrix)
    #for i in range(0, l):
    #    goal_board.append([])
    #for i in range(0, l):
    #    for j in range(0, l):
    #        goal_board[j].append((i+1)+(l*j))
    #goal_board[l-1][l-1] = 0
    cur_zero = board.find_element(0)
    #goal_zero = goal_board.find_element(0)
    start_zero = (0, l-1)
    dep = abs(start_zero[0] - cur_zero[0]) + abs(start_zero[1] - cur_zero[1])
    return current_depth + dep


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
    return lambda heuristic, current_board: heuristic(current_board, goal_board)


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

def my_heuristic(current_board, goal_board):
    num = 0
    cur = current_board.matrix
    goal = goal_board.matrix
    l = len(cur)
    for i in range(1, l*l):
        g = goal_board.find_element(i)
        c = current_board.find_element(i)
        num += (abs(g[0] - c[0]) + abs(g[1] - c[1]))
    for col in range(0, l):
        for row in range(0, l):
            if cur[row][col] != goal[row][col]:
                num += 1
    return num
    

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
    cur = current_state.board
    valid_moves = [(-1,0), (0,1), (1,0), (0, -1)]
    for x in range(0, len(valid_moves)):
        if cur.slide_blank(valid_moves[x]) is not None:
            cur = cur.slide_blank(valid_moves[x])
            new = State.State(cur, current_state, current_state.depth, current_state.fvalue+1)
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
    if fringe is None:
        STOP
    else:
        top = heapq.heappop(fringe)
        for i in explored:
            if top == explored[i]:
                if top.fvalue > fringe[0].fvalue:
                    explored.add(top)
                    return CONTINUE
                    #informed_search(fringe, goal_board, f_function, explored)
        if top.board == goal_board:
            return top
        else:
            expand_fringe(top, fringe)
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


def ids(start_board, goal_board, final_depth):
    raise NotImplementedError
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
    my_board = Board.Board([[7, 3, 1],
                            [0, 6, 2],
                            [8, 5, 4]])
    assert my_heuristic(simple_board, goal_board) >= manhattan_distance(simple_board, goal_board)
    assert my_heuristic(my_board, goal_board) >= manhattan_distance(my_board, goal_board)

    # Simple test for Informed Expansion
    node1 = State.State(simple_board, None, 0, 0)
    fringe1 = []
    informed_expansion(node1, fringe1, ucs_f_function)
    assert State.State(simple_board.slide_blank((-1, 0)), node1, 0, 0) not in fringe1
    #assert State.State(simple_board.slide_blank((0, -1)), node1, 0, 1) in fringe1

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
    assert ids(node1.board, goal_board, 1) is None
    result = ids(node1.board, goal_board, 2)
    assert type(result) is Board.Board

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
    assert ids(node1.board, goal_board, 1) == None
    result = ids(node1.board, goal_board, 4)
    assert type(result) is Board.Board


if __name__ == "__main__":
    main()
