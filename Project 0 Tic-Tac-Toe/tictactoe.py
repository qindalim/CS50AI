"""
Tic Tac Toe Player
"""

from ctypes import util
from curses.ascii import EM
import math
from operator import ilshift
from queue import Empty

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    num_x = 0
    num_o = 0

    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == X:
                num_x += 1
            elif board[i][j] == O:
                num_o += 1
    
    if num_x > num_o:
        return O
    else:
        return X

def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    all_possible_actions = set()

    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == EMPTY:
                all_possible_actions.add((i,j))
    
    return all_possible_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    new_board = initial_state()

    for i in range(len(board)):
        for j in range(len(board[0])):
            new_board[i][j] = board[i][j]
    
    new_board[action[0]][action[1]] = player(board)

    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    if board[0][0] == board[1][1] == board[2][2]:
        return board[0][0]
    elif board[2][0] == board[1][1] == board[0][2]:
        return board[2][0]

    for i in range(len(board)):
        if board[i][0] == board[i][1] == board[i][2]:
            return board[i][0]
    
    for i in range(len(board[0])):
        if board[0][i] == board[1][i] == board[2][i]:
            return board[0][i]
    
    return EMPTY


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board):
        return True

    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == EMPTY:
                return False

    return True



def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return EMPTY 
    else:
        if player(board) == X:
            return max_value(board)[1]
        else:
            return min_value(board)[1]
    

def max_value(board): 
    if terminal(board):
        return utility(board), EMPTY
    
    final_v = float("-inf")
    final_action = EMPTY

    for action in actions(board):
        temp_v, temp_action = min_value(result(board, action))
        if temp_v > final_v:
            final_v = temp_v
            final_action = action
            if final_v == 1:
                return final_v, final_action
    
    return final_v, final_action

def min_value(board): 
    if terminal(board):
        return utility(board), EMPTY
    
    final_v = float("inf")
    final_action = EMPTY

    for action in actions(board):
        temp_v, temp_action = max_value(result(board, action))
        if temp_v < final_v:
            final_v = temp_v
            final_action = action
            if final_v == -1:
                return final_v, final_action
    
    return final_v, final_action