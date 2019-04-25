import tkinter as tk
import time
import argparse
import glob, os
import numpy as np
from utils import *
import copy

from ThaiCheckers.preprocessing import index_to_move, move_to_index, index_to_move_human
from ThaiCheckers.ThaiCheckersGame import ThaiCheckersGame as Game, display
from ThaiCheckers.ThaiCheckersPlayers import minimaxAI
from ThaiCheckers.pytorch.NNet import NNetWrapper as nn
from MCTS_th_checkers import MCTS

# Constants
ENGINE_1 = 1
ENGINE_2 = -1
PAUSE = 0
LOOP_ACTIVE = True

# Argument parsers
parser = argparse.ArgumentParser('Bot select')
parser.add_argument('--player1', dest='player1', type=str)
parser.add_argument('--player2', dest='player2', type=str)

args = parser.parse_args()

checkers = Game()
board = checkers.getInitBoard()
DEPTH = 7

if args.player1 == 'minimax':
    AI_1 = minimaxAI(checkers,depth=DEPTH,verbose=True)

else:
    nnet = nn(checkers, gpu_num=0)
    nnet.load_checkpoint(folder='models_minimax', filename='train_iter_268.pth.tar')
    args1 = dotdict({'numMCTSSims':200, 'cpuct': 1.0})
    AI_1 = MCTS(checkers, nnet, args1, eval=True, verbose=True)

if args.player2 == 'minimax':
    AI_2 = minimaxAI(checkers,depth=DEPTH,verbose=True)
    print("minimax")
else:
    nnet2 = nn(checkers, gpu_num=0)
    nnet2.load_checkpoint(folder='models_minimax', filename='train_iter_268.pth.tar')
    args2 = dotdict({'numMCTSSims':200, 'cpuct': 1.0})
    AI_2 = MCTS(checkers, nnet2, args2, eval=True, verbose=True) 
    


state = ENGINE_1
root = tk.Tk()
main_canvas = tk.Canvas(root, width=480, height = 480)

click_value = None
def draw_board(canvas):
    d = {0:'a',1:'b',2:'c',3:'d',4:'e',5:'f',6:'g',7:'h'}
    for row in range(8):
        for col in range(8):
            x = col * 60
            y = row * 60
            color = 'red'
            if (row + col) % 2 == 0:
                color = 'white'
            canvas.create_rectangle(x, y, x + 60, y + 60, fill=color)
            
            if(row==0):
                main_canvas.create_text(x+50,y+5,fill="darkblue",font="Helvetica 14 italic bold",text=d[col])

            if(col==0):
                main_canvas.create_text(x+5,y+50,fill="darkblue",font="Helvetica 14 italic bold",text=row+1)

def draw_pieces(canvas, checkers):
    for row in range(8):
        for col in range(8):
            x = col * 60
            y = row * 60 
            
            if board[row, col] == checkers.gameState.PLAYER_1:
                canvas.create_oval(x+10, y+10, x + 50, y + 50, fill='white', tags='piece')
            elif board[row, col] == checkers.gameState.PLAYER_2:
                canvas.create_oval(x+10, y+10, x + 50, y + 50, fill='black', tags='piece')

            elif board[row, col] == checkers.gameState.PLAYER_1_KING:
                canvas.create_oval(x+10, y+10, x + 50, y + 50, fill='white', tags='piece')
                canvas.create_oval(x + 15, y + 15, x + 45, y + 45, fill='yellow', tags='piece')
            elif board[row, col] == checkers.gameState.PLAYER_2_KING:
                canvas.create_oval(x+10, y+10, x + 50, y + 50, fill='black', tags='piece')
                canvas.create_oval(x + 15, y + 15, x + 45, y + 45, fill='yellow', tags='piece')

def draw_highlight(canvas, filtered_moves):
    for (_, end) in filtered_moves:
        y = end[0] * 60
        x = end[1] * 60
        canvas.create_rectangle(x, y, x+60, y+60, outline='yellow', width=5, tags='piece')
  
def update(canvas):
    global checkers
    canvas.delete('piece')
    draw_pieces(canvas, checkers)

def update_click_value(val):
    global click_value
    click_value = val

def onclick(event):
    x = int(event.x / 60)
    y = int(event.y / 60)
    click_value = (y, x)
    update_click_value(click_value)

def engine_1_move(board_input):

    global board
    global board_history
    valid_moves = checkers.getValidMoves(checkers.getCanonicalForm(board_input, state), 1)
    if np.sum(valid_moves)==1:
        time.sleep(0.2)
        #print(index_to_move_human(np.argmax(valid_moves)))
        board, _ = checkers.getNextState(board_input, ENGINE_1, np.argmax(valid_moves))
        board_history.append(copy.deepcopy(checkers))
        return
    if args.player1 == 'minimax':
        action = AI_1.get_move(checkers.getCanonicalForm(board_input, state))
    else:
        action = np.random.choice(32*32, p=AI_1.getActionProb((checkers.getCanonicalForm(board_input, state)), temp=0))

    if move_num < len(board_history)-1:
        board_history = board_history[:move_num+1]
    board, _ = checkers.getNextState(board_input, ENGINE_1, action)
    board_history.append(copy.deepcopy(checkers))
    display(board)
    
def engine_2_move(board_input):
    global board
    global board_history
    valid_moves = checkers.getValidMoves(checkers.getCanonicalForm(board_input, state), 1)
    if np.sum(valid_moves)==1:
        time.sleep(0.2)
        #print(index_to_move_human(np.argmax(valid_moves)))
        board, _ = checkers.getNextState(board_input, ENGINE_2, np.argmax(valid_moves))
        board_history.append(copy.deepcopy(checkers))
        return
    if args.player2 == 'minimax':
        action = AI_2.get_move(checkers.getCanonicalForm(board_input, state))
    else:
        action = np.random.choice(32*32, p=AI_2.getActionProb((checkers.getCanonicalForm(board_input, state)), temp=0))
        
    if move_num < len(board_history)-1:
        board_history = board_history[:move_num+1]
    board, _ = checkers.getNextState(board_input, ENGINE_2, action)
    board_history.append(copy.deepcopy(checkers))
    display(board)


def reset():
    global checkers
    global main_canvas
    global state
    global board
    global board_history
    global move_num
    global AI_1
    global AI_2
    turn_label['text'] = 'ENGINES ARE PLAYING!'
    checkers = Game()
    board = checkers.getInitBoard()
    board_history = [copy.deepcopy(checkers)]
    move_num = 0
    state = ENGINE_1
    update(main_canvas)
    if args.player1 != 'minimax':
        AI_1 = MCTS(checkers, nnet, args1, eval=True, verbose=True)
    if args.player2 != 'minimax':
        AI_2 = MCTS(checkers, nnet2, args2, eval=True, verbose=True) 


def previous():
    global checkers
    global board
    global move_num
    global board_history
    global cur_state
    global AI_1
    global AI_2
    if move_num>0:
        move_num -= 1
        checkers = copy.deepcopy(board_history[move_num])
        board = checkers.gameState.board
        update(main_canvas)
        cur_state = -cur_state
        #print('saved state:',cur_state)
    if args.player1 != 'minimax':
        AI_1.game = checkers
    if args.player1 != 'minimax':
        AI_2.game = checkers
    
    

def forward():
    global checkers
    global board
    global move_num
    global board_history
    global cur_state
    global AI_1
    global AI_2
    if move_num<len(board_history)-1:
        move_num += 1
        checkers = copy.deepcopy(board_history[move_num])
        board = checkers.gameState.board
        update(main_canvas)
        cur_state = -cur_state
        #print('saved state:',cur_state)
    if args.player1 != 'minimax':
        AI_1.game = checkers
    if args.player1 != 'minimax':
        AI_2.game = checkers
    

def pause():
    global state
    global cur_state
    turn_label['text'] = 'GAME PAUSED!'
    cur_state = state
    #print('saved state:',cur_state)
    state = PAUSE

def resume():
    global state
    global cur_state
    turn_label['text'] = 'ENGINES ARE PLAYING!'
    #print('Resume state:',cur_state)
    state = cur_state

main_canvas.bind('<Button-1>', onclick)
title = tk.Label(root, text='Engine vs Engine!', font=("Helvetica", 35))
title.pack()
turn_label = tk.Label(root, text='ENGINES ARE PLAYING!', font=("Helvetica", 30))
turn_label.pack()
main_canvas.pack()
reset_button = tk.Button(root, text='<<< Previous', command=previous, font=("Helvetica", 20))
reset_button.pack()
reset_button = tk.Button(root, text='Reset', command=reset, font=("Helvetica", 20))
reset_button.pack()
reset_button = tk.Button(root, text='Forward >>>', command=forward, font=("Helvetica", 20))
reset_button.pack()

reset_button = tk.Button(root, text='Pause', command=pause, font=("Helvetica", 20))
reset_button.pack()
reset_button = tk.Button(root, text='Resume', command=resume, font=("Helvetica", 20))
reset_button.pack()


draw_board(main_canvas)
draw_pieces(main_canvas, checkers)

board_history = [copy.deepcopy(checkers)]
move_num = 0

cur_state = state



while LOOP_ACTIVE:
    root.update()
    if checkers.getGameEnded(board,state) != 0:
        winner = checkers.getGameEnded(board,state)
        if winner==-1:
            turn_label['text'] = 'ENGINE 2 WON!'
        elif winner==1:
            turn_label['text'] = 'ENGINE 1 WON!'
        else:
            turn_label['text'] = 'DRAW!'
        #print('Winner is:',winner)
        continue
    if state == ENGINE_1:
        engine_1_move(board)
        move_num += 1
        update(main_canvas)
        state = ENGINE_2
        
    elif state == ENGINE_2:
        engine_2_move(board)
        move_num += 1
        update(main_canvas)
        state = ENGINE_1
    else:
        continue
    
    # Interval between timesteps
    time.sleep(0.2)


