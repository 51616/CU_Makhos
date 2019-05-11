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



# Argument parsers
parser = argparse.ArgumentParser('Bot select')
parser.add_argument('--type', '-t', nargs='?', dest='type', type=str)
parser.add_argument('--player2', nargs='?', dest='player2', type=bool)
parser.add_argument('--hint', nargs='?', dest='hint', type=bool, default=False)
parser.add_argument('--depth', nargs='?', dest='depth', type=int, default = 5)
parser.add_argument('--mcts', nargs='?', dest='mcts', type=int, default = 100)
args = parser.parse_args()

# Constants
PLAYER_SELECT_START = 0
PLAYER_SELECT_END = 1
BOT_SELECT = -1
LOOP_ACTIVE = True

if args.player2:
    state = BOT_SELECT
else:
    state = PLAYER_SELECT_START

checkers = Game()
board = checkers.getInitBoard()


if args.type == 'minimax':
    AI = minimaxAI(checkers, depth=args.depth,verbose=True)
    print("minimax")

else:
    print('Neural network model')
    nnet = nn(checkers, gpu_num=0)
    nnet.load_checkpoint(folder='models_minimax', filename='train_iter_268.pth.tar')
    args1 = dotdict({'numMCTSSims':args.mcts, 'cpuct': 1.0})
    AI = MCTS(checkers, nnet, args1, eval=True, verbose=True)
    # def AI(x): return np.random.choice(
    #     32*32, p=mcts1.getActionProb(x, temp=0))

if args.hint:
    nnet_hint = nn(checkers, gpu_num=0)
    nnet_hint.load_checkpoint(folder='models_minimax', filename='train_iter_268.pth.tar')
    args_hint = dotdict({'numMCTSSims':args.mcts, 'cpuct': 1.0})
    AI_hint = MCTS(checkers, nnet_hint, args_hint, eval=True, verbose=True)

display(board)


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
            if args.player2:
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
            else:
                if board[row, col] == checkers.gameState.PLAYER_1:
                    canvas.create_oval(x+10, y+10, x + 50, y + 50, fill='black', tags='piece')
                elif board[row, col] == checkers.gameState.PLAYER_2:
                    canvas.create_oval(x+10, y+10, x + 50, y + 50, fill='white', tags='piece')

                elif board[row, col] == checkers.gameState.PLAYER_1_KING:
                    canvas.create_oval(x+10, y+10, x + 50, y + 50, fill='black', tags='piece')
                    canvas.create_oval(x + 15, y + 15, x + 45, y + 45, fill='yellow', tags='piece')
                elif board[row, col] == checkers.gameState.PLAYER_2_KING:
                    canvas.create_oval(x+10, y+10, x + 50, y + 50, fill='white', tags='piece')
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

def move_ai(board_input):
    print('Calculating...')
    #global state
    global board
    valid_moves = checkers.getValidMoves(checkers.getCanonicalForm(board_input, state), 1)
    if np.sum(valid_moves)==1:
        time.sleep(0.2)
        #print(index_to_move_human(np.argmax(valid_moves)))
        board, _ = checkers.getNextState(board_input, BOT_SELECT, np.argmax(valid_moves))
        board_history.append(copy.deepcopy(checkers))
        display(board)
        return
    if args.type == 'minimax':
        action = AI.get_move(checkers.getCanonicalForm(board_input, state))
    else:
        #action, pi, _, _ = AI.act(checkers.gameState, 0)
        action = np.random.choice(32*32, p=AI.getActionProb((checkers.getCanonicalForm(board_input, state)), temp=0))
        #print(action)
        #move = index_to_move(action)
    #checkers.step(move)
    board, _ = checkers.getNextState(board_input, BOT_SELECT, action)
    board_history.append(copy.deepcopy(checkers))
    display(board)

def hint(board_input):
    #action, pi, _, _ = AI.act(checkers.gameState, 0)
    probs = AI_hint.getActionProb((checkers.getCanonicalForm(board_input, PLAYER_SELECT_END)), temp=1)
    rec_moves = np.argsort(probs)[::-1][:5]
    print('Recommended moves:')
    for idx in rec_moves:
        if(probs[idx]>0):
            move = index_to_move_human(idx)

            print(''.join(np.array(move[0],dtype=str)) + ' to ' + ''.join(np.array(move[1],dtype=str)) + '  ' ,round(probs[idx],2))

    





def reset():
    global checkers
    global main_canvas
    global state
    global board
    global board_history
    global move_num
    global AI
    checkers = Game()
    board = checkers.getInitBoard()
    board_history = [copy.deepcopy(checkers)]
    move_num = 0
    if args.player2:
        state = BOT_SELECT
        turn_label['text'] = 'Opponent\'s turn!'
    else:
        state = PLAYER_SELECT_START
        turn_label['text'] = 'Your turn!'
    update(main_canvas)
    if args.type != 'minimax':
        AI = MCTS(checkers, nnet, args1, eval=True, verbose=True)


def previous():
    global checkers
    global board
    global move_num
    global board_history
    if move_num>0:
        move_num -= 1
        checkers = copy.deepcopy(board_history[move_num])
        board = checkers.gameState.board
        update(main_canvas)
    if args.type != 'minimax':
        AI.game = checkers

def forward():
    global checkers
    global board
    global move_num
    global board_history
    if move_num<len(board_history)-1:
        move_num += 1
        checkers = copy.deepcopy(board_history[move_num])
        board = checkers.gameState.board
        update(main_canvas)
    if args.type != 'minimax':
        AI.game = checkers


main_canvas.bind('<Button-1>', onclick)
title = tk.Label(root, text='CU Checkers', font=("Helvetica", 35))
title.pack()
turn_label = tk.Label(root, text='', font=("Helvetica", 30))
turn_label.pack()
if args.player2:
    turn_label['text'] = 'Opponent\'s turn!'
else:
    turn_label['text'] = 'Your turn!'
main_canvas.pack()
reset_button = tk.Button(root, text='<<< Previous', command=previous, font=("Helvetica", 20))
reset_button.pack()
reset_button = tk.Button(root, text='Reset', command=reset, font=("Helvetica", 20))
reset_button.pack()
reset_button = tk.Button(root, text='Forward >>>', command=forward, font=("Helvetica", 20))
reset_button.pack()

draw_board(main_canvas)
draw_pieces(main_canvas, checkers)
prev_state = BOT_SELECT

board_history = [copy.deepcopy(checkers)]
move_num = 0


while LOOP_ACTIVE:
    root.update()
    

    #print('state:',state)
    if (state == PLAYER_SELECT_START) & (checkers.getGameEnded(board,1) != 0) or (checkers.getGameEnded(board,state) != 0):
        if state == PLAYER_SELECT_START:
            winner = checkers.getGameEnded(board,1)
        else:
            winner = checkers.getGameEnded(board,state)
        if winner==-1:
            turn_label['text'] = 'YOU LOST!'
        elif winner==1:
            turn_label['text'] = 'YOU WON!'
        else:
            turn_label['text'] = 'DRAW!'
        #print('Winner is:',winner)
        #state = PLAYER_SELECT_START
        #state = 10 #pause the game

    elif state == PLAYER_SELECT_START:
        if (args.hint and prev_state == BOT_SELECT):
            hint(board)
        prev_state = PLAYER_SELECT_START
        turn_label['text'] = 'Your turn!'
        value = click_value
        if value != None:
            filtered_moves = []
            possible_moves = []
            possible_move_idx = checkers.getValidMoves(board, PLAYER_SELECT_END)
            for i,idx in enumerate(possible_move_idx):
                if (idx==1):
                    possible_moves.append(index_to_move(i))

            for move in possible_moves:
                if value == move[0]:
                    filtered_moves.append(move)
            if filtered_moves != []:
                starting_point = value
                state = PLAYER_SELECT_END
                draw_highlight(main_canvas, filtered_moves)
            click_value = None

    elif state == PLAYER_SELECT_END:
        value = click_value
        if value != None:
            end_point = value
            #print(end_point)
            possible_moves = []
            possible_move_idx = checkers.getValidMoves(board, PLAYER_SELECT_END)
            for i,idx in enumerate(possible_move_idx):
                if (idx==1):
                    possible_moves.append(index_to_move(i))
            if (starting_point, end_point) in possible_moves:

                if move_num < len(board_history)-1:
                    board_history = board_history[:move_num+1]

                board, _ = checkers.getNextState(board, PLAYER_SELECT_END, move_to_index((starting_point, end_point)))
                board_history.append(copy.deepcopy(checkers))
                move_num += 1
                turn_label['text'] = 'Opponent\'s turn'
                state = BOT_SELECT
            else:
                state = PLAYER_SELECT_START
            starting_point = None
            end_point = None
            click_value = None
            update(main_canvas)
    elif state == BOT_SELECT:
        move_ai(board)
        move_num += 1
        update(main_canvas)
        state = PLAYER_SELECT_START
        prev_state = BOT_SELECT
    
    # Interval between timesteps
    #time.sleep(0.2)


