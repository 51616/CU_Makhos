import numpy as np

def preprocess_state(board):
	tensor = board.reshape(1, 1, 8, 8)
	return tensor

def flatten_idx(position, size, needhalf=True):
	(x, y) = position
	if needhalf:
		return x * size + y//2
	else:
		return x * size + y

def unflatten_idx(idx, size):
	start =  idx // size
	end = idx % size
	return start, end


def preprocess_move(move):
	(start, end) = move
	start_idx = flatten_idx(start, 4)
	end_idx = flatten_idx(end, 4)

	idx = flatten_idx((start_idx, end_idx), 32, False)
	mv_tensor = np.zeros((1, 32 * 32))
	mv_tensor[0][idx] = 1

	return mv_tensor

def move_to_index(move):
	(start, end) = move
	start_idx = flatten_idx(start, 4)
	end_idx = flatten_idx(end, 4)

	idx = flatten_idx((start_idx, end_idx), 32, False)

	return idx

def index_to_move(idx):
	start_idx, end_idx = unflatten_idx(idx, 32)
	start_x, start_y = unflatten_idx(start_idx, 4)
	if start_x % 2 == 0:
		start_y = start_y * 2 +1
	else:
		start_y = start_y * 2
	
	end_x, end_y = unflatten_idx(end_idx, 4)
	if end_x % 2 == 0:
		end_y = end_y * 2 +1
	else:
		end_y = end_y * 2

	return ((start_x, start_y),(end_x, end_y))