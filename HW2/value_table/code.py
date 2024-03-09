import argparse
import numpy as np
import random
import time

from keras.layers import Dense, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import Adam

# constant
probabilities = np.array([1/2] + [1/16]*8)

# four_consecutive checks if there are 4 consecutive 1s or -1s in an array
def four_consecutive(arr):
    for i in range(len(arr)-3):
        four_sum = np.sum(arr[i:i+4])
        if four_sum == 4:
            return 1
        if four_sum == -4:
            return -1
    return 0

class Board():
    def __init__(self, board):
        self.board = board
    
    def copy(self):
        return Board(self.board.copy())
    
    def get_available_actions(self):
        empty_grids = np.where(self.board==0)
        return empty_grids[0] * 6 + empty_grids[1]

    def get_winner(self):
        # check rows and columns
        for i in range(6):
            row_consecutive = four_consecutive(self.board[i,:])
            if row_consecutive:
                return row_consecutive
            column_consecutive = four_consecutive(self.board[:,i])
            if column_consecutive:
                return column_consecutive
        # check diagonals
        for i in range(-2,3):
            flipped_board = np.fliplr(self.board)
            left_down_consecutive = four_consecutive(np.diag(self.board, k=i))
            if left_down_consecutive:
                return left_down_consecutive
            left_up_consecutive = four_consecutive(np.diag(flipped_board, k=i))
            if left_up_consecutive:
                return left_up_consecutive
        return 0
    
    def is_full(self):
        return np.sum(self.board == 0) == 0
    
    def get_next_steps(self, action, color):
        all_possible_boards = []
        for shift in [(0,0),(1,1),(1,0),(0,1),(-1,1),(1,-1),(-1,-1),(-1,0),(0,-1)]:
            new_board = self.board.copy()
            row, column = divmod(action, 6)
            row += shift[0]
            column += shift[1]
            if 0 <= row < 6 and 0 <= column < 6 and new_board[row][column] == 0:
                new_board[row][column] = color
            all_possible_boards.append(new_board)
        return np.array(all_possible_boards) # The first one is 1/2 and others are 1/16
    
    def step(self, action, color, display):
        row, column = divmod(action, 6)
        shift = (0, 0)
        if random.random() < 0.5:
            shift = random.choice([(1,1),(1,0),(0,1),(-1,1),(1,-1),(-1,-1),(-1,0),(0,-1)])
        row += shift[0]
        column += shift[1]
        if 0 <= row < 6 and 0 <= column < 6 and self.board[row][column] == 0:
            self.board[row][column] = color
        if not display:
            return
        print(f'=======player{1 if color == 1 else 2}({"X" if color == 1 else "O"}) action={action}')
        for row in self.board:
            for cell in row:
                if cell == 1:
                    print('X', end=' ')
                elif cell == -1:
                    print('O', end=' ')
                else:
                    print('-', end=' ')
            print()

def tic_tac_toe(player1, player2, display = False):
    states = []
    greedy = []
    board = Board(np.zeros((6,6)))
    current_player = player1
    current_color = 1
    while True:
        current_action, current_greedy = current_player(board)
        states.append(board.copy())
        greedy.append(current_greedy)
        board.step(current_action, current_color, display)
        winner, full = board.get_winner(), board.is_full()
        if winner or full:
            return winner, states + [board], greedy
        current_player = player2 if current_color == 1 else player1
        current_color = -current_color

def random_player(board):
    return random.choice(board.get_available_actions()), False

def human_player(board):
    user_input = input("Please enter a number: ")
    return int(user_input), False

def new_keras_model(layers, learning_rate):
    model = Sequential()
    model.add(Flatten(input_shape=(6, 6)))
    for layer_size in layers:
        model.add(Dense(layer_size, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

class AIPlayer():
    def __init__(self, color, layers, network_learning_rate, td_learning_rate, model_path=None):
        self.color = color
        self.value_net = load_model(model_path) if model_path else new_keras_model(layers, network_learning_rate)
        self.td_learning_rate = td_learning_rate
    
    def get_player(self, explore_rate):
        def player(board):
            available_actions = board.get_available_actions()
            if random.random() < explore_rate:
                return random.choice(available_actions), False
            else:
                predictions = self.value_net(np.concatenate([board.get_next_steps(action,self.color) for action in available_actions]))
                action_scores = [(available_actions[i], np.sum(predictions[i*9:(i+1)*9,0] * probabilities)) for i in range(len(available_actions))]
                random.shuffle(action_scores)
                sorted_scores = sorted(action_scores, key=lambda x: x[1], reverse=True)
                return sorted_scores[0][0], True
        return player
    
    def learn_from_experience(self, winner, states, greedy):
        all_boards = np.array([b.board for b in states])
        board_values = self.value_net(all_boards).numpy()
        board_values[-1] = winner * self.color
        i = len(greedy) - 1
        while i >= 0:
            if greedy:
                board_values[i] += self.td_learning_rate * (board_values[i+1] - board_values[i])
            i -= 1
        # can learn 8x faster since status value of should be same after rotate and flip left right
        rots = [all_boards, np.rot90(all_boards, k=1, axes=(1,2)), np.rot90(all_boards, k=2, axes=(1,2)), np.rot90(all_boards, k=-1, axes=(1,2))]
        Xs = np.concatenate(rots+[np.fliplr(r) for r in rots])
        ys = np.concatenate([board_values]*8)
        self.value_net.fit(Xs, ys, verbose=0, epochs=5)
    
    def save_model(self, path):
        self.value_net.save(path)

def estimate_models(player1, player2, num_trails):
    wins = [0, 0, 0]
    for _ in range(num_trails):
        result,_,_ = tic_tac_toe(random_player, random_player)
        wins[int(result)+1] += 1
    print(f'random player VS random player: win {wins[2]}, tie {wins[1]}, loss {wins[0]}')

    wins = [0, 0, 0]
    for _ in range(num_trails):
        result,_,_ = tic_tac_toe(player1, random_player)
        wins[int(result)+1] += 1
    print(f'player1 VS random player: win {wins[2]}, tie {wins[1]}, loss {wins[0]}')

    wins = [0, 0, 0]
    for _ in range(num_trails):
        result,_,_ = tic_tac_toe(random_player, player2)
        wins[int(result)+1] += 1
    print(f'random player VS player2: win {wins[2]}, tie {wins[1]}, loss {wins[0]}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Assignment 2.')
    parser.add_argument('--init_explore_rate', type=float, default=0.5, help='Initial exploration rate')
    parser.add_argument('--last_explore_rate', type=float, default=0.2, help='Last exploration rate')
    parser.add_argument('--change_explore_at', type=int, default=15000, help='Change exploration rate from this epoch on')
    parser.add_argument('--num_epochs', type=int, default=40000, help='Number of epochs')
    parser.add_argument('--model_source', type=str, choices=['train', 'load'], default='train', help="Source of the model ('train' or 'load')")
    parser.add_argument('--model_save_paths', type=str, default='agent1,agent2', help='Comma-separated paths to save trained models')
    parser.add_argument('--layers', type=lambda s: tuple(map(int, s.split(','))), default=(128, 128), help='Comma-separated numbers representing the sizes of layers')
    parser.add_argument('--network_learning_rate', type=float, default=0.001, help='Learning rate for neural network')
    parser.add_argument('--td_learning_rate', type=float, default=0.01, help='Update speed for TD learning')
    parser.add_argument('--test_times', type=int, default=1000, help='Number of trails for testing.')
    args = parser.parse_args()

    print(f'Arguments parsed: {args}')
    begin_time = time.time()
    [model_path_1, model_path_2] = args.model_save_paths.split(',')

    if args.model_source == 'train':
        agent1 = AIPlayer(1, args.layers, args.network_learning_rate, args.td_learning_rate)
        agent2 = AIPlayer(-1, args.layers, args.network_learning_rate, args.td_learning_rate)
        for i in range(args.num_epochs):
            if (i+1) % 250 == 0:
                print(f'Iteration {i+1} done. it takes {(time.time() - begin_time):.2f} seconds.')
            if (i+1) % 1000 == 0:
                estimate_models(agent1.get_player(0), agent2.get_player(0), 100)
            winner, states, greedy = tic_tac_toe(agent1.get_player(args.init_explore_rate if i<args.change_explore_at else args.last_explore_rate), agent2.get_player(args.init_explore_rate if i<args.change_explore_at else args.last_explore_rate))
            agent1.learn_from_experience(winner, states, greedy)
            agent2.learn_from_experience(winner, states, greedy)
        print(f'Training done, it takes {(time.time() - begin_time):.2f} seconds.')
        agent1.save_model(model_path_1)
        agent2.save_model(model_path_2)
    else:
        agent1 = AIPlayer(1, args.layers, args.network_learning_rate, args.td_learning_rate, model_path=model_path_1)
        agent2 = AIPlayer(-1, args.layers, args.network_learning_rate, args.td_learning_rate, model_path=model_path_2)
        print(f'Agents loaded, it takes {(time.time() - begin_time):.2f} seconds.')

    # compete with random policy
    estimate_models(agent1.get_player(0), agent2.get_player(0), args.test_times)
    # compete with human
    winner, states, greedy = tic_tac_toe(agent1.get_player(0), human_player, True)
    winner, states, greedy = tic_tac_toe(human_player, agent2.get_player(0), True)