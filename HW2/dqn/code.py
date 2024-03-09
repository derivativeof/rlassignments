from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import load_model

import argparse
import numpy as np
import random

import numpy as np

# four_consecutive checks if there are 4 consecutive 1s or -1s in an array
def four_consecutive(arr):
    for i in range(len(arr)-3):
        four_sum = np.sum(arr[i:i+4])
        if four_sum == 4:
            return 1
        if four_sum == -4:
            return -1
    return 0

class SuperTicTacToe():

    def __init__(self, player1, player2):
        self.player1 = player1 # use 1 (for X in display)
        self.player2 = player2 # player 2 always use -1 (for O in display)
    
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
        print(f'=======player{1 if color == 1 else 2}({"X" if color == 1 else "O"}) action={action}, effective row={row} column={column}')
        for row in self.board:
            for cell in row:
                if cell == 1:
                    print('X', end=' ')
                elif cell == -1:
                    print('O', end=' ')
                else:
                    print('-', end=' ')
            print()
    
    def is_full(self):
        return np.sum(np.abs(self.board)) == 36
    
    def play(self, display = False):
        records = []
        self.board = np.zeros((6,6))
        while True:
            player1_action = self.player1(self.board)
            records.append([player1_action, self.board.copy()])
            self.step(player1_action, 1, display)
            if self.get_winner() == 1:
                return 1.0, records, self.board.copy()
            if self.is_full():
                return 0.0, records, self.board.copy()
            player2_action = self.player2(self.board)
            records.append([player2_action, self.board.copy()])
            self.step(player2_action, -1, display)
            if self.get_winner() == -1:
                return -1.0, records, self.board.copy()
            if self.is_full():
                return 0.0, records, self.board.copy()

def new_keras_model(layers=(128, 64, 64), learning_rate=0.001):
    model = Sequential()
    model.add(Flatten(input_shape=(6, 6)))
    for layer_size in layers:
        model.add(Dense(layer_size, activation='relu'))
    model.add(Dense(36, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

def random_player(_):
    return random.randint(0, 35)

def human_player(_):
    user_input = input("Please enter a number: ")
    return int(user_input)

def player_with_keras_model(model, explore_rate):
    def model_player(board):
        if random.random() < explore_rate:
            return random.randint(0,35)
        return np.argmax(model(np.array([board]))[0])
    return model_player

def get_exp_from_records(model1, model2, winner, records):
    num_rounds, num_last = divmod(len(records) ,2)

    X1 = np.array([records[2*i][1] for i in range(num_rounds+num_last)])
    y1 = model1(X1).numpy()
    for i in range(num_rounds+num_last-1):
        y1[i][records[2*i][0]] = np.max(y1[i+1])
    y1[-1][records[2*(num_rounds+num_last-1)][0]] = winner

    X2 = np.array([records[2*i+1][1] for i in range(num_rounds)])
    y2 = model2(X2).numpy()
    for i in range(num_rounds-1):
        y2[i][records[2*i+1][0]] = np.max(y2[i+1])
    y2[-1][records[2*(num_rounds-1)+1][0]] = -winner
    
    return [X1, y1], [X2, y2]

def estimate_models(model1, model2, num_trails):
    wins = [0, 0, 0]
    for _ in range(num_trails):
        result,_,_ = SuperTicTacToe(random_player, random_player).play(False)
        wins[int(result)+1] += 1
    print(f'random player VS random player: win {wins[2]}, tie {wins[1]}, loss {wins[0]}')

    wins = [0, 0, 0]
    for _ in range(num_trails):
        result,_,_ = SuperTicTacToe(player_with_keras_model(model1, 0), random_player).play(False)
        wins[int(result)+1] += 1
    print(f'model1 VS random player: win {wins[2]}, tie {wins[1]}, loss {wins[0]}')

    wins = [0, 0, 0]
    for _ in range(num_trails):
        result,_,_ = SuperTicTacToe(random_player, player_with_keras_model(model2, 0)).play(False)
        wins[int(result)+1] += 1
    print(f'random player VS model2: win {wins[2]}, tie {wins[1]}, loss {wins[0]}. (Noted that "loss" means random player loss (model2 win))')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--init_explore_rate', type=float, default=1.0, help='Initial exploration rate')
    parser.add_argument('--least_explore_rate', type=float, default=0.05, help='Minimum exploration rate')
    parser.add_argument('--explore_rate_decrease', type=float, default=0.9999, help='Exploration rate decrease factor')
    parser.add_argument('--num_epochs', type=int, default=50000, help='Number of epochs')
    parser.add_argument('--model_source', type=str, choices=['train', 'load'], default='train', help="Source of the model ('train' or 'load')")
    parser.add_argument('--model_save_paths', type=str, default='model1_demo,model2_demo', help='Comma-separated paths to save models')
    parser.add_argument('--layers', type=lambda s: tuple(map(int, s.split(','))), default=(128, 64, 64), help='Comma-separated numbers representing the sizes of layers (e.g., 128,64,64)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    model1_path, model2_path = args.model_save_paths.split(',')
    if args.model_source == 'train':
        model1 = new_keras_model(args.layers, args.learning_rate)
        model2 = new_keras_model(args.layers, args.learning_rate)
        current_explore = args.init_explore_rate
        n = 0
        for _ in range(args.num_epochs):
            n += 1
            if current_explore > args.least_explore_rate:
                current_explore *= args.explore_rate_decrease
            winner, records, _ = SuperTicTacToe(player_with_keras_model(model1, current_explore), player_with_keras_model(model2, current_explore)).play(False)
            exp1, exp2 = get_exp_from_records(model1, model2, winner, records)
            # Now this can be used to train the model
            model1.fit(exp1[0], exp1[1], verbose=0)
            model2.fit(exp2[0], exp2[1], verbose=0)
            if (n)%100==0:
                print(f'epoch {n} done: explore rate = {current_explore}')
            if (n)%2000==0:
                estimate_models(model1, model2, 100)
        model1.save(model1_path)
        model2.save(model2_path)
    else:
        model1 = load_model(model1_path)
        model2 = load_model(model2_path)

    # estimate fully trained model
    estimate_models(model1, model2, 1000)
    # human trail
    def debug_model1_player(board):
        predictions = model1(np.array([board]))
        print(f'predictions={predictions}')
        return np.argmax(predictions)
    winner, record, last = SuperTicTacToe(debug_model1_player, human_player).play(True)

    def debug_model2_player(board):
        predictions = model2(np.array([board]))
        print(f'predictions={predictions}')
        return np.argmax(predictions)
    winner, record, last = SuperTicTacToe(human_player, debug_model2_player).play(True)