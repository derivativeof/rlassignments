import argparse
import numpy as np
import random
import time
import keras
from keras.layers import Dense, Flatten

import matplotlib.pyplot as plt

# constant
ACTION_INDEX = np.array(range(36), dtype=np.float64).reshape((6,6))
PROBABILITIES = np.array([1/2] + [1/16]*8)
EMA_WEIGHT = 0.1

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

class ReplayBuff():

    def __init__(self, size):
        self.data = [None] * size
        self.size = size
        self.count = 0
        self.insert_index = 0
    
    def insert(self, data):
        self.data[self.insert_index] = data
        self.insert_index = (self.insert_index+1) % self.size
        if self.count < self.size:
            self.count += 1
    
    def sample(self, n):
        return random.sample(self.data[:self.count], min(n, self.count))

def super_tic_tac_toe(player1, player2, display = False):
    states = []
    actions = []
    board = Board(np.zeros((6,6)))
    current_player = player1
    current_color = 1
    while True:
        current_action = current_player(board)
        states.append(board.copy())
        actions.append(current_action)
        board.step(current_action, current_color, display)
        winner, full = board.get_winner(), board.is_full()
        if winner or full:
            return winner, states + [board.copy()], actions
        current_player = player2 if current_color == 1 else player1
        current_color = -current_color

def random_player(board):
    return np.random.choice(board.get_available_actions())

def human_player(board):
    user_input = input("Please enter a number: ")
    return int(user_input)

def get_network():
    model = keras.Sequential()
    model.add(Flatten(input_shape=(6, 6)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(36, activation='tanh'))
    model.compile(optimizer='sgd', loss='mse')
    return model

class DqnPlayer():

    def __init__(self, policy_path="", target_path=""):
        if policy_path and target_path:
            self.policy = keras.models.load_model(policy_path)
            self.target = keras.models.load_model(target_path)
        else:
            self.policy = get_network()
            self.target = get_network()

    def get_player(self, color, explore_rate):
        def player(board):
            available_actions = board.get_available_actions()
            if random.random() < explore_rate:
                return random.choice(available_actions)
            else:
                preds = self.policy(np.array([board.board * color])).numpy()[0]
                action_scores = [(available_actions[i], preds[available_actions[i]]) for i in range(len(available_actions))]
                random.shuffle(action_scores)
                sorted_scores = sorted(action_scores, key=lambda x: x[1], reverse=True)
                return sorted_scores[0][0]
        return player

    def learn(self, replay_buff, batch_size=1024, n_times = 2):
        for _ in range(n_times):
            mini_batch = replay_buff.sample(batch_size)
            cur_boards = np.array([data['cur_board'] for data in mini_batch])
            cur_board_masks = np.array([data['cur_board_mask'] for data in mini_batch])
            next_boards = np.array([data['next_board'] for data in mini_batch])
            next_board_masks = np.array([data['next_board_mask'] for data in mini_batch])
            next_preds_target = self.target(next_boards).numpy()
            next_preds_policy = self.policy(next_boards).numpy()
            masked_next_preds_target = next_preds_target * next_board_masks + (next_board_masks - 1.0)
            masked_next_preds_policy = next_preds_policy * next_board_masks + (next_board_masks - 1.0)
            is_last_steps = np.array([data['is_last_step'] for data in mini_batch])
            rewards = np.array([data['reward'] for data in mini_batch])
            choosen_policy = np.argmax(masked_next_preds_policy, 1)
            target_vals = masked_next_preds_target[np.arange(masked_next_preds_target.shape[0]), choosen_policy] * (1 - is_last_steps) + rewards * is_last_steps
            cur_pred = self.policy(cur_boards).numpy()
            train_y = cur_pred * cur_board_masks + (cur_board_masks - 1.0)
            actions = np.array([data['action'] for data in mini_batch])
            train_y[range(len(mini_batch)),actions] = target_vals
            self.policy.fit(cur_boards, train_y, epochs=1, verbose=0)
            # exp moving average update
            policy_weights = self.policy.get_weights()
            target_weights = self.target.get_weights()
            ema_weights = [policy_weights[i] * EMA_WEIGHT + target_weights[i] * (1-EMA_WEIGHT) for i in range(len(policy_weights))]
            self.target.set_weights(ema_weights)

def board_expand(board):
    roated_boards = [np.rot90(board, k=i) for i in range(4)]
    flipped_boards = [np.fliplr(b) for b in roated_boards]
    return roated_boards + flipped_boards

def line_expand(line):
    board = line.reshape((6,6))
    expand_boards = board_expand(board)
    return [board.reshape(36) for board in expand_boards]

def point_expand(point):
    board = np.zeros((6,6))
    board[divmod(point,6)] = 1
    expand_boards = board_expand(board)
    return [int(np.sum(board*ACTION_INDEX)) for board in expand_boards]

def ts_expand(ts):
    cur_boards = board_expand(ts['cur_board'])
    next_boards = board_expand(ts['next_board'])
    actions = point_expand(ts['action'])
    cur_board_masks = line_expand(ts['cur_board_mask'])
    next_board_masks = line_expand(ts['next_board_mask'])
    return [dict(cur_board=cur_boards[i], next_board=next_boards[i], action=actions[i], cur_board_mask=cur_board_masks[i], next_board_mask=next_board_masks[i], reward=ts['reward'], is_last_step=ts['is_last_step']) for i in range(8)]

def insert_experience(replay_buff, winner, states, actions):
    len_actions = len(actions)
    avails = [1.0 * (state.board.reshape(36) == 0) for state in states]
    for i in range(len_actions):
        color = 1 if i%2 == 0 else -1
        cur_board = states[i].board * color
        cur_board_mask = avails[i]
        next_index = min(i+2, len_actions)
        next_board = states[next_index].board * color
        next_board_mask = avails[next_index]
        action = actions[i]
        reward = 0 if i<len_actions-2 else color * winner
        is_last_step = 0 if i<len_actions-2 else 1
        ts = dict(cur_board=cur_board, next_board=next_board, action=action, cur_board_mask=cur_board_mask, next_board_mask=next_board_mask, reward=reward, is_last_step=is_last_step)
        expanded_ts = ts_expand(ts)
        for ets in expanded_ts:
            replay_buff.insert(ets)

def estimate_models(player1, player2, num_trails):
    wins = [0, 0, 0]
    for _ in range(num_trails):
        result,_,_ = super_tic_tac_toe(player1, random_player)
        wins[int(result)+1] += 1

    wins2 = [0, 0, 0]
    for _ in range(num_trails):
        result,_,_ = super_tic_tac_toe(random_player, player2)
        wins2[int(result)+1] += 1

    win_rate_first = wins[2]/(wins[0]+wins[2])*100
    win_rate_second = wins2[0]/(wins2[0]+wins2[2])*100
    print(f'Win rate as first player:{win_rate_first:.2f}%, as second player:{win_rate_second:.2f}%')
    return win_rate_first, win_rate_second

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Assignment 2 (new DQN version)')
    parser.add_argument('--init_explore_rate', type=float, default=1.0, help='Initial exploration rate')
    parser.add_argument('--min_explore_rate', type=float, default=0.1, help='Last exploration rate')
    parser.add_argument('--explore_decrease', type=float, default=0.999_995, help='Change exploration rate from this epoch on')
    parser.add_argument('--num_epochs', type=int, default=500_000, help='Number of epochs')
    parser.add_argument('--replay_buff_size', type=int, default=500_000, help='Replay buff size.')
    parser.add_argument('--timer_interval', type=int, default=500, help='Time interval for display number of epoches and time taken')
    parser.add_argument('--eval_interval', type=int, default=2_000, help='Model evalution interval.')
    parser.add_argument('--model_source', type=str, choices=['train', 'load'], default='train', help="Source of the model ('train' or 'load')")
    parser.add_argument('--policy_save_path', type=str, default='ddqn_0_policy', help='Path to save trained policy model')
    parser.add_argument('--target_save_path', type=str, default='ddqn_0_target', help='Path to save trained target model')
    parser.add_argument('--curve_save_path', type=str, default='ddqn_0_curve.png', help='Path to save converge curve')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size for training.')
    parser.add_argument('--train_times', type=int, default=1, help='Number of mini batch to train in each epoch')
    parser.add_argument('--middle_test_times', type=int, default=100, help='Number of trails for testing.')
    parser.add_argument('--final_test_times', type=int, default=10_000, help='Number of trails for testing.')
    args = parser.parse_args()

    if args.model_source == 'train':
        agent = DqnPlayer()
        rb = ReplayBuff(args.replay_buff_size)
        begin_time = time.time()
        explore_rate = args.init_explore_rate
        iters, win1s, win2s = [], [], []
        for i in range(args.num_epochs):
            if explore_rate > args.min_explore_rate:
                explore_rate *= args.explore_decrease
            winner, states, actions = super_tic_tac_toe(agent.get_player(1, explore_rate), agent.get_player(-1, explore_rate), False)
            insert_experience(rb, winner, states, actions)
            agent.learn(rb, args.batch_size, args.train_times)
            if (i+1) % args.timer_interval == 0:
                print(f'Iteration {i+1} done. Current Explore rate is {explore_rate:.4f}. It takes {time.time() - begin_time:.2f} seconds')
            if (i+1) % args.eval_interval == 0:
                win1, win2 = estimate_models(agent.get_player(1, 0), agent.get_player(-1, 0), args.middle_test_times)
                iters.append(i+1)
                win1s.append(win1)
                win2s.append(win2)
        agent.policy.save(args.policy_save_path)
        agent.target.save(args.target_save_path)
        print(f'Training done, it takes {time.time() - begin_time: .2f} seconds. Model saved to {args.policy_save_path} and {args.target_save_path}')
        # plot converge rates
        plt.plot(iters, win1s, label='first player')
        plt.plot(iters, win2s, label='second player')
        plt.xlabel('num epoches')
        plt.ylabel('win rate(%)')
        plt.title('Model converge rates')
        plt.legend()
        plt.savefig(args.curve_save_path)
        print(f'Converge curve saved to {args.curve_save_path}.')
    else:
        agent = DqnPlayer(args.policy_save_path, args.target_save_path)

    # estimate trained agent
    print(f'Testing model for {args.final_test_times} times.')
    estimate_models(agent.get_player(1, 0), agent.get_player(-1, 0), args.final_test_times)