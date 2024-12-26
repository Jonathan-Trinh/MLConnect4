import random
import numpy as np
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from typing import Dict
import matplotlib.pyplot as plt
from collections import deque
import copy

class Player: 
    # Initialize the player object
    def __init__(self, name: str, turn: int, test: bool): 
        self.name = name
        self.turn = turn # Turn is -1 or 1 -> Red or Yellow
        self.test = test 

    # Optional error message 
    def move (self, board_arr) -> int:
        raise NotImplementedError("Implemented by child class.")
    
# Inherit attributes and methods for the RL Bot
class Bot(Player):
    def __init__(self, name, turn, test):
        super().__init__(name, turn, test)
    
    # The agent should make moves based on the current state
    def move(self, board_arr) -> int:
        # Checks for all not filled columns and returns T/F  EX: Given 3x3 matrix [2, 2, 3] < 3  Output: [True, True, False]
        # Board_arr.shape[0] gives the number of rows 
        # Board_arr.sum(axis = 0) gives us the sum of 1s and 0s along the columns since when a piece is placed the index will be set to 1
        available_cols = board_arr.sum(axis=0) < board_arr.shape[0] 
        # Filters out the available columns into a list for the bot to choose 
        # Board_arr.shape[0] generates the column indices and attaches a true false value from available columns 
        # If the the columns are true then add it to the list of final available column indicies
        # EX: [0: True, 1: True, 2: False] -> [0, 1]  Only columns 0 and 1 are available 
        available_cols = [c for c, i in zip(list(range(board_arr.shape[1])), available_cols) if i] 
        # Choose a random column
        col = random.choice(available_cols)
        return col
    
class RLBot(Player):
    def __init__(self, name, turn, test):
        super().__init__(name, turn, test)

        # Start the CNN f(x) = max(0,x) x > 0 returns x, x <= 0 returns 0
        self.activation = torch.nn.ReLU
        # Start the RL Agent and stores the Neural Network
        self.model = self.initialize_model()
        # Call the mean squared error function to calculate the loss between target and predicted q-vlaue optimizing for the minimum
        self.loss_fn = torch.nn.MSELoss()
        # Set the learning rate to 1 e-3 for gradual convergence and avoiding overestimation
        self.lr = 1e-3
        # Takes in the parameters(weights, biases) gathered by the model above adjusting them based on the gradient and 
        # takes in the learning rate to determine how large the steps are during the descent of the gradient
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr) 
        # The gamma (0 - 1) determines how much the agent cares about future rewards vs immediate rewards 
        # numbers closer to 1 like 0.9 will promote long term play
        self.gamma = 0.9
        # The epsilon determines rate at which the agent will try to explore new moves or choose known good ones
        # If the RLbot is the one being trained then it will will choose the best moves otherwise choose random moves half of the time
        self.epsilon = 0.0 if (self.turn!=-1 or self.test) else 0.5
        # The epsilon decay is the parameter that fine tunes the agent to choose more "good" moves after exploring  
        self.epsilon_decay = 0.0007

        # Early Stopping Flag
        self.stop_training = False 
        # Tracks the min loss since last min loss
        self.min_loss_dict = {'current_min': np.inf, 'num_steps': 0}
        # Num steps allowed before stopped
        self.patience = 200 
        self.losses = []
        self.rewards = []
        self.n_moves = 0
        #sqars: state_t, q_vals_t, action_t, reward_t, state_t+1
        self.current_sqars = [None, None, None, None, None]
        self.reward_vals = {
            'draw': 5,
            'win': 10,
            'loss': -10,
            'move': 0,
        }
    
    def initialize_model(self):
        input_n = 84
        input_n = 84
        hidden_n = 150
        hidden_n_2 = 100
        output_n = 7
        model = torch.nn.Sequential(
            torch.nn.Linear(input_n, hidden_n),
            self.activation(),
            torch.nn.Linear(hidden_n, hidden_n_2),
            self.activation(),
            torch.nn.Linear(hidden_n_2, output_n),
        )
        model.to(device)
        return model
    
    def get_state_array(self, piece_arrays: Dict):
        # Stack the tensors to get the board state
        state_array = np.stack([piece_arrays[self.turn], piece_arrays[-self.turn]])
        return state_array

    def process_state(self, curr_state: np.array):
        num_cells = curr_state[0].shape[0]*curr_state[0].shape[1]
        curr_state = curr_state.reshape(1, num_cells*2)
        if isinstance(self.activation, torch.nn.ReLU):
            curr_state += np.random.rand(1, num_cells*2)/10.0

        state = torch.from_numpy(curr_state).float().to(device)
        return state
    
    def move(self, piece_arrays: Dict) -> int:
        """
        @curr_state: 6x7x2 array, rows x cols x sides
        Sides represent friendly and enemy pieces.
        """
        curr_state = self.get_state_array(piece_arrays)
        state = self.process_state(curr_state)
        q_vals = self.model(state)
        q_vals_ = q_vals.data.cpu().numpy()
        
        board_arr = piece_arrays[self.turn] + piece_arrays[-self.turn]
        available_cols = board_arr.sum(axis=0) < board_arr.shape[0]
        # Exploration condition
        if random.random() < self.epsilon:
            action_ = random.choice([i for i,n in enumerate(available_cols) if n])
        # Exploitation condition
        else:
            action_ = np.argmax(q_vals_*available_cols.astype(int))
        self.epsilon *= (1-self.epsilon_decay)
        self.current_sqars[0] = state
        self.current_sqars[1] = q_vals
        self.current_sqars[2] = action_
        self.n_moves += 1
        return action_
    
    def get_reward(self, move_result):
        # No rewards for intermediate moves
        if move_result is None:
            reward = self.reward_vals['move']
        # Draws give a middle reward of 5
        elif 'draw' in move_result:
            reward = self.reward_vals['draw']
        # Give a reward of 10 for P1 Win -10 for P2 Win
        else:
            win_side = int(move_result.split('_')[-1])
            reward = self.reward_vals['win'] if win_side==self.turn else self.reward_vals['loss']
        self.current_sqars[3] = reward
        self.rewards.append(reward)
        return reward
    
    def reset_vars(self):
        self.current_sqars = [None, None, None, None, None]

    # Stop early when no significant improvement 
    def update_early_stopping(self):
        check_window_steps = 50
        if len(self.losses)>=check_window_steps:
            current_avg_loss = np.mean(self.losses[-check_window_steps:])
            if current_avg_loss < self.min_loss_dict['current_min']:
                self.min_loss_dict['current_min'] = current_avg_loss
                self.min_loss_dict['num_steps'] = 0
            else:
                self.min_loss_dict['num_steps'] += 1
            
            if self.min_loss_dict['num_steps']>=self.patience:
                self.stop_training = True       
    # Update Q-Values
    def train(self, new_piece_arrays, result):
        new_state = self.get_state_array(new_piece_arrays)
        new_state = self.process_state(new_state)

        self.current_sqars[4] = new_state

        reward = self.get_reward(result)
        
        # get Q values of new state to update last state's Q values
        with torch.no_grad():
            new_q = self.model(new_state)
        max_q = torch.max(new_q)

        # target value
        Y = reward if result is not None else reward + (self.gamma*max_q)
        Y = torch.Tensor([Y]).detach().squeeze().to(device)
        X = self.current_sqars[1].squeeze()[self.current_sqars[2]]
        loss = self.loss_fn(X, Y)
        self.optimizer.zero_grad()
        loss.backward()
        self.losses.append(loss.item())
        self.optimizer.step()

        self.update_early_stopping()
        if result is not None: # game epoch is over
            self.reset_vars()
    # Load Previous Models
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path).state_dict())

    # Plot the results onto the csv file
    def plot_results(self, show=False, save_path=None):
        """Plots losses, rewards by move"""
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Moves")
        ax1.set_ylabel("Rewards")
        ax1.plot(np.arange(len(self.rewards)), self.rewards, color='r')

        ax2 = ax1.twinx()
        ax2.set_ylabel("Loss")
        ax1.plot(np.arange(len(self.losses)), self.losses, color='b')

        fig.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)
        if show:
            plt.show()

    # Save model and graphing info 
    def save_model_and_results(self, model_path: str):
        # model
        torch.save(self.model, model_path+'model.pth')
        # results: losses, rewards, avg. win rate
        np.savetxt(model_path+'losses.csv', np.array(self.losses), delimiter=',', header='losses')
        self.plot_results(save_path=model_path+'results.png')

class Human(Player):
    def __init__(self, name, turn, test):
        super().__init__(name, turn, test)
# RLBot but with DDQN
class RLBotDDQN(RLBot):
    def __init__(self, name: str, turn: int, test: bool) -> None:
        super().__init__(name, turn, test)
        self.memory = deque(maxlen=1000)
        self.batch_size = 200

        self.target_sync_freq = 200
        self.target_model = copy.deepcopy(self.model)
        self.target_model.load_state_dict(self.model.state_dict())

    def train(self, new_piece_arrays, result):
        new_state = self.get_state_array(new_piece_arrays)
        new_state = self.process_state(new_state)
        self.current_sqars[4] = new_state

        reward = self.get_reward(result)
        curr_experience = (#sqars: state_t, q_vals_t, action_t, reward_t, state_t+1
            self.current_sqars[0], # state t
            self.current_sqars[2], # action t
            self.current_sqars[3], # reward t
            self.current_sqars[4], # state t+1
            int(result is not None)
        )
        self.memory.append(curr_experience)

        if len(self.memory) <= self.batch_size:
            return None

        minibatch = random.sample(self.memory, self.batch_size)
        s_batch = torch.cat([s for (s,a,r,s2,d) in minibatch]).to(device)
        a_batch = torch.Tensor([a for (s,a,r,s2,d) in minibatch]).to(device)
        r_batch = torch.Tensor([r for (s,a,r,s2,d) in minibatch]).to(device)
        s2_batch = torch.cat([s2 for (s,a,r,s2,d) in minibatch]).to(device)
        d_batch = torch.Tensor([d for (s,a,r,s2,d) in minibatch]).to(device)

        q1 = self.model(s_batch)
        # get Q values of new state to update last state's Q values
        with torch.no_grad():
            new_q = self.target_model(s2_batch)
        max_q = torch.max(new_q, dim=1)

        # target value
        Y = r_batch + self.gamma*((1-d_batch)*max_q[0])
        X = q1.gather(dim=1, index=a_batch.long().unsqueeze(dim=1)).squeeze()

        loss = self.loss_fn(X, Y.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.losses.append(loss.item())
        self.optimizer.step()

        if self.n_moves % self.target_sync_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        self.update_early_stopping()
        if result is not None: # game epoch is over
            self.reset_vars()