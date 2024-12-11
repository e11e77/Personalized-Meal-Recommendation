import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, observation_space, action_space, hidden_dim, linear=False):
        """
        Initializes the QNetwork with either a fully connected linear or LSTM-based architecture.
        
        Args:
            observation_space (int): The number of input features (state space dimension).
            action_space (int): The number of possible actions (action space dimension).
            hidden_dim (int): The number of units in the hidden layers.
            linear (bool): If True, use a fully connected linear network; otherwise, use an LSTM.
        """
        super(QNetwork, self).__init__()
        self.linear = linear
        if linear:
            # Define a fully connected linear network if 'linear' is True
            self.layer1 = nn.Linear(observation_space, hidden_dim)
            self.layer2 = nn.Linear(hidden_dim, hidden_dim)
            self.layer3 = nn.Linear(hidden_dim, action_space)
        else:
            # Define an LSTM network if 'linear' is False
            self.lstm = nn.LSTM(input_size=observation_space, hidden_size=hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, action_space)

    def forward(self, x):
        """
        Defines the forward pass through the network.

        Args:
            x (Tensor): The input tensor representing the state.

        Returns:
            Tensor: The Q-values corresponding to each action.
        """
        if self.linear:
            x = torch.relu(self.layer1(x))
            x = torch.relu(self.layer2(x))
            x = self.layer3(x)
        else:
            output, (h_n, c_n) = self.lstm(x)
            x = self.fc(output)
        return x
    
    
class DDQNAgent:
    def __init__(self, env,
                 lr=1e-3, gamma=0.99, 
                 epsilon=0.9, epsilon_decay=0.995, epsilon_min=0.1, 
                 episodes = 100, target_update = 10,
                 log_interval=10, eval_interval=10,
                 batch_size=32, memory_size=10000, hidden_dim=64, linear=False):
        """
        Initializes the Double Deep Q-Learning (DDQN) agent.

        Args:
            env (gym.Env): The custom meal planning environment where the agent interacts.
            lr (float): The learning rate for the optimizer.
            gamma (float): The discount factor for future rewards.
            epsilon (float): The probability of taking a random action (exploration).
            epsilon_decay (float): Decay rate for epsilon over time (for exploration-exploitation balance).
            epsilon_min (float): Minimum value for epsilon.
            episodes (int): Number of episodes for training.
            target_update (int): Interval for updating the target network.
            log_interval (int): Interval for logging training progress.
            eval_interval (int): Interval for evaluating the agent's performance.
            batch_size (int): The batch size for training.
            memory_size (int): The size of the replay buffer.
            hidden_dim (int): Number of units in the hidden layer(s).
            linear (bool): Whether to use a fully connected linear or LSTM network.
        """
        self.observation_space = env.observation_space.shape[0] * env.observation_space.shape[1]
        self.action_space = env.action_space.n
        self.env = env
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.obs_history = []
        
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.episodes = episodes
        self.target_update = target_update
        self.batch_size = batch_size

        self.count_taken_actions = 0
        self.count_exploration = 0
        self.q_values_array = []

        # Initialize the replay buffer (deque) and Q-networks (main and target)
        self.memory = deque(maxlen=memory_size)
        self.q_network = QNetwork(self.observation_space, self.action_space, hidden_dim, linear)
        self.target_network = QNetwork(self.observation_space, self.action_space, hidden_dim, linear)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

        # Perform an initial copy of the parameters from the Q-network to the target network
        self.update_target_network()

    def update_target_network(self):
        """
        Performs a hard update of the target network by copying the parameters from the Q-network.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """
        Stores the experience in the replay buffer.
        
        Args:
            state (array): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (array): The next state after the action.
            done (bool): Whether the episode has ended.
        """
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state, action_mask):
        """
        Selects an action based on the epsilon-greedy policy.
        
        Args:
            state (array): The current state.
            action_mask (array): A mask indicating which actions are valid.

        Returns:
            tuple: The selected action and the corresponding Q-value (if applicable).
        """
        self.count_taken_actions += 1
        if np.random.rand() < self.epsilon:
            # Select a random action from valid actions
            self.count_exploration += 1
            valid_actions = [i for i, value in enumerate(action_mask) if value != 0]
            return np.random.choice(valid_actions), None
        # If not exploring, select the best action based on the Q-values
        state = torch.FloatTensor(state).unsqueeze(0)   # Add batch dimension
        q_values = self.q_network(state)
        q_values = q_values.squeeze(0)      # Remove batch dimension  
        masked_q_values = torch.tensor(action_mask) * q_values
        return torch.argmax(masked_q_values).item(), torch.max(masked_q_values).item()
    
    def replay(self):
        """
        Performs a training step using experience replay.
        
        Randomly sample a batch from the memory, compute the loss, and update the Q-network.

        Returns:
            loss (Tensor): The computed loss for this training step.
        """
        if len(self.memory) < self.batch_size:
            return None     # Not enough samples for training
        # Sample a batch from the replay memory
        batch = random.sample(self.memory, self.batch_size)
        loss = self.compute_loss(batch)
        
        # Perform backpropagation to update the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def compute_loss(self, batch):
        """
        Computes the loss for the current batch of experiences.
        
        Args:
            batch (list): A batch of experiences (state, action, reward, next_state, done).

        Returns:
            loss (Tensor): The computed loss.
        """
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert the batch into tensors
        states = torch.FloatTensor(np.array(states).reshape(len(states), -1))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states).reshape(len(states), -1))
        dones = torch.FloatTensor(dones)

        # Compute Q-values and next Q-values with target network (double DQN update rule)
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, target_q_values.detach())
        return loss

    def compute_average_return(self):
        """
        Evaluates the agent's performance by computing the average reward over 20 episodes.
        
        Returns:
            average_return (float): The average reward over 20 episodes.
        """
        average_return = 0
        for _ in range(20):
            obs, _ = self.env.reset()
            done = False
            while not done:
                action_mask = self.env.action_masks()
                action, _ = self.select_action(obs.flatten(), action_mask)   
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated
                average_return += reward
        self.env.close()
        return average_return/20


    def train(self):
        """
        Trains the DDQN agent over multiple episodes.
        
        For each episode, the agent interacts with the environment, stores experiences, performs 
        training steps, and logs progress periodically.

        Returns:
            rewards (list): The list of total rewards obtained in each episode.
            mse_losses (list): The list of mean squared errors for each episode.
            eval_avg_return (list): The average reward during evaluation intervals.
        """
        rewards = []
        mse_losses = []
        eval_avg_return = []
        self.obs_history = []
        for episode in range(self.episodes):
            state, _ = self.env.reset()
            state = state.flatten()
            total_reward = 0.0
            total_mse_loss = 0.0
            done = False
            total_max_q_value = 0.0

            # Keep interacting with the environment in the current episode until the environment has been terminated (7 days reached)
            while not done:
                action_mask = self.env.action_masks()
                action, max_q_value = self.select_action(state, action_mask)
                
                # Take the selected action and observe the next state and reward
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated
                obs = next_state
                next_state = next_state.flatten()
                
                # Store the experience in the replay buffer
                self.remember(state, action, reward, next_state, done)
                mse_loss = self.replay()
                state = next_state
                # Save parameters for postprocessing 
                if mse_loss is not None:
                    total_mse_loss += mse_loss
                total_reward += reward
                if max_q_value is not None:
                    total_max_q_value += max_q_value

            rewards.append(total_reward)
            mse_losses.append(total_mse_loss)
            self.obs_history.append(obs)
            self.q_values_array.append(total_max_q_value)

            if episode % self.target_update == 0:
                self.update_target_network()

            if episode % self.log_interval == 0:
                print(f"Episode: {episode}, Total Reward: {total_reward}, Total MSE Loss: {total_mse_loss}, Epsilon: {self.epsilon}")
                print(f"# Taken Exploration / # Taken Actions: {self.count_exploration} / {self.count_taken_actions}")
            if episode % self.eval_interval == 0:
                average_return = self.compute_average_return()
                print(f"Episode: {episode}, Average Return: {average_return}")
                eval_avg_return.append(average_return)

        return rewards, mse_losses, self.episodes, eval_avg_return