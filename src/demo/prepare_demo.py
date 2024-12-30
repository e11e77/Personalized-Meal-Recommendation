import random
import numpy as np
import pickle 

import sys
import os

# Necessary for importing modules
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from custom_env import *
from networks import DDQNAgent
from preprocessing import *

# Predefined Hyperparameters
hyperparameters = {
    "lr": 1e-3,                 # Learning rate
    "gamma": 0.9,               # Discount factor for future rewards
    "epsilon": 0.9,             # Initial exploration probability
    "epsilon_decay": 0.995,     # Decay rate for exploration probability
    "epsilon_min": 0.1,         # Minimum exploration probability
    "episodes": 150,            # Total number of episodes for training
    "target_update": 10,        # Frequency of updating the target network
    "log_interval": 50,        # Interval for logging metrics
    "eval_interval": 50,       # Interval for evaluating the agent
    "batch_size": 32,           # Number of experiences sampled per training step
    "memory_size": 200,         # Maximum capacity of the replay buffer
    "hidden_dim": 64,           # Number of units in hidden layers of the neural network
    "linear": False             # Whether to use a linear model or LSTM
}


def save_model(model, filename='meal_planner.sav'):
    """
    Save an object to a specified file.

    Parameters:
    - model: The object to be saved.
    - filename: The name of the file where the model will be saved (default is 'meal_planner.sav').

    This function uses the `pickle` module to serialize the object and store it in a file.
    """
    current_dir = os.path.dirname(__file__)  
    file_path = os.path.join(current_dir, filename)

    pickle.dump(model, open(file_path, 'wb'))


def predict_week(agent, recipe_mapping):
    """
    Predict a week's worth of meals based on the trained agent.

    Parameters:
    - agent: The trained agent used to make meal predictions.
    - recipe_mapping: A mapping of RecipeId to meal names.

    Returns:
    - named_meals: A list of meal names for the entire week based on the agent's actions.
    """
    obs, info = agent.env.reset()
    meals = None        # Proposed meals for the entire week
    # Simulate one week
    for _ in range(7):
        action_mask = agent.env.action_masks()
        action, _ = agent.select_action(obs.flatten(), action_mask)
        obs, reward, terminated, truncated, info = agent.env.step(action)
        if terminated:
            meals = obs
            obs, info = agent.env.reset()
    agent.env.close()

    named_meals = []
    for meal in meals:
        name = recipe_mapping[recipe_mapping['RecipeId']
                              == meal[0]]['Name'].values[0]
        named_meals.append(name)

    return named_meals



if __name__ == '__main__':
    """
    Main entry point of the script. This section of the code initializes the environment, 
    gets user preferences for meals, sets up and trains the agent, and saves the trained model.

    It saves the trained agent and recipe mapping for the final demo.
    """
    # Set random seed for reproducibility
    seed_number = 20
    random.seed(seed_number)

    # Preprocess data
    df_raw = read_data()
    df_raw = df_raw.sample(frac=1).reset_index(drop=True)
    recipe_mapping, df_processed = inspect_and_transform_entries_of_df(df_raw)

    # Get User Preferences
    user_indices = []
    random_indices = random.sample(range(0, df_processed.shape[0]-1), 5)
    for random_index in random_indices:
        print('------------------------------')
        print(random_index)
        if not recipe_mapping['RecipeId'].isin([random_index]).any():
            print('Index does not exist in mapping - skipping element.')
            continue
        name = recipe_mapping[recipe_mapping['RecipeId'] == random_index]['Name'].values[0]
        print(f"Meal: {name}")
        choice = input("Add that meal to preferred recipes [y/n]?: ")
        if choice == 'y':
            user_indices.append(random_index)
    if len(user_indices) == 0:
        user_indices = None

    # Initialize the meal planning environment
    env_py = MealPlannerEnv(
        df_processed, user_dietary_guidelines, user_indices)
    
    # Create and train the DDQN agent
    agent = DDQNAgent(env_py, **hyperparameters)
    rewards, mse_losses, num_episodes, eval_avg_return = agent.train()
    
    # Store mapping and model
    save_model(agent, 'meal_planner.sav')
    save_model(recipe_mapping, 'recipe_mapping.sav')
    

