from preprocessing import *
from custom_env import *
from networks import DDQNAgent
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA

# Predefined Hyperparameters
hyperparameters = {
    "lr": 1e-3,         # Learning rate
    "gamma": 0.9,      # Discount factor for future rewards
    "epsilon": 0.9,     # Initial exploration probability
    "epsilon_decay": 0.995,     # Decay rate for exploration probability
    "epsilon_min": 0.1,         # Minimum exploration probability
    "episodes": 150,            # Total number of episodes for training
    "target_update": 10,        # Frequency of updating the target network
    "log_interval": 100,        # Interval for logging metrics
    "eval_interval": 100,       # Interval for evaluating the agent
    "batch_size": 32,           # Number of experiences sampled per training step
    "memory_size": 200,         # Maximum capacity of the replay buffer
    "hidden_dim": 64,           # Number of units in hidden layers of the neural network
    "linear": False             # Whether to use a linear model or LSTM
}


def evaluate_agent(agent, env, recipe_mapping, df_raw):
    """
    Evaluates the trained agent by simulating its performance for 7 days.
    Displays the meals selected by the agent and their respective nutrient content.

    Parameters:
    - agent: Trained DDQN agent.
    - env: Meal planning environment.
    - recipe_mapping: Mapping of recipe IDs to their names.
    - df_raw: Original unprocessed dataframe containing meal information.
    """
    obs, info = env.reset()
    total_rewards = []
    meals = None        # Proposed meals for the entire week
    # Simulate one week
    for _ in range(7):
        action_mask = env.action_masks()
        action, _ = agent.select_action(obs.flatten(), action_mask)
        obs, reward, terminated, truncated, info = env.step(action)
        total_rewards.append(reward)
        if terminated:
            meals = obs
            obs, info = env.reset()
    env.close()
    print(f"Total evaluation reward: {np.mean(total_rewards)}")

    # Display meals for each day and their respective nutrient details
    day_counter = 1
    for meal in meals:
        name = recipe_mapping[recipe_mapping['RecipeId']
                              == meal[0]]['Name'].values[0]
        print(f'Day - {day_counter}: {name}')
        print(f"{df_raw[df_raw['RecipeId'] == meal[0]]['Keywords'].values[0]}")
        day_counter += 1

    # Plot nutrient content of selected meals
    plot_state_contents(meals, agent)


def calculate_average_content(agent):
    """
    Calculates the average nutritional content (protein, fiber, saturated fat) per episode
    of meals selected by the agent over the entire training history.

    Parameters:
    - agent: Trained DDQN agent.

    Returns:
    - A list containing average values for protein, fiber, and saturated fat per episode.
    """
    # Function to scale nutrient values back to their original range
    def unscaled_value(x, column): return (((x - 0) / (1 - 0)) *
                                           (agent.env.df[column].max() - agent.env.df[column].min())) + agent.env.df[column].min()
    # Arrays to store average values per episode
    avg_protein_per_episode = []
    avg_fiber_per_episode = []
    avg_saturated_fat_per_episode = []
    df_columns = np.array(agent.env.df.columns)

    # Process observed states from each episode
    for episode in agent.obs_history:
        weekly_protein_content = 0
        weekly_fiber_content = 0
        weekly_saturated_fat_content = 0

        for state in episode:
            # Extract and unscale nutrient values
            usncaled_protein_content = unscaled_value(
                state[np.where(df_columns == 'ProteinContent')[0][0]], 'ProteinContent')
            usncaled_fiber_content = unscaled_value(
                state[np.where(df_columns == 'FiberContent')[0][0]], 'FiberContent')
            usncaled_saturated_fat_content = unscaled_value(state[np.where(
                df_columns == 'SaturatedFatContent')[0][0]], 'SaturatedFatContent')
            weekly_protein_content += usncaled_protein_content
            weekly_fiber_content += usncaled_fiber_content
            weekly_saturated_fat_content += usncaled_saturated_fat_content

        # Calculate daily averages
        avg_protein = weekly_protein_content/7
        avg_fiber = weekly_fiber_content/7
        avg_saturated_fat = weekly_saturated_fat_content/7
        avg_protein_per_episode.append(avg_protein)
        avg_fiber_per_episode.append(avg_fiber)
        avg_saturated_fat_per_episode.append(avg_saturated_fat)

    return [avg_protein_per_episode, avg_fiber_per_episode, avg_saturated_fat_per_episode]


def calculate_overlap_preferences(agent, episodes):
    """
    Computes the average overlap between meals selected by the agent 
    and user-preferred keyword-based meals.

    Parameters:
    - agent: Trained DDQN agent.
    - episodes: Total number of episodes trained.
    """
    if agent.env.user_preference_indices:
        avg_overlap_fraction = 0
        for obs in agent.obs_history:
            overlap_fraction_week = 0
            for day in obs:
                # Calculates for each day the fraction of matching keywords with user preference
                overlap_fraction = agent.env.calculate_keyword_reward(day)
                overlap_fraction_week += overlap_fraction
            # For each episode the average daily fraction is computed
            avg_overlap_fraction += overlap_fraction_week/7
        print(
            f"Average overlap fraction per episode with user preferred keywords: {avg_overlap_fraction/episodes}")
    else:
        print("No user preferences found")


def plot_average_content(avg_content_array, episodes):
    """
    Plots the average nutrient content (protein, fiber, saturated fat) of meals 
    over the episodes.

    Parameters:
    - avg_content_array: List of average nutrient values for each episode.
    - episodes: Total number of episodes.
    """
    labels = ['Average Protein', 'Average Fiber', 'Average Saturated Fat']
    colors = ['#1f77b4', '#2ca02c', '#d62728']  # Blue, Green, Red
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))  # 3 rows, 1 column

    for i, avg_content in enumerate(avg_content_array):
        # Plot each nutrient content over episodes
        axs[i].plot(range(episodes), avg_content,
                    label=labels[i], color=colors[i], linewidth=2)
        axs[i].set_title(f'{labels[i]} per Meal', fontsize=14)
        axs[i].set_xlabel('Episodes')
        axs[i].set_ylabel('Average Content')
        axs[i].legend()
        axs[i].grid(True, alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_cumsum(rewards, episodes):
    """
    Plots the cumulative sum of rewards over all episodes.

    Parameters:
    - rewards: List of total rewards per episode.
    - episodes: Total number of episodes.
    """
    cum_rewards = np.cumsum(rewards)
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(0, episodes), cum_rewards, color='#ff7f0e', linewidth=2)
    plt.title("Cumulative Rewards Over Episodes", fontsize=16)
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_rewards(rewards, episodes):
    """
    Plots the total reward obtained in each episode.

    Parameters:
    - rewards: List of total rewards per episode.
    - episodes: Total number of episodes.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(0, episodes), rewards, color='#9467bd', linewidth=2)
    plt.title("Total Rewards Over Episodes", fontsize=16)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_mse_loss(mse_losses, episodes):
    """
    Plots the mean squared error (MSE) loss during training over episodes.

    Parameters:
    - mse_losses: List of total MSE loss values per episode.
    - episodes: Total number of episodes.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(0, episodes), mse_losses, color='#8c564b', linewidth=2)
    plt.title("MSE Loss Over Episodes", fontsize=16)
    plt.xlabel("Episodes")
    plt.ylabel("MSE Loss")
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_q_values(q_values, episodes):
    """
    Plots the total maximum Q-values over episodes to monitor the stability of the policy.

    Parameters:
    - q_values: List of total maximum Q-values per episode.
    - episodes: Total number of episodes.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(0, episodes), q_values, color='#3c564b', linewidth=2)
    plt.title("Max Q Values Over Episodes", fontsize=16)
    plt.xlabel("Episodes")
    plt.ylabel("Max Q Value")
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()


def success_rates(avg_content_array, episodes, env):
    """
    Calculates the success rate of achieving dietary guidelines for nutrients within an allowed threshold.

    Parameters:
    - avg_content_array: List of average nutrient values per episode (protein, fiber, saturated fat).
    - episodes: Total number of episodes.
    - env: Meal planning environment containing dietary guidelines.

    Returns:
    - List of success counts for each nutrient type.
    """
    labels = ['Average Protein', 'Average Fiber', 'Average Saturated Fat']
    ideal_values = [env.user_dietary_guidelines["ProteinContent"],
                    env.user_dietary_guidelines["FiberContent"], env.user_dietary_guidelines["SaturatedFatContent"]]  # Ideal values for Protein, Fiber, Saturated Fat
    tolerance = [5, 5, 5]  # Tolerances for each content type
    success = []
    for i, content in enumerate(avg_content_array):
        success_count = 0  # Reset the success count for each content type

        for episode in content:
            # Special case for saturated fat (it should be below the ideal value)
            if labels[i] == 'Average Saturated Fat' and episode < ideal_values[i]:
                success_count += 1

            # Check if within tolerance range for protein and fiber
            elif abs(episode - ideal_values[i]) < tolerance[i]:
                success_count += 1

        print(
            f"Number of successes for {labels[i]}: {success_count} / {episodes}")
        success.append(success_count)
    return success


def plot_state_contents(meal_plan, agent):
    """
    Visualizes the daily nutrient content of meals selected by the agent.

    Parameters:
    - meal_plan: List of meals selected over a week.
    - agent: Trained DDQN agent.
    """
    # Function to scale nutrient values back to their original range
    def unscaled_value(x, column): return (((x - 0) / (1 - 0)) *
                                           (agent.env.df[column].max() - agent.env.df[column].min())) + agent.env.df[column].min()
    df_columns = np.array(agent.env.df.columns)
    protein_values = []
    fiber_values = []
    saturated_fat_values = []

    # Extract nutrient values for each day in the meal plan
    for day in meal_plan:
        unscaled_protein_content = unscaled_value(
            day[np.where(df_columns == 'ProteinContent')[0][0]], 'ProteinContent')
        unscaled_fiber_content = unscaled_value(
            day[np.where(df_columns == 'FiberContent')[0][0]], 'FiberContent')
        unscaled_saturated_fat_content = unscaled_value(
            day[np.where(df_columns == 'SaturatedFatContent')[0][0]], 'SaturatedFatContent')
        protein_values.append(unscaled_protein_content)
        fiber_values.append(unscaled_fiber_content)
        saturated_fat_values.append(unscaled_saturated_fat_content)

    # Display the nutrient contents
    print(f"Protein Content of the meals: {protein_values}")
    print(f"Fiber Content of the meals: {fiber_values}")
    print(f"Saturated Fat Content of the meals: {saturated_fat_values}")

    # Plot nutrient values over the week
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(0, len(meal_plan)), protein_values,
             label='Protein Content', color='#1f77b4', linewidth=2)
    plt.plot(np.arange(0, len(meal_plan)), fiber_values,
             label='Fiber Content', color='#2ca02c', linewidth=2)
    plt.plot(np.arange(0, len(meal_plan)), saturated_fat_values,
             label='Saturated Fat Content', color='#d62728', linewidth=2)
    plt.xlabel('Days', fontsize=12)
    plt.ylabel('Content Value', fontsize=12)
    plt.legend(fontsize=12)
    plt.title('Content Value of Meals Over the Week', fontsize=16)
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Set random seed for reproducibility
    seed_number = 20
    random.seed(seed_number)

    # Preprocess data
    df_raw = read_data()
    df_raw = df_raw.sample(frac=1).reset_index(drop=True)
    recipe_mapping, df_processed = inspect_and_transform_entries_of_df(df_raw)

    # Simulate User Preferences
    user_meal_indices = random.sample(range(0, df_processed.shape[0]-1), 10)

    # Initialize the meal planning environment
    env_py = MealPlannerEnv(
        df_processed, user_dietary_guidelines, user_meal_indices)

    # Create and train the DDQN agent
    agent = DDQNAgent(env_py, **hyperparameters)
    rewards, mse_losses, num_episodes, eval_avg_return = agent.train()

    # Evaluate the trained agent
    evaluate_agent(agent, env_py, recipe_mapping, df_raw)

    # Analyze and visualize training metrics
    content_arrays = calculate_average_content(agent)
    plot_average_content(content_arrays, num_episodes)

    calculate_overlap_preferences(agent, num_episodes)
    success_rates(content_arrays, num_episodes, env_py)

    # Visualize rewards and training progress
    plot_cumsum(rewards, num_episodes)
    plot_rewards(rewards, num_episodes)
    plot_mse_loss(rewards, num_episodes)
    plot_q_values(agent.q_values_array, num_episodes)