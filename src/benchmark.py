from preprocessing import *
from custom_env import *
from networks import DDQNAgent
import matplotlib.pyplot as plt
import random
import itertools
from postprocessing import calculate_average_content, success_rates, calculate_overlap_preferences

# Hyperparameter ranges for tuning
# Each key-value pair represents a parameter and its possible values
hyperparameter_grid = {
    "lr": [1e-3, 1e-5, 1e-6],               # Learning rate values
    # Discount factor for future rewards
    "gamma": [0.99, 0.95],
    "epsilon": [1.0, 0.8],                  # Initial epsilon for exploration
    "epsilon_decay": [0.995, 0.99],         # Rate at which epsilon decreases
    "epsilon_min": [0.1],                   # Minimum epsilon value
    "episodes": [150],                      # Number of training episodes
    # Frequency of target network updates
    "target_update": [10],
    "log_interval": [50],                   # Logging frequency during training
    # Evaluation frequency during training
    "eval_interval": [50],
    "batch_size": [32],                     # Batch size for training
    "hidden_dim": [64]                      # Number of units in hidden layers
}


def plot_cumsum_rewards_parameters(results_array_values, param_array):
    """
    Plots the cumulative rewards over episodes for different hyperparameter configurations.

    Parameters:
    - results_array_values (list): List of dictionaries containing cumulative rewards for each configuration.
    - param_array (list): List of parameter configurations used for training.
    """
    plt.figure(figsize=(12, 6))
    plt.title(
        "Cumulative Rewards Over Episodes for Different Hyperparameter Configurations")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    for result, params in zip(results_array_values, param_array):
        plt.plot(result["cumsum_rewards"], label=f"Config: {params['params']}")
    plt.tight_layout()
    plt.show()


def plot_avg_content_parameters(results_array_values, param_array):
    """
    Plots average content metrics over episodes for different hyperparameter configurations.

    Parameters:
    - results_array_values (list): List of dictionaries containing average content metrics.
    - param_array (list): List of parameter configurations used for training.
    """
    labels = ['Average Protein Content',
              'Average Fiber content', 'Average Saturated Fat Content']
    for i in range(len(labels)):
        plt.figure(figsize=(12, 6))
        plt.title(
            f"{labels[i]} Over Episodes for Different Hyperparameter Configurations")
        plt.xlabel("Episodes")
        plt.ylabel(f"{labels[i]}")
        for result, params in zip(results_array_values, param_array):
            plt.plot(result["avg_contents"][i],
                     label=f"Config: {params['params']}")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # Set seed for reproducibility
    seed_number = 20
    random.seed(seed_number)

    # Preprocess data
    df_raw = read_data()
    recipe_mapping, df_processed = inspect_and_transform_entries_of_df(df_raw)

    # Simulate user preferences
    user_meal_indices = random.sample(range(0, df_processed.shape[0] - 1), 10)

    # Initialize the custom meal planning environment
    env_py = MealPlannerEnv(
        df_processed, user_dietary_guidelines, user_meal_indices)

    # Generate all combinations of hyperparameters
    keys, values = zip(*hyperparameter_grid.items())
    param_combinations = [dict(zip(keys, v))
                          for v in itertools.product(*values)]

    # Train and evaluate with linear and non-linear network configurations
    network_params = [False, True]
    for linear_network in network_params:
        results = []
        results_array_values = []

        print(f"Usage of linear network: {linear_network}")

        # Hyperparameter tuning loop
        for params in param_combinations:
            print(f"Training with hyperparameters: {params}")

            # Initialize and train the agent with current hyperparameters
            agent = DDQNAgent(env_py, **params, linear=linear_network)
            rewards, mse_losses, num_episodes, eval_avg_return = agent.train()

            # Compute evaluation metrics
            avg_reward = np.mean(rewards)
            avg_loss = np.mean([float(loss)
                               for loss in mse_losses if loss is not None])
            total_reward = np.sum(rewards)
            cumsum_rewards = np.cumsum(rewards)
            avg_content_per_episode = calculate_average_content(agent)
            success_param = success_rates(
                avg_content_per_episode, num_episodes, env_py)
            calculate_overlap_preferences(agent, num_episodes)

            # Store results for comparison and analysis
            results.append({
                "params": params,
                "linear_network_used": linear_network,
                "avg_reward": avg_reward,
                "avg_loss": avg_loss,
                "total_reward": total_reward,
                "success_protein": success_param[0],
                'success_fiber': success_param[1],
                "success_saturated_fat": success_param[2]
            })
            results_array_values.append({
                "cumsum_rewards": cumsum_rewards,
                "avg_contents": avg_content_per_episode
            })

            # Log training progress
            print(
                f"Avg Reward: {avg_reward}, Avg Loss: {avg_loss}, Total Reward: {total_reward}")

        # Sort results by average reward in descending order and save to a CSV
        sorted_results = sorted(
            results, key=lambda x: x['avg_reward'], reverse=True)
        df_results = pd.DataFrame(sorted_results)
        df_results.to_csv(
            f"hyperparameter_comparison_results_{linear_network}.csv", index=False)

        # Plot results for analysis
        plot_cumsum_rewards_parameters(results_array_values, results)
        plot_avg_content_parameters(results_array_values,  results)
