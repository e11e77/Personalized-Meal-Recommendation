import pytest
import pandas as pd
from unittest.mock import patch
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from custom_env import MealPlannerEnv
from networks import DDQNAgent
from postprocessing import *

@pytest.fixture
def mock_mapping():
    return pd.DataFrame({
        'RecipeId': [0, 1, 2, 3, 4, 5, 6, 7, 8],
        'Name': ['0', '1', '2', '3', '4', '5', '6', '7', '8']
    })


@pytest.fixture
def mock_env():
    df_mock = pd.DataFrame({
        'RecipeId': [0, 1, 2, 3, 4, 5, 6, 7, 8],
        'ProteinContent': [10, 20, 15, 10, 20, 15, 10, 20, 15],
        'FiberContent': [5, 10, 7, 5, 10, 7, 5, 10, 7],
        'SugarContent': [3, 4, 5, 5, 10, 7, 5, 10, 7],
        'Calories': [300, 500, 400, 300, 500, 400, 300, 500, 400],
        'CarbohydrateContent': [50, 60, 55, 50, 60, 55, 50, 60, 55],
        'SodiumContent': [500, 700, 600, 500, 700, 600, 500, 700, 600],
        'CholesterolContent': [100, 120, 110, 100, 120, 110, 100, 120, 110],
        'SaturatedFatContent': [10, 15, 12, 10, 15, 12, 10, 15, 12],
        'FatContent': [30, 40, 35, 30, 40, 35, 30, 40, 35],
        'ReviewCount': [50, 30, 45, 50, 30, 45, 50, 30, 45],
        'AggregatedRating': [4.5, 3.8, 4.2, 4.5, 3.8, 4.2, 4.5, 3.8, 4.2],
        'TotalTime': [30, 45, 40, 30, 45, 40, 30, 45, 40],
        'PrepTime': [15, 20, 18, 15, 20, 18, 15, 20, 18],
        'CookTime': [15, 25, 22, 15, 25, 22, 15, 25, 22],
        'RecipeServings': [2, 4, 3, 2, 4, 3, 2, 4, 3],
        "Keyword1": [1, 0, 0, 1, 0, 0, 1, 0, 0],
        "Keyword2": [0, 1, 1, 0, 1, 1, 0, 1, 1]
    })
    user_dietary_guidelines = {
        'ProteinContent': 50/3,
        'FiberContent': 25/3,
        'SugarContent': 25/3,
        'Calories': 800,
        'CarbohydrateContent': 350,
        'SodiumContent': 2000/3,
        'CholesterolContent': 300/3,
        'SaturatedFatContent': 20,
        'FatContent': 40/3
    }
    return MealPlannerEnv(df_mock, user_dietary_guidelines)


@pytest.fixture
def mock_agent(mock_env):
    return DDQNAgent(env=mock_env, hidden_dim=32, linear=True, episodes=1)


def test_evaluate_agent(mock_agent, mock_env, mock_mapping):
    # Test if the function runs without errors
    mock_agent.train()
    df_raw = mock_env.df.copy()
    df_raw["Keywords"] = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    assert evaluate_agent(mock_agent, mock_env, mock_mapping, df_raw) == None


def test_calculate_average_content(mock_agent):
    # Run the function and check the results
    mock_agent.train()
    result = calculate_average_content(mock_agent)
    assert len(result) == 3
    assert all(isinstance(x, float) for x in result[0])
    assert all(isinstance(x, float) for x in result[1])
    assert all(isinstance(x, float) for x in result[2])
    assert len(result[0]) == mock_agent.episodes


def test_calculate_overlap_preferences(mock_agent):
    # Test if the function runs without errors
    mock_agent.train()
    assert calculate_overlap_preferences(
        mock_agent, mock_agent.episodes) == None


def test_calculate_success_rates(mock_agent, mock_env):
    mock_agent.train()
    results = calculate_average_content(mock_agent)
    successes = success_rates(results, mock_agent.episodes, mock_env)

    assert len(successes) == 3
    assert all(isinstance(x, int) for x in successes)
    assert all(x >= 0 for x in successes)


@patch("postprocessing.plt.show")
def test_plot_average_content(mock_show):
    avg_content_array = [
        [20.0, 25.0, 22.0],  # Example data for Protein
        [10.0, 15.0, 12.0],  # Example data for Fiber
        [5.0, 6.0, 7.0]      # Example data for Saturated Fat
    ]
    episodes = 3
    plot_average_content(avg_content_array, episodes)
    mock_show.assert_called_once()


@patch("postprocessing.plt.show")
def test_plot_cumsum(mock_show):
    rewards = [10, 15, 20, 25]  # Example rewards
    episodes = len(rewards)
    plot_cumsum(rewards, episodes)
    mock_show.assert_called_once()


@patch("postprocessing.plt.show")
def test_plot_mse_loss(mock_show):
    mse_losses = [0.1, 0.08, 0.07, 0.05]  # Example loss values
    episodes = len(mse_losses)
    plot_mse_loss(mse_losses, episodes)
    mock_show.assert_called_once()


@patch("postprocessing.plt.show")
def test_plot_rewards(mock_show):
    rewards = [20, 0.08, 31, 0.05]  # Example loss values
    episodes = len(rewards)
    plot_mse_loss(rewards, episodes)
    mock_show.assert_called_once()


@patch("postprocessing.plt.show")
def test_plot_q_values(mock_show):
    q_values = [20, 100, 3, 4.89]  # Example loss values
    episodes = len(q_values)
    plot_mse_loss(q_values, episodes)
    mock_show.assert_called_once()


@patch("postprocessing.plt.show")
def test_plot_state_contents(mock_show, mock_agent):
    # Mock meal plan
    meal_plan = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    mock_agent.env.df = pd.DataFrame({
        "ProteinContent": [0.0, 1.0],
        "FiberContent": [0.0, 1.0],
        "SaturatedFatContent": [0.0, 1.0]
    })
    plot_state_contents(meal_plan, mock_agent)
    mock_show.assert_called_once()
