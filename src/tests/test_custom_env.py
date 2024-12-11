import warnings
from stable_baselines3.common.env_checker import check_env
from custom_env import MealPlannerEnv
from unittest.mock import patch
import pytest
import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def mock_df():
    return pd.DataFrame({
        'RecipeId': [0, 1, 2],
        'ProteinContent': [10, 20, 15],
        'FiberContent': [5, 10, 7],
        'SugarContent': [3, 4, 5],
        'Calories': [300, 500, 400],
        'CarbohydrateContent': [50, 60, 55],
        'SodiumContent': [500, 700, 600],
        'CholesterolContent': [100, 120, 110],
        'SaturatedFatContent': [10, 15, 12],
        'FatContent': [30, 40, 35],
        'ReviewCount': [50, 30, 45],
        'AggregatedRating': [4.5, 3.8, 4.2],
        'TotalTime': [30, 45, 40],
        'PrepTime': [15, 20, 18],
        'CookTime': [15, 25, 22],
        'RecipeServings': [2, 4, 3],
        "Keyword1": [1, 0, 0],
        "Keyword2": [0, 1, 1]
    })


@pytest.fixture
def mock_env(mock_df):
    user_dietary_guidelines = {
        'ProteinContent': 50 / 3,
        'FiberContent': 25 / 3,
        'SugarContent': 25 / 3,
        'Calories': 800,
        'CarbohydrateContent': 350,
        'SodiumContent': 2000 / 3,
        'CholesterolContent': 300 / 3,
        'SaturatedFatContent': 20,
        'FatContent': 40 / 3
    }
    return MealPlannerEnv(mock_df, user_dietary_guidelines)


@pytest.fixture
def mock_user_env(mock_df):
    user_dietary_guidelines = {
        'ProteinContent': 50 / 3,
        'FiberContent': 25 / 3,
        'SugarContent': 25 / 3,
        'Calories': 800,
        'CarbohydrateContent': 350,
        'SodiumContent': 2000 / 3,
        'CholesterolContent': 300 / 3,
        'SaturatedFatContent': 20,
        'FatContent': 40 / 3
    }
    user_meal_indices = [1]
    return MealPlannerEnv(mock_df, user_dietary_guidelines, user_meal_indices)


def test_initialization(mock_env, mock_df):
    assert mock_env.size_meals == len(mock_df)
    assert mock_env.state.shape == (
        mock_env.days_in_week, mock_env.num_features)
    assert mock_env.action_space.n == len(mock_df)
    assert mock_env.observation_space.shape == (
        mock_env.days_in_week, mock_env.num_features)
    assert mock_env.user_preference_indices == None
    assert mock_env.state.dtype == mock_env.observation_space.dtype


def test_reset(mock_env):
    state, _ = mock_env.reset()
    assert np.array_equal(state, np.zeros_like(state))  # Reset state to zeros
    assert mock_env.current_day == 0
    assert np.array_equal(mock_env.invalid_action_mask,
                          np.ones(mock_env.size_meals, dtype=np.int64))


def test_step(mock_env):
    for action in [0, 2]:
        mock_env.reset()
        state, reward, terminated, truncated, _ = mock_env.step(action)
        assert np.array_equal(
            state[mock_env.current_day - 1], mock_env.scaled_df.iloc[action].values)
        assert mock_env.invalid_action_mask[action] == 0
        assert not terminated or mock_env.current_day == mock_env.days_in_week
        assert not truncated


def test_action_masks(mock_env):
    mock_env.reset()
    assert np.array_equal(mock_env.action_masks(), np.ones(
        mock_env.size_meals, dtype=np.int64))
    mock_env.step(0)
    assert mock_env.action_masks()[0] == 0  # Action 0 should now be invalid


def test_reward_calculation(mock_user_env):
    with patch.object(mock_user_env, 'calculate_keyword_reward', return_value=1) as mocked_keyword_func:
        mock_user_env.reset()
        state, reward, terminated, truncated, _ = mock_user_env.step(0)
        # Check reward logic for the first step
        assert reward > 0
        mocked_keyword_func.assert_called_once()  # Ensure keyword reward was computed


def test_environment_compatibility(mock_env):
    # Validate compliance with Gym API using check_env
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        check_env(mock_env)


def test_reward_on_termination(mock_env):
    mock_env.reset()
    for day in range(mock_env.days_in_week):
        state, reward, terminated, truncated, _ = mock_env.step(
            day % mock_env.size_meals)
    assert terminated
    assert reward > 0
