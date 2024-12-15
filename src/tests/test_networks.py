import pandas as pd

import pytest
import torch
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from networks import QNetwork, DDQNAgent
from custom_env import MealPlannerEnv

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
def mock_network():
    return QNetwork(observation_space=9, action_space=3, hidden_dim=32, linear=True)


@pytest.fixture
def mock_agent(mock_env):
    return DDQNAgent(env=mock_env, hidden_dim=32, linear=True)


def test_qnetwork_initialization(mock_network):
    assert isinstance(mock_network, QNetwork)
    assert hasattr(mock_network, "linear")
    assert mock_network.layer1.weight.shape[1] == 9  # Observation space size
    assert mock_network.layer3.weight.shape[0] == 3  # Action space size


def test_qnetwork_forward_linear(mock_network):
    input_data = torch.rand(1, 9)  # Batch size of 1, observation space of 9
    output = mock_network(input_data)
    assert output.shape == (1, 3)  # Output should match action space size


def test_qnetwork_forward_lstm():
    lstm_network = QNetwork(observation_space=9,
                            action_space=3, hidden_dim=32, linear=False)
    input_data = torch.rand(1, 9)  # Batch size 1,  features 9
    output = lstm_network(input_data)
    assert output.shape == (1, 3)


def test_ddqnagent_initialization(mock_agent, mock_env):
    assert isinstance(mock_agent, DDQNAgent)
    assert mock_agent.observation_space == mock_env.observation_space.shape[
        0] * mock_env.observation_space.shape[1]
    assert mock_agent.action_space == 9


def test_remember(mock_agent):
    state = np.random.rand(mock_agent.observation_space)
    next_state = np.random.rand(mock_agent.observation_space)
    mock_agent.remember(state, 1, 10, next_state, False)
    assert len(mock_agent.memory) == 1


def test_select_action(mock_agent):
    for epsilon in [1.0, 0.0]:
        mock_agent.epsilon = epsilon
        state = np.random.rand(mock_agent.observation_space)
        action_mask = np.ones(9)
        action, _ = mock_agent.select_action(state, action_mask)
        assert action in range(9)  # Valid action space


def test_replay(mock_agent):
    for _ in range(mock_agent.batch_size):
        state = np.random.rand(mock_agent.observation_space)
        next_state = np.random.rand(mock_agent.observation_space)
        mock_agent.remember(state, 1, 10, next_state, False)

    loss = mock_agent.replay()
    assert loss is not None
    assert isinstance(loss.item(), float)  # Ensure loss is a scalar


def test_train(mock_agent):
    mock_agent.episodes = 1
    mock_agent.epsilon = 1
    rewards, mse_losses, episodes, eval_avg_return = mock_agent.train()
    assert len(rewards) == episodes
    assert len(mse_losses) == episodes


def test_compute_average_return(mock_agent):
    mock_agent.episodes = 1
    average_return = mock_agent.compute_average_return()
    assert isinstance(average_return, float)


def test_update_target_network(mock_agent):
    original_weights = mock_agent.q_network.layer1.weight.clone().detach()
    mock_agent.q_network.layer1.weight.data += 1.0  # Modify weights
    mock_agent.update_target_network()
    updated_weights = mock_agent.target_network.layer1.weight.clone().detach()
    assert torch.equal(original_weights + 1.0, updated_weights)
