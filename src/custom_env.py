import numpy as np
import gymnasium as gym
from preprocessing import *
from stable_baselines3.common.env_checker import check_env

# Predefined dietary restrictions/goals
# These values represent the approximate target nutrient intake
user_dietary_guidelines = {
    'ProteinContent': 50/3,         # Protein target per meal
    'FiberContent': 25/3,           # Fiber target per meal
    'SugarContent': 25/3,           # Sugar target per meal
    'Calories': 800,                # Calorie target limit per meal
    'CarbohydrateContent': 350,     # Daily carbohydrate target
    # Sodium limit per meal (max 2000 mg per day)
    'SodiumContent': 2000/3,
    # Cholesterol limit per meal (max 300 mg per day)
    'CholesterolContent': 300/3,
    # Saturated fat limit pear meal (max 20 g per day)
    'SaturatedFatContent': 20,
    'FatContent': 40/3              # Fat target per meal
}


class MealPlannerEnv(gym.Env):
    """
    A custom environment for meal planning. The environment allows 
    for the selection of meals over a week based on user dietary preferences and restrictions.

    Attributes:
        df (pd.DataFrame): Unscaled meal dataset.
        scaled_df (pd.DataFrame): Min-max scaled version of the meal dataset.
        size_meals (int): Number of available meals in the dataset.
        num_features (int): Number of features describing each meal.
        days_in_week (int): Fixed number of days for planning (7).
        current_day (int): Tracks the current day in the planning process.
        user_preference_indices (list): List of meal indices preferred by the user.
        user_dietary_guidelines (dict): Nutritional targets and limits set by the user.
        action_space (gym.spaces.Discrete): Discrete space for selecting meal indices.
        observation_space (gym.spaces.Box): Continuous space representing the state of the environment.
        state (np.ndarray): Current state of meal selection over the week.
        invalid_action_mask (np.ndarray): Array indicating valid/invalid actions based on selections.
    """

    def __init__(self, df_meals: pd.DataFrame, user_dietary_guidelines, user_selected_meals=None):
        """
        Initializes the MealPlannerEnv.

        Parameters:
        - df_meals (pd.DataFrame): DataFrame containing meal data.
        - user_dietary_guidelines (dict): User-defined dietary targets.
        - user_selected_meals (list): Indices of meals preferred by the user (optional).
        """
        self.df = df_meals
        self.scaled_df = min_max_scale_df(df_meals.copy())
        self.size_meals = df_meals.shape[0]     # Total number of meals
        self.num_features = df_meals.shape[1]
        self.days_in_week = 7                   # Fixed planning period
        self.current_day = 0
        self.user_preference_indices = user_selected_meals
        self.user_dietary_guidelines = user_dietary_guidelines

        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(self.size_meals)
        self.observation_space = gym.spaces.Box(0, self.size_meals, shape=(
            self.days_in_week, self.num_features), dtype=np.float64)
        # Initialize state and action mask
        self.state = np.zeros(
            (self.days_in_week, self.num_features), dtype=np.float64)
        self.invalid_action_mask = np.ones(self.size_meals, dtype=np.int64)

    def reset(self, seed=None):
        """
        Resets the environment to its initial state.

        Parameters:
        - seed (int): Optional seed for randomization.

        Returns:
        - tuple: Initial observation (state) and an empty dictionary (could have been used for additional info but not relevant in this implementation).
        """
        self.current_day = 0
        self.state = np.zeros(
            (self.days_in_week, self.num_features), dtype=np.float64)
        self.invalid_action_mask = np.ones(self.size_meals, dtype=np.int64)
        return self.state, {}

    def step(self, action):
        """
        Takes an action (meal selection), updates the environment state, and calculates a reward.

        Parameters:
        - action (int): Index of the selected meal.

        Returns:
        - tuple: Updated state, reward, termination flag, truncation flag, and additional info (empty, not used).
        """
        if self.current_day == self.days_in_week - 1:
            terminated = True
        else:
            terminated = False
        # Update state with the selected meal
        self.state[self.current_day] = self.scaled_df.iloc[action].values
        # Selected meals get probability of 0 for future selection
        self.invalid_action_mask[action] = 0
        reward = self.calculate_reward(terminated)

        self.current_day += 1
        truncated = False       # Truncation not applicable
        return self.state, reward, terminated, truncated, {}

    def action_masks(self):
        """
        Provides a mask for invalid actions (meals that have already been selected). 

        The probability of taking an action done by the q-network is multiplied with this mask. 
        Previously taken actions (invalid actions) are assigned a probabilty of 0 and cannot be selected anymore. 

        Returns:
        - np.ndarray: Mask array (1 for valid actions, 0 for invalid actions).
        """
        return self.invalid_action_mask

    def calculate_keyword_reward(self, meal_features):
        """
        Calculates the reward based on matching keywords between the selected meal and 
        user-preferred meals.

        Parameters:
        - meal_features (np.ndarray): Features of the selected meal.

        Returns:
        - float: Number of matching keywords divided by the total number of keywords liked by the user.
        """
        keyword_columns = self.df.columns[self.df.columns.get_loc(
            'RecipeServings') + 1:]

        # Aggregate keywords from user-preferred meals
        liked_meals = self.df.iloc[self.user_preference_indices]
        liked_keywords = liked_meals[keyword_columns].sum(axis=0)

        # Match keywords for the selected meal
        selected_keywords = pd.Series(meal_features, index=self.df.columns)[
            keyword_columns]
        matching_keywords = (liked_keywords * selected_keywords).sum()
        max_possible_keywords = liked_keywords.sum()

        if max_possible_keywords == 0:
            return 0

        reward = matching_keywords / max_possible_keywords
        return reward

    def calculate_reward(self, terminated):
        """
        Calculates the reward based on nutrient adherence and user preferences.

        Parameters:
        - terminated (bool): Whether the episode has ended.

        Returns:
        - float: Total reward for the current step.
        """

        def unscaled_value(x, column): return (((x - 0) / (1 - 0)) *
                                               (self.df[column].max() - self.df[column].min())) + self.df[column].min()
        reward = 0
        if terminated:
            # Calculate average nutrient values for the week
            avg_protein = 0
            avg_fiber = 0
            avg_saturated_fat = 0
            for meal_features in self.state:
                avg_protein += unscaled_value(
                    meal_features[self.df.columns.get_loc('ProteinContent')], 'ProteinContent')
                avg_fiber += unscaled_value(
                    meal_features[self.df.columns.get_loc('FiberContent')], 'FiberContent')
                avg_saturated_fat += unscaled_value(meal_features[self.df.columns.get_loc(
                    'SaturatedFatContent')], 'SaturatedFatContent')

            # Compare averages with user dietary guidelines
            if avg_protein/7 > self.user_dietary_guidelines["ProteinContent"] - 5:
                reward += 3
            if avg_fiber/7 > self.user_dietary_guidelines["FiberContent"] - 5:
                reward += 1
            if avg_saturated_fat/7 < self.user_dietary_guidelines["SaturatedFatContent"] - 5:
                reward += 1
        else:
            # Reward based on daily nutrient adherence
            protein_content = unscaled_value(
                self.state[self.current_day][self.df.columns.get_loc('ProteinContent')], 'ProteinContent')
            fiber_content = unscaled_value(
                self.state[self.current_day][self.df.columns.get_loc('FiberContent')], 'FiberContent')
            saturated_fat_content = unscaled_value(self.state[self.current_day][self.df.columns.get_loc(
                'SaturatedFatContent')], 'SaturatedFatContent')
            if protein_content/7 > self.user_dietary_guidelines["ProteinContent"] - 5:
                reward += 5
            if fiber_content/7 > self.user_dietary_guidelines["FiberContent"] - 5:
                reward += 5
            if saturated_fat_content/7 < self.user_dietary_guidelines["SaturatedFatContent"] - 5:
                reward += 5
            if self.user_preference_indices:
                reward += self.calculate_keyword_reward(
                    self.state[self.current_day])
        return reward
