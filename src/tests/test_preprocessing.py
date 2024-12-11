import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing import (
    convert_time_to_minutes,
    min_max_scale_df,
    inspect_and_transform_entries_of_df,
    read_data
)


@pytest.fixture
def mock_df():
    data = {
        'RecipeId': [1, 2],
        'Name': ['Recipe 1', 'Recipe 2'],
        'AuthorId': [10, 20],
        'AuthorName': ['Author 1', 'Author 2'],
        'DatePublished': ['2023-01-01', '2023-01-02'],
        'Description': ['Delicious recipe', 'Tasty meal'],
        'Images': ['image1.png', 'image2.png'],
        'RecipeCategory': ['Main', 'Dessert'],
        'RecipeIngredientQuantities': ['1 cup', '2 tbsp'],
        'RecipeIngredientParts': ['Ingredient 1', 'Ingredient 2'],
        'RecipeYield': ['4 servings', '2 servings'],
        'RecipeInstructions': ['Step 1', 'Step 2'],
        'Keywords': ['"Quick" "Easy"', '"Healthy" "Vegan"'],
        'RecipeServings': [4, 2],
        'ProteinContent': [20, 10],
        'SugarContent': [5, 15],
        'FiberContent': [3, 2],
        'CarbohydrateContent': [50, 40],
        'SodiumContent': [500, 600],
        'CholesterolContent': [10, 20],
        'SaturatedFatContent': [5, 10],
        'FatContent': [10, 20],
        'Calories': [200, 300],
        'ReviewCount': [50, 100],
        'AggregatedRating': [4.5, 3.8],
        'TotalTime': ["1H30M", "45M"],
        'PrepTime': ["30M", "15M"],
        'CookTime': ["1H", "30M"]
    }
    return pd.DataFrame(data)

@pytest.mark.parametrize("time_string, expected", [
    ("1H30M", 90),
    ("2H", 120),
    ("45M", 45),
    ("", 0),
    (None, 0)
])
def test_convert_time_to_minutes(time_string, expected):
    assert convert_time_to_minutes(time_string) == expected

def test_read_data():
    data = read_data()
    assert isinstance(data, pd.DataFrame)


def test_min_max_scale_df(mock_df):
    mock_df['TotalTime'] = 5
    mock_df['PrepTime'] = 2
    mock_df['CookTime'] = 3
    scaled_df = min_max_scale_df(mock_df)
    for col in [
        'RecipeServings', 'ProteinContent', 'SugarContent',
        'FiberContent', 'CarbohydrateContent', 'SodiumContent',
        'CholesterolContent', 'SaturatedFatContent', 'FatContent',
        'Calories', 'ReviewCount', 'AggregatedRating',
        'TotalTime', 'PrepTime', 'CookTime'
    ]:
        assert scaled_df[col].min() >= 0
        assert scaled_df[col].max() <= 1


def test_inspect_and_transform_entries_of_df(mock_df):
    recipe_mapping, transformed_df = inspect_and_transform_entries_of_df(mock_df)

    # Check recipe mapping
    assert isinstance(recipe_mapping, pd.DataFrame)
    assert list(recipe_mapping.columns) == ['RecipeId', 'Name']

    # Check transformed DataFrame
    assert 'Keywords' not in transformed_df.columns  # Ensure Keywords are encoded
    assert 'Quick' in transformed_df.columns  # Ensure MultiLabelBinarizer worked
    assert 'Healthy' in transformed_df.columns
    assert 'Vegan' in transformed_df.columns
    assert transformed_df['Quick'].sum() == 1
    assert transformed_df['Healthy'].sum() == 1
    assert transformed_df['Vegan'].sum() == 1

    # Check time columns transformed
    assert transformed_df['TotalTime'].dtype == np.int64
    assert transformed_df['PrepTime'].dtype == np.int64
    assert transformed_df['CookTime'].dtype == np.int64

    # Check dropped rows with TotalTime > 180
    assert all(transformed_df['TotalTime'] <= 180)


def test_inspect_transform_empty_df():
    empty_df = pd.DataFrame()
    with pytest.raises(KeyError):  # Expecting error due to missing columns
        inspect_and_transform_entries_of_df(empty_df)
