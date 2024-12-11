import pandas as pd
import os
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
import re


def read_data() -> pd.DataFrame:
    """
    Reads all CSV files from the 'data' directory and returns the first CSV file found as a pandas DataFrame.

    The function extracts the parent directory of the current directory ('src'). 

    Returns:
        pd.DataFrame: The first CSV file from the 'data' directory as a pandas DataFrame.
    """
    all_data = []
    current_dir = os.path.dirname(__file__)  # os.getdir(__file__)
    data_path = os.path.join(os.path.dirname(current_dir), 'data')
    for file in os.listdir(data_path):
        if file.endswith('.csv'):
            raw_data = pd.read_csv(os.path.join(data_path, file))
            all_data.append(raw_data)
    return all_data[0]


def convert_time_to_minutes(time):
    """
    Converts a time string of the format 'XHYM' into total minutes.

    The function extracts hours (H) and minutes (M) from a time string (e.g., '2H30M') and calculates
    the total time in minutes. If no hours or minutes are provided, it returns zero.

    Args:
        time (str): A string representing time (e.g., '2H30M').

    Returns:
        int: Total time in minutes.
    """
    if not time:
        return 0
    hours = re.search(r'(\d+)H', time)
    minutes = re.search(r'(\d+)M', time)
    if hours:
        total_hours = int(hours.group(0).replace('H', ''))
    else:
        total_hours = 0
    if minutes:
        total_minutes = int(minutes.group(0).replace('M', ''))
    else:
        total_minutes = 0
    return total_minutes + total_hours * 60


def min_max_scale_df(df):
    """
    Scales numeric columns in the DataFrame to a range of [0, 1] using MinMaxScaler.

    This function applies MinMax scaling to various numeric columns in the dataset. 

    Args:
        df (pd.DataFrame): The DataFrame containing the data to be scaled.

    Returns:
        pd.DataFrame: The DataFrame with scaled numeric columns.
    """
    scaled_columns = ['RecipeServings', 'ProteinContent', 'SugarContent',
                      'FiberContent', 'CarbohydrateContent', 'SodiumContent',
                      'CholesterolContent', 'SaturatedFatContent',
                      'FatContent', 'Calories', 'ReviewCount', 'AggregatedRating',
                      'TotalTime', 'PrepTime', 'CookTime']
    min_max_scaler = MinMaxScaler()
    df[scaled_columns] = min_max_scaler.fit_transform(df[scaled_columns])
    return df


def inspect_and_transform_entries_of_df(df: pd.DataFrame) -> None:
    """
    Inspects the DataFrame, removes irrelevant columns, and applies necessary data transformations.

    This function performs several operations:
    - Drops irrelevant columns like 'Name', 'Description', 'RecipeInstructions' (columns that are not needed for analysis)
    - Removes rows with missing values
    - Converts the 'Keywords' column from strings to lists of keywords and filters out certain excluded keywords
    - Applies multi-hot encoding to the 'Keywords' column
    - Transforms time columns ('CookTime', 'PrepTime', 'TotalTime') into minutes
    - Filters out recipes with excessive total time (over 180 minutes)

    Args:
        df (pd.DataFrame): The raw DataFrame to be processed.

    Returns:
        tuple: A tuple containing:
            - recipe_mapping (pd.DataFrame): A DataFrame mapping RecipeId to Name.
            - df_nhot_encoded (pd.DataFrame): The transformed and processed DataFrame.
    """
    pd.set_option('display.max_columns', None)
    # Save Name and RecipeId as a mapping for later use
    recipe_mapping = df[['RecipeId', 'Name']].copy()

    # Drop columns that are not necessary for further analysis or modeling
    df.drop(columns=['Name', 'AuthorId', 'AuthorName',
                     'DatePublished', 'Description', 'Images', 'RecipeCategory',
                     'RecipeIngredientQuantities', 'RecipeIngredientParts', 'RecipeYield', 'RecipeInstructions'
                     ], axis=1, inplace=True)
    print(df.info(verbose=True))
    df_no_nan = df.dropna().reset_index(drop=True)

    # Convert 'Keywords' column from a string of keywords to a list of keywords
    df_no_nan['Keywords'] = df_no_nan['Keywords'].apply(
        lambda x: [item.strip().strip('"') for item in re.findall(r'"(.*?)"', x)])

    # Define excluded keywords that we want to filter out
    excluded_keywords = {'Dessert', 'Breakfast', 'Dinner', 'Toddler Friendly', 'Thanksgiving', "St. Patrick's Day", "Shakes", "Served Hot New Years",
                         "Sauces", "Salad Dressings", "Savory Pies", "Punch Beverage", "Kid Friendly", "Halloween Cocktail", "Savy", "Savory",
                         "Halloween", "Cookie & Brownie", "Christmas", "Chinese New Year", "Birthday", "Beverages", "Baking", "Sweet", "Kosher"}
    keyword_filter = df_no_nan['Keywords'].map(lambda keywords: not any(
        keyword in excluded_keywords for keyword in keywords))
    df_filtered = df_no_nan[keyword_filter]

    # Multi-hot encode the 'Keywords' column (transform the list of keywords into separate binary columns)
    mlb = MultiLabelBinarizer()
    df_nhot_encoded = df_filtered.join(pd.DataFrame(
        mlb.fit_transform(df_filtered['Keywords']), columns=mlb.classes_))
    df_nhot_encoded = df_nhot_encoded.fillna(0)
    df_nhot_encoded.drop(columns='Keywords', axis=1, inplace=True)

    # Convert time columns (CookTime, PrepTime, TotalTime) to minutes
    df_nhot_encoded['CookTime'] = df_nhot_encoded['CookTime'].apply(
        lambda x: convert_time_to_minutes(x))
    df_nhot_encoded['PrepTime'] = df_nhot_encoded['PrepTime'].apply(
        lambda x: convert_time_to_minutes(x))
    df_nhot_encoded['TotalTime'] = df_nhot_encoded['TotalTime'].apply(
        lambda x: convert_time_to_minutes(x))
    # Filter out recipes with total time greater than 180 minutes
    df_nhot_encoded.drop(
        df_nhot_encoded[df_nhot_encoded.TotalTime > 180].index, inplace=True)

    # print(df_nhot_encoded.info(verbose=True))
    return recipe_mapping, df_nhot_encoded
