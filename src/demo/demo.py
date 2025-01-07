import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pickle
import random
import plotly.graph_objs as go
import sys
import requests
import os
import io

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from preprocessing import *
from custom_env import *


def load_model(filename):
    """
    Load a trained model or recipe mapping from a local file or, if not present, download it from a GitHub release.

    Parameters:
        filename (str): The name of the file (either 'meal_planner.sav' or 'recipe_mapping.sav') to load.

    Returns:
        object: The loaded object.

    Raises:
        ValueError: If the model cannot be found locally or from the GitHub release.
    """
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, filename)
    if os.path.exists(file_path):
        # Load the model from the local filesystem
        print(f"Loading file from path: {file_path}")
        return pickle.load(open(file_path, 'rb'))
    else:
        # Construct the release URL for downloading the model
        release_url = f"https://github.com/e11e77/Personalized-Meal-Recommendation/releases/download/v1.0.0/{filename}"
        print(f"Loading file from release: {release_url}")
        response = requests.get(release_url)
        if response.status_code == 200:
            # The file is fetched successfully, now load it
            model_file = io.BytesIO(response.content)
            print("Finished.")
            return pickle.load(model_file)
        else:
            raise ValueError(
                f"Failed to download model from GitHub. Status code: {response.status_code}")


def get_random_meals(agent, recipe_mapping, num_meals=8):
    """
     Get a list of random meal names and IDs based on available recipes.

    Parameters:
        agent (object): The trained agent.
        recipe_mapping (pandas.DataFrame): The recipe mapping dataframe containing the recipe IDs and names.
        num_meals (int, optional): The number of random meals to return. Default is 8.

    Returns:
        list of dict: A list of dictionaries containing the 'label' (meal name) and 'value' (meal ID) for each random meal.

    Raises:
        ValueError: If the requested number of meals exceeds the available recipes in the agent.
    """
    current_preferred_indices = agent.env.user_preference_indices
    available_recipe_ids = random.sample(
        range(0, agent.env.df.shape[0]-1), num_meals-len(current_preferred_indices))
    if num_meals > agent.env.df.shape[0]:
        raise ValueError(
            "Requested number of meals exceeds available recipes.")

    random_meals = []
    for random_index in available_recipe_ids:
        if not recipe_mapping['RecipeId'].isin([random_index]).any():
            continue
        name = recipe_mapping[recipe_mapping['RecipeId']
                              == random_index]['Name'].values[0]
        random_meals.append({'label': name, 'value': int(random_index)})
    
    # Add original preferred meals to the list as well
    current_preferred_indices
    for preferred_index in current_preferred_indices:
        if not recipe_mapping['RecipeId'].isin([preferred_index]).any():
            continue
        name = recipe_mapping[recipe_mapping['RecipeId']
                              == preferred_index]['Name'].values[0]
        random_meals.append({'label': name, 'value': int(preferred_index)})
    return random_meals


def predict_week(agent, recipe_mapping):
    """
    Generate a meal plan for a 7-day week by selecting meals based on the trained agent.

    Parameters:
        agent (object): The trained agent.
        recipe_mapping (pandas.DataFrame): The recipe mapping dataframe used to get meal names by IDs.

    Returns:
        tuple: A tuple containing a list of meal names and their corresponding meal IDs for the week.
    """
    obs, info = agent.env.reset()
    meals = []
    for _ in range(7):  # Simulate 7 days
        action_mask = agent.env.action_masks()
        action, _ = agent.select_action(obs.flatten(), action_mask)
        obs, reward, terminated, truncated, info = agent.env.step(action)
        if terminated:
            meals = obs
            obs, info = agent.env.reset()
    agent.env.close()

    # Map meal IDs to names
    named_meals = []
    for meal in meals:
        name = recipe_mapping[recipe_mapping['RecipeId']
                              == meal[0]]['Name'].values[0]
        named_meals.append(name)
    return named_meals, meals


def extract_nutrient_values(meal_plan, agent):
    """
    Extract the unscaled nutrient values (protein, fiber, saturated fat) for each day in the given meal plan.

    Parameters:
        meal_plan (list of lists): The meal plan for the week where each meal is represented by a list of nutritional data.
        agent (object): The trained agent.

    Returns:
        tuple: A tuple containing three lists: protein, fiber, and saturated fat values for the meal plan.
    """
    # The content values are normalized. To comapare with target values they need to be unscaled.
    def unscaled_value(x, column):
        return (((x - 0) / (1 - 0)) *
                (agent.env.df[column].max() - agent.env.df[column].min())) + agent.env.df[column].min()

    df_columns = np.array(agent.env.df.columns)
    protein_values = []
    fiber_values = []
    saturated_fat_values = []

    # Extract nutrient values
    for day in meal_plan:
        unscaled_protein_content = unscaled_value(
            day[np.where(df_columns == 'ProteinContent')[0][0]], 'ProteinContent')
        unscaled_fiber_content = unscaled_value(
            day[np.where(df_columns == 'FiberContent')[0][0]], 'FiberContent')
        unscaled_saturated_fat_content = unscaled_value(
            day[np.where(df_columns == 'SaturatedFatContent')[0][0]], 'SaturatedFatContent')
        protein_values.append(round(unscaled_protein_content, 1))
        fiber_values.append(round(unscaled_fiber_content, 1))
        saturated_fat_values.append(round(unscaled_saturated_fat_content, 1))

    return protein_values, fiber_values, saturated_fat_values


# ====================================================== INITIALIZE RESOURCES ======================================================
try:
    # Load the trained agent and recipe mapping
    agent = load_model('meal_planner.sav')
    recipe_mapping = load_model('recipe_mapping.sav')
    meal_options = get_random_meals(agent, recipe_mapping, 10)
except Exception as e:
    print(f"Error occurred during initialization : {e}")
    sys.exit(1)

# =================================================== START of DASH APP ==============================================================
# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the app layout
app.layout = html.Div(
    style={'display': 'flex', 'backgroundColor': '#f4f4f9', 'minHeight': '100vh'},
    children=[
        # Left Section
        html.Div(
            style={'flex': '1', 'padding': '20px',
                   'maxWidth': '50%', 'backgroundColor': '#f4f4f9'},
            children=[
                # Title
                html.Div(
                    children=[
                        html.H1("Meal Planner Overview", style={
                                'color': '#155724', 'fontSize': '30px', 'fontFamily': 'Arial'}),
                    ],
                    style={'padding': '10px', 'textAlign': 'center'}
                ),
                # Sliders and Checkboxes
                html.Div(
                    style={'display': 'flex', 'gap': '20px',
                           'justifyContent': 'space-between', 'marginBottom': '20px'},
                    children=[
                        # Sliders Section for Nutrients
                        html.Div(
                            children=[
                                html.H3("Adjust the Desired Content",
                                        style={'color': 'black'}),
                                html.Label("Protein Content (per meal, g)", style={
                                           'fontSize': '14px'}),
                                dcc.Slider(
                                    min=0,
                                    max=50,
                                    step=5,
                                    value=user_dietary_guidelines['ProteinContent'],
                                    marks={i: str(i) for i in range(0, 51, 5)},
                                    id='protein-slider'
                                ),
                                html.Br(),
                                html.Label("Fiber Content (per meal, g)", style={
                                           'fontSize': '14px'}),
                                dcc.Slider(
                                    min=0,
                                    max=50,
                                    step=5,
                                    value=user_dietary_guidelines['FiberContent'],
                                    marks={i: str(i) for i in range(0, 51, 5)},
                                    id='fiber-slider'
                                ),
                                html.Br(),
                                html.Label("Saturated Fat Content (per meal, g)", style={
                                           'fontSize': '14px'}),
                                dcc.Slider(
                                    min=0,
                                    max=50,
                                    step=5,
                                    value=user_dietary_guidelines['SaturatedFatContent'],
                                    marks={i: str(i) for i in range(0, 51, 5)},
                                    id='saturated-fat-slider'
                                ),
                            ],
                            style={'padding': '20px', 'backgroundColor': '#a8d5ba',
                                   'borderRadius': '8px', 'flex': '1'}
                        ),
                        # Checkboxes Section (Random Meal Selection)
                        html.Div(
                            children=[
                                html.H3("Select Preferred Meals",
                                        style={'color': 'black'}),
                                dcc.Checklist(
                                    options=meal_options,
                                    value=[meal['value'] for meal in meal_options if meal['value'] in agent.env.user_preference_indices],  # initially preferred meals stored in env are checked
                                    id='checkboxes',
                                    labelStyle={
                                        'display': 'block', 'marginTop': '10px', 'marginBottom': '5px', 'marginLeft': '5px'},
                                    inputStyle={'marginRight': '10px'}
                                ),
                            ],
                            style={'padding': '20px', 'backgroundColor': '#a8c5e6',
                                   'borderRadius': '8px', 'flex': '1'}
                        ),
                    ]
                ),

                # Buttons 
                html.Div(
                    children=[
                        # Train Model Button
                        html.Button(
                            "Train Model", id='train-button', n_clicks=0,
                            style={
                                'backgroundColor': '#0056b3', 'color': 'white', 'fontSize': '16px',
                                'padding': '10px', 'borderRadius': '8px', 'cursor': 'pointer',
                                'marginRight': '10px', 'marginLeft': '10px'
                            }
                        ),

                        # Meal Planner button
                        html.Button(
                            "Get Weekly Meal Plan", id='predict-button', n_clicks=0,
                            style={
                                'backgroundColor': '#155724', 'color': 'white', 'fontSize': '16px',
                                'padding': '10px', 'borderRadius': '8px', 'cursor': 'pointer',
                                'marginRight': '10px', 'marginLeft': '10px'
                            }
                        ),
                        # Save Model Button
                        html.Button(
                            "Save Model", id='save-button', n_clicks=0,
                            style={
                                'backgroundColor': '#007bff', 'color': 'white', 'fontSize': '16px',
                                'padding': '10px', 'borderRadius': '8px', 'cursor': 'pointer',
                                'marginRight': '10px', 'marginLeft': '10px'
                            }
                        ),
                        # Loading Screen while model is being trained and status message output
                        dcc.Loading(
                            id='loading-train-status',
                            type='default',
                            children=html.Div(
                                id='status-message', style={'marginTop': '10px', 'color': '#007bff'})
                        )
                    ],
                    style={'textAlign': 'center', 'marginBottom': '20px'}
                ),
                # Plot Section (Average Nutrient Content)
                html.Div(
                    dcc.Graph(id='average-nutrient-plot',
                              style={'height': '350px'}),
                    style={'marginTop': '10px'}
                ),
            ]
        ),
        # Right Section (Meal Plan Display)
        html.Div(
            style={'flex': '1', 'padding': '20px',
                   'maxWidth': '50%', 'backgroundColor': '#f4f4f9'},
            children=[
                html.H1("Weekly Meal Plan", style={
                        'color': '#155724', 'fontSize': '30px', 'fontFamily': 'Arial', 'textAlign': 'center'}),
                html.Div(id='meal-output')
            ]
        )
    ]
)


@app.callback(
    Output('status-message', 'children', allow_duplicate=True),
    Input('train-button', 'n_clicks'),
    [
        State('checkboxes', 'value'),
        State('protein-slider', 'value'),
        State('fiber-slider', 'value'),
        State('saturated-fat-slider', 'value')],
    prevent_initial_call=True
)
def train_model(n_clicks, selected_meals, protein_target, fiber_target, saturated_fat_target):
    """
    Trains the agent using the updated dietary guidelines and selected meals.

    Parameters:
        n_clicks (int): The number of times the 'Train Model' button has been clicked.
        selected_meals (list): The list of selected meals chosen by the user.
        protein_target (float): The target protein content for each meal, based on the user's input.
        fiber_target (float): The target fiber content for each meal, based on the user's input.
        saturated_fat_target (float): The target saturated fat content for each meal, based on the user's input.

    Returns:
        str: A message indicating that the model has been trained successfully.
    """
    # Update dietary guidelines based on slider values
    agent.env.user_dietary_guidelines['ProteinContent'] = protein_target
    agent.env.user_dietary_guidelines['FiberContent'] = fiber_target
    agent.env.user_dietary_guidelines['SaturatedFatContent'] = saturated_fat_target

    # Update meal preference based on selected meals
    if len(selected_meals) != 0:
        agent.env.user_preference_indices = selected_meals

    agent.train()
    return "Model Trained!"


@app.callback(
    Output('status-message', 'children', allow_duplicate=True),
    Input('save-button', 'n_clicks'),
    prevent_initial_call=True
)
def save_model(n_clicks):
    """
    Saves the current agent to a local file.

    Parameters:
        n_clicks (int): The number of times the 'Save Model' button has been clicked. Not used.

    Returns:
        str: A message indicating that the model has been saved.
    """
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, 'meal_planner.sav')

    with open(file_path, 'wb') as f:
        pickle.dump(agent, f)
    return "Model Saved!"



@app.callback(
    [Output('meal-output', 'children'),
     Output('average-nutrient-plot', 'figure'),
     Output('status-message', 'children', allow_duplicate=True)],
    [Input('predict-button', 'n_clicks')],
    prevent_initial_call=True
)
def display_meal_plan(n_clicks):
    """
    Generates a weekly meal plan using the trained agent and returns a meal plan with daily meal details,
    including nutrient content and a plot showing average nutrient values.

    Parameters:
        n_clicks (int): The number of times the 'Get Weekly Meal Plan' button has been clicked. Not used.

    Returns:
        tuple: A tuple containing the HTML structure for the meal plan display and the plot figure for the average nutrient values.
    """
    # Generate meal plan for the week and then extract the average nutrient value per day
    meal_plan, meals_id = predict_week(agent, recipe_mapping)
    protein, fiber, saturated_fat = extract_nutrient_values(meals_id, agent)

    days = ['Monday', 'Tuesday', 'Wednesday',
            'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Original color palette for weekdays
    colors = ['#e8f5e9', '#f1f8e9', '#e0f7fa',
              '#f3e5f5', '#fff9c4', '#ffe0b2', '#fce4ec']
    weekday_colors = ['#6aa74e', '#81c784', '#4fc3f7',
                      '#ba68c8', '#fbc02d', '#ffb74d', '#f48fb1']

    # Meal plan sections
    meal_sections = []
    for i, day in enumerate(days):
        meal_sections.append(
            html.Div(
                children=[
                    html.H4(day, style={
                            'margin': '0', 'color': weekday_colors[i], 'fontWeight': 'bold', 'fontSize': '22px'}),
                    html.P(f"{meal_plan[i]}", style={
                           'margin': '5px 0', 'fontWeight': 'bold', 'color': '#333'}),
                    html.P(f"Protein: {protein[i]}g | Fiber: {fiber[i]}g | Saturated Fat: {saturated_fat[i]}g",
                           style={'margin': '5px 0', 'color': '#555'})
                ],
                style={
                    'backgroundColor': colors[i],
                    'padding': '15px', 'borderRadius': '8px', 'marginBottom': '10px',
                    'boxShadow': '0px 4px 6px rgba(0, 0, 0, 0.1)'
                }
            )
        )

    # Create the average nutrient plot
    figure = go.Figure(data=[
        go.Bar(
            x=['Protein', 'Fiber', 'Saturated Fat'],
            y=[sum(protein)/7, sum(fiber)/7, sum(saturated_fat)/7],
            marker={'color': ['#388e3c', '#fbc02d', '#e53935']}
        )
    ])
    figure.update_layout(
        title="Average Nutrient Content Per Day",
        xaxis_title="Nutrient",
        yaxis_title="Average Content (g)",
        plot_bgcolor='#f4f4f9'
    )

    return meal_sections, figure, ""


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
