# Personalized-Meal-Recommendation
This repository will be used for the programming project of the lecture Applied Deep Learning WS24/25. The README gives an overview of the chosen project and the road map. 

## How to run
This project is written in Python and contains a **requirements.txt** file containing all necessary libraries. The specific python version used for development is:
- ***Python 3.8.10***
The development is done in Ubuntu (WSL 2) under Windows 10. 

A makefile has been provided for ease of running the application. By changing into the directory with **Makefile** the user can download all necessary libraries with the command:
```
make install
```
The file ***recipes.csv*** needs to exist in the directory **data**, otherwise the application cannot be run successfully. To run the entire end-to-end pipeline containing tests, preprocessing and postprocessing the following commmand can be used:
```
make run
```
Only running the tests can be done with:
```
make test
```

To start the demo application for this project:
```
make demo
```

## Changes in Development
During the development of this project, several adjustments were made compared to the initial goals and plans. These changes were driven by practical challenges and limitations encountered during implementation.

### Budget Constraints
The **Dossier** article on Austrian supermarket prices was considered as a potential data source for product pricing:
- [Dossier Article: Supermarket Prices](https://www.dossier.at/dossiers/supermaerkte/quellen/anatomie-eines-supermarkts-die-methodik/)

However, the plan to incorporate budget constraints was abandoned due to several issues:
- **Inconsistent Pricing**: Variability in product prices, such as multiple price options for the same item and inconsistent pricing per quantity.
- **Unit Mismatch**: The recipe dataset did not include ingredient quantities in standardized units (e.g., kilograms), while the price dataset did. Resolving discrepancies like "2 apples in a recipe" versus "1 kg of apples costs X" would introduce too much impreciseness and could make results unreliable.

### Recipe Dataset
The dataset was updated to ensure the inclusion of nutritional information necessary for the project’s goals of adhering to dietary restrictions.  
- **Final Dataset**: [FoodCom Recipes and Reviews](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews?select=recipes.csv)  
- **Initial Dataset**: [Outdated FoodCom Dataset](https://www.kaggle.com/datasets/shuyangli94/foodcom-recipes-with-search-terms-and-tags)

### Algorithm Changes
The initial approach using **Proximal Policy Optimization (PPO)** was replaced with **Double Deep Q-Learning (DDQN)** due to several factors:
- **PPO Challenges**: Implementing PPO in the custom meal environment proved overly complex and infeasible. Adjustments for specific constraints and rewards were difficult to integrate.

The environment requires that a meal be selected only once per episode, as repeated meals within a week are not desirable. Being able to implement the DDQN algorithm without issues from scratch provided greater control and made it easier to incorporate invalid action masking. Negative rewards for invalid actions did not effectively discourage repeated selections, causing the agent to get stuck. Therefore, an invalid action mask was introduced, which directly prevented the agent from reselecting meals within the same episode, successfully resolving the issue.

## Error Metric
The goal of this deep reinforcement model is to create a balanced weekly meal plan that adhers to user dietary guidelines. Therefore, custom error metrics are defined to adhere to this goal. The success of the model is primarily measured by how well the meals selected over the week meet specific nutritional requirements, with a focus on protein, fiber, and saturated fat content.

### Success Rate
The **success rate** measures the fraction of episodes where the average content of protein, fiber, and saturated fat across the selected meals meets the user’s dietary guidelines within a specified threshold. For fiber and protein content, their average values should reach a desired threshold, while the average values for saturated fat content need to fall below a target value. Ideally, the agent should achieve a **90% success rate** for each nutrient, meaning at least 90% of episodes should see the average content fit the specified targets. While saturated fat content performs well on average — being able to even reach the required threshold in some runs — fiber content and protein content generally fluctuate in performance. Runs where the protein content is able to achieve over a 70% success rate lead to the success rate for fiber content massively underperforming, sitting around a 10% success rate, and vice versa. This can be attributed to the meal selection in the dataset. Meals that are usually rich in protein lack fiber. Considering this observation, the specified target thresholds for protein and fiber might be chosen too high compared to the values in the dataset.

Further metrics that are used to help evaluate the model are the following two listed below. However, it is important to note that ultimately, the success rate determines the final performance of the model.
### Average Return
The **average return** reflects the overall performance of the agent by measuring the rewards accumulated across episodes. This metric helps assess the stability of the learning process but is not necessarily a significant influence when analyzing the error metric. The agent is expected to converge to a consistent and high average return, but occasional spikes in reward indicate some instability in the network's learning.

### MSE - Loss Function
The ***DDQN*** agent uses the ***mean squared error*** for the loss function during training. 

## Work breakdown
Breakdown structure for the individual tasks during development and their evaluated time:

 - Dataset collection: 1 hour
 - Dataset preparation, processing and transformation: 6 hours

 - Designing and building an appropriate network: 22 hours 
 - Training and fine-tuning that network: 14 hours

  - Building an application to present the results: 15 hours

Breakdown structure and estimated time for tasks still to be done in assignment 3:
 - Writing the final report: 10 hours
 - Preparing the presentation of my work: 4 hours

## About
As a topic, I have decided on a **Deep Reinforcement Learning** topic -  specifically, as mentioned above, a **Personalized Meal Recommendation**. 
The project type is **bring your own method** as I use existing datasets and alter existing implementations to suit my project and improve results. 
The project uses the **Double Deep Q Learning** algorithm.

The idea to use Deep Reinforcement Learning for my project was taken from the two papers ***Delighting Palates with AI*** and ***Recipe Rennaissance*** which use Deep Q-Learning and Deterministic Policy Gradient respectively.
-  Amiri, M.; Sarani Rad, F.; Li, J. Delighting Palates with AI: Reinforcement Learning’s Triumph in Crafting Personalized Meal Plans with High User Acceptance. _Nutrients_ **2024**, _16_, 346. https://doi.org/10.3390/nu16030346
- Neha Tyagi; Vijay Maheshwari. Recipe Renaissance: Leveraging Deep Reinforcement Learning for Food
Recommendations Systems in Culinary Innovation. Volume: 11 Issue: 05 | May 2024. https://www.irjet.net/archives/V11/i5/IRJET-V11I5243.pdf

### Demo
For the demo, a web application has been created. This application recommends recipes for at least a week based on stored personal preferences. The necessary data is fetched from a GitHub release if no local data can be found. (Files need to be saved in the ***demo*** folder within the ***src*** directory) 

The model will then give meal recommendations for at least a week. To increase the personal user acceptance of the model, the user can declare favorite recipes and adjust target nutrient content. With this information, the model can be re-trained and fine-tuned to the specific user.

The coding logic can be found in the ***demo*** folder within the ***src*** directory. The script ***prepare_demo.py*** is used during development to initialize the custom trained model. For the final demonstration the user only needs the script ***demo.py***.

## Datasets
During the development the recipe dataset underwent some changes. The final dataset is:
- https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews?select=recipes.csv

This change was done due to the old dataset lacking nutritional information (such as proteint content, etc.). 

The dataset in current use is an extensive recipe dataset with crawled data from Food.com. It contains several files, however, the one of interest that is used for this application is ***recipes.csv*** the other files are not included.

The file holds over 500K recipes and is around 700 MB in size. Altogether there are 28 columns, containing categorical values and also missing values. Each recipe has a list of recipe ingredients in words as well as further metadata regarding the recipe. Additionally, the dataset contains search queries that return the recipe and tags assigned by the users to the recipe. 
