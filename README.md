# Personalized-Meal-Recommendation

This repository will be used for the programming project of the lecture Applied Deep Learning WS24/25. The README gives an overview of the chosen project and the estimated road map. 

## About
 As a topic, I have decided on a **Deep Reinforcement Learning** topic -  specifically, as mentioned above, a **Personalized Meal Recommendation**. 
 The project type is **bring your own method** as I plan to use existing datasets and alter existing implementations to suit my project and improve results. 

The idea to use Deep Reinforcement Learning for my project was taken from the two papers ***Delighting Palates with AI*** and ***Recipe Rennaissance*** which use Deep Q-Learning and Deterministic Policy Gradient respectively.
-  Amiri, M.; Sarani Rad, F.; Li, J. Delighting Palates with AI: Reinforcement Learning’s Triumph in Crafting Personalized Meal Plans with High User Acceptance. _Nutrients_ **2024**, _16_, 346. https://doi.org/10.3390/nu16030346
- Neha Tyagi; Vijay Maheshwari. Recipe Renaissance: Leveraging Deep Reinforcement Learning for Food
Recommendations Systems in Culinary Innovation. Volume: 11 Issue: 05 | May 2024. https://www.irjet.net/archives/V11/i5/IRJET-V11I5243.pdf

The goal of this project is to build an application that recommends recipes for at least a week based on personal preferences. The recommended meals should be balanced, operate under budget constraints and cater to specific dietary constrictions (e.g. vegetarian, etc.).

The structure of this application should be to provide input for the user to share personal preferences. The model will then give meal recommendations for at least a week. To increase the personal user acceptance of the model, the user should be able to declare unacceptable meals and combinations as well as favorite recipes from the recommendations. With this information, the model can be fine-tuned to the specific user. 

The approach for this project is to use Deep Reinforcement Learning with a **Policy Gradient Algorithm**. As a starting point, the **Proximal Policy Optimization** will be taken to solve the problem and improve recommendations. Furthermore, as a first step, the following datasets will need to be processed and combined. 

## Datasets

In this section, I will briefly describe the datasets that I am about to use. 

The main dataset will be an extensive recipe dataset with crawled data from Food.com. It holds over 500K recipes and is around 800 MB in size. Altogether there are 10 columns. The data contains categorical values. This will require N-hot encoding in the preprocessing steps. Each recipe has a list of recipe ingredients in words as well as further metadata regarding the recipe. Additionally, the dataset contains search queries that return the recipe and tags assigned by the users to the recipe. 
- https://www.kaggle.com/datasets/shuyangli94/foodcom-recipes-with-search-terms-and-tags

This dataset has been used/derived in the following papers:
- **SHARE: a System for Hierarchical Assistive Recipe Editing**  
Shuyang Li, Yufei Li, Jianmo Ni, Julian McAuley  
arXiv, 2021  
[https://arxiv.org/pdf/2105.08185.pdf](https://arxiv.org/pdf/2105.08185.pdf)
- **Generating Personalized Recipes from Historical User Preferences**  
Bodhisattwa Prasad Majumder*, Shuyang Li*, Jianmo Ni, Julian McAuley  
EMNLP, 2019  
[https://www.aclweb.org/anthology/D19-1613/](https://www.aclweb.org/anthology/D19-1613/)

For this project, I would like the application to recommend meals based on a budget that should be ideally catered to Austria. In 2020 the magazine **Dossier** published an article about the online product prices in the local Austrian supermarkets Billa and Interspar. The article provides the raw data as downloadable files.
- https://www.dossier.at/dossiers/supermaerkte/quellen/anatomie-eines-supermarkts-die-methodik/

The files contain a list of products with respective prices and categories. An important step in the preprocessing phase will be to map the local products with their price to the respective ingredients in the above recipe dataset. Additionally, the supermarket data also contains discounted and special event products. Such cases will need to be excluded since the application should look at generalized product prices independent of the current time and event.

## Work breakdown

Breakdown structure for the individual tasks with time estimates:

 - Dataset collection: 1 hour
 - Dataset preparation, processing and transformation: 10 hours
 - Designing and building an appropriate network: 28 hours 
 - Training and fine-tuning that network: 20 hours
 - Building an application to present the results: 20 hours
 - Writing the final report: 10 hours
 - Preparing the presentation of my work: 4 hours
