# About

In this directory, the necessary scripts to run the demo for Assignment 3 can be found.

- ***prepare_demo.py*** is not for the final demonstration and is merely a helper script to generate the serialized files of the trained agent and recipe mapping. These files are manually added to a GitHub release. The agent is trained based on my custom preferences.  
- ***demo.py*** is the demo application. This starts a web application that either loads the trained agent locally or from the GitHub release — the same procedure applies for the recipe mapping.

The demo ***demo.py*** starts a web application where the user can generate a meal plan for the week. The display shows the currently defined nutrient target values and preferred meals of the trained agent. Additionally, the average nutrient values per day are plotted. To start the application, the user needs to be at the top of the project folder (where the file **Makefile** can be found) and run the command:  

```
make demo
```

If desired, new selections can be defined, and the model can be re-trained and then saved locally. If files are saved locally then they need to be stored in the current ***demo** folder, otherwise the application will not be able to find them.  
