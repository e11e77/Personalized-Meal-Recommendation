# About

In this directory the necessary scripts to run the demo for assignment 3 can be found.

- ***prepare_demo.py*** is not for the final demonstrantion and is merely a help scrip to generate the serialized files of the trained agent and recipe mapping. These files are manually added to a github release. The agent is trained based on my custom preferences.
- ***demo.py*** the demo application. This starts a web-application that either loads the trained agent locally or from the github release - same procedure for the recipe mapping. 

The demo ***demo.py*** starts a web-application where the user can generate a meal plan for the week. The display shows the currently defined nutrient target values and preferred meals of the trained agent. Additionally, the average nutrient values per day are plotted.

If one wishes new selections can be defined and the model can be re-trained and then saved locally. 