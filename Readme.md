## Table of Contents
1. [Installation](#installation)
2. [Project Motivation](#project-motivation)
3. [Instructions](#instructions)
5. [Licensing](#licensing)

# Installation
There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. No issues should be faced running the code using Python version 3.*.

# Project Motivation
The aim of the project was to build a web app that analyzes text messages and classifies them into categories in order to easier organize help in case of natural disasters. 
The aim of the project can be divided into three steps:
1. creating an ETL-pipeline that loads data from two CSV files, merges and cleans data and stores it into a SQLite database.
2. creating a ML-pipeline that processes text taken from the database and performs a multi-output classsification
3. deploying the trained model in a Flask web app where new messages can be input and will be classified by the model into different categories (or not).

# Instructions
This instructions were originally made by the team at Udacity's.

1. Run the following commands in the project's root directory to set up your database and model.
- To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
- To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
2. Run the following command in the app's directory to run your web app. python run.py


# Licensing
Data from [Appen](https://appen.com/) (formerly FigureEight) was used. 