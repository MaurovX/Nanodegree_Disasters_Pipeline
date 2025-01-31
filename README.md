# Nanodegree_Disasters_Pipeline

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)
 
 ## Installation <a name="installation"></a>
* Developed under Python 3.6
* Anaconda distribution of generic wrangling Pandas, Numpy
* Model, Scikit-learn and NLP libraries from nltk
* Data exports using SQLite, Pickle


 
 ## Project Motivation <a name="motivation"></a>
 
 As a part of the Data Science Nanodegree in Udacity, we are required to develop a small flask app that displays the results of the Disasters dataset. The project should
1. Take both csv files merge, clean, drop duplicates and export clean data to an sqlite database.
2. Import the clean data and build an NLP pipeline for text classification under 36 different categories
3. Deploy the model on a small flask app and display visualizations from the cleaned dataset & model.

The full set of files related to this repo are public and free of use. 

## File Descriptions <a name="files"></a>
The repository structure is defined as follows:
* data folder contains both input .csv files, process.py which can be called from terminal. eg.
  `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
  also contains ETL notebook which displays the logic necessary to build process.py
* model folder contains train_classifier.py which prepares, builds, trains and reports the metrics for the NLP model. Should be called from terminal
eg. `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl` 
it also contains the companion notebook used to build train_classifier.py
* app folder contains run.py used to deploy the flask app using the model built on train_classifier.py in order to run the app 

Run the following command in the app's directory to run your web app.
    `python run.py`

Go to http://0.0.0.0:3001/

## Results<a name="results"></a>

The main findings of the code can be found at each of the companion notebooks

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Code formatted via black. Useful for formatting under PEP8

Must give credit to Udacity and its partners for the data and the amazing DataScience course.  You can find the Licensing for the data and other descriptive information at the udacity link available [here](https://www.udacity.com/course/data-scientist-nanodegree--nd025).  Otherwise, feel free to use the code here as you would like! 
