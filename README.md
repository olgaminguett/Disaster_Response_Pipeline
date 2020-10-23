# Disaster Response Pipeline

# Description

This Project is part of Data Science Nanodegree Program by Udacity in partnership with Figure Eight. The dataset
contains tweets and messages from real-life disaster. The aim of the project is to build a Natural Language
Processing  (NLP) tool that categorize messages.

# Files

The Project is divided in the following Sections:

* Data Processing, ETL Pipeline to extract data from source, clean data and save them in a proper database structure
* Machine Learning Pipeline to train a model able to classify text message in categories
* Web App to show model results in real time.


# Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

# License

License: MIT

# Acknowledgements

   Figure Eight for providing the dataset
   Udacity for providing advice and review through the Data Science Nano Degree

Projects develop while studying for Udacity Nanodegrees:
- [Data Science Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025)
- [Data Analyst Nanodegree](https://www.udacity.com/course/data-analyst-nanodegree--nd002)

## Projects

- [Disaster Response Pipeline](./Disaster_Response_Pipeline/README.md) 
- [Missing Migrants Project](./MissingMigrantsProject/README.md)
- [Suicide Rates Overview](./Suicide-Rates-Overviewt/README.md)
- [Write a Data Science Blog Post](./Write-a-Data-Science-Blog-Post/README.md)
- [Recommendations with IBM](./Recommendations-with-IBM/README.md)
   
