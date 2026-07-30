# Over/Under 2.5 Goals Logistic Regression Model
This is a simple logistic regression notebook to demonstrate understanding of the model building process and the relevant libraries (pandas, scikit-learn, numpy). 

## Data
The data is sourced from football-data.co.uk, where basic match statistics are uploaded weekly.

## Methodology
The methodology flows as follows:
1. Data Downloading and Parsing/Cleaning
Goals, shots and odds data from Europe's top 5 leagues (E0:English Premier League, SP1: Spanish La Liga, D1: German Bundesliga, I1: Italian Serie A, F1: French Ligue 1) is downloaded using the requests library from football-data.co.uk.\n For each league, data from the 05/06 to 25/26 seasons are stored in a dictionary, leagueDict, with the league code as key.
2. Feature Engineering
2 form features are built for each side for each game: average shots per game and average goals per game from the past 5 games. These features are 2 of the most obvious predictors for the number of goals in a match that are also easily accessible from this dataset. 
3. Model Building
A series of logistic regression models are then built using a simple pipeline, where inputs are first standardized before being pushed through the scikit-learn logistic regression function. Each season from 2007/2008 is tested on a model which was trained on all seasons prior. 
4. Model Testing/Plots
The model outputs for each season are then assessed compared to 3 other probabilities: the baseline approach, which just uses a single predicted probability of the observed frequency in the training set, price implied probability, which is the implied probability from the price, and the market implied probability, which uses the normalised (devigged) probability from the price.

## Results
![Log loss and calibration by league](figures/loglossCalibration.png)
 

