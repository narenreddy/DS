# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os


os.chdir("C:\\Users\\NarendraReddyVassipa\\Dropbox\\Technical\\Datascience ML\\Heroku-Deployment")
os.listdir()

dataset = pd.read_csv('hiring.csv')

# Replace NaN in experience with 0
dataset['experience'].fillna(0, inplace=True)
# Replace NaN in test_score with Average of test_score data
dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

# iloc indexer for Pandas Dataframe is used for integer-location based indexing / selection by position.
# Select all independent features
X = dataset.iloc[:, :3]

#Converting words to integer values in experience column
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

# Dependet feature Salary
y = dataset.iloc[:, -1]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results. We can deploy this model any where.
model = pickle.load(open('model.pkl','rb'))
# Predict salary based on experience, test_score & interview_score
print(model.predict([[2, 9, 6]]))
