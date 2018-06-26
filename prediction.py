import pandas as pd
import numpy as np
from sklearn import linear_model

def smoothGaussian(column, column_name, degree=5):
    list = column.tolist()
    window = degree * 2 - 1
    weight = np.array([1.0] * window)
    weightGauss=[]
    for i in range(window):
        i = i - degree + 1
        frac = i / float(window)
        gauss = 1 / (np.exp((4 * (frac)) ** 2))
        weightGauss.append(gauss)
    weight=np.array(weightGauss) * weight
    smoothed=[0.0] * (len(list) - window)
    for i in range(len(smoothed)):
        smoothed[i] = sum(np.array(list[i:i + window]) * weight) / sum(weight)
    return pd.DataFrame({column_name:smoothed})

# Read all csv files
dengue_features_train = pd.read_csv("dengue_features_train.csv")
dengue_labels_train = pd.read_csv("dengue_labels_train.csv")
dengue_features_test = pd.read_csv("dengue_features_test.csv")
submission_format = pd.read_csv("submission_format.csv")

# Extracting independent variable columns
x_train = dengue_features_train.iloc[:,4:]

# Extracting dependent variable column
y_train = dengue_labels_train[["total_cases"]]

# Concatenating independent and dependent variable columns into one dataframe
dataset = pd.concat([x_train, y_train], axis=1)

# Interpolation of null values of the whole dataframe
dataset = dataset.interpolate()

#new_dataframe = pd.DataFrame()
#for index in range(len(dataset.columns)):
#    column_name = dataset.columns[index]
#    new_dataframe = pd.concat([new_dataframe, smoothGaussian(dataset.iloc[:, index], column_name)], axis=1)

a = dataset.rolling(5, center=False).mean()
a = a.dropna()

dataset = a

# Selecting independent and dependent variable columns after interpolation
x_train = dataset.iloc[:,0:-1]
y_train = dataset[["total_cases"]]

# Selecting test data values
test_data = dengue_features_test.iloc[:,4:]

# Interpolation of null values present in test data
test_data = test_data.interpolate()

# Creating a linear regression model
regression = linear_model.LinearRegression()

# Training the model with data
regression.fit(x_train, y_train)

# Predicting the new values from the model and rounds them to the nearest integer
predictions = regression.predict(test_data).round()

# Convert the prediction into a dataframe and naming the column as 'total_cases'
predictions = pd.DataFrame(predictions)
predictions.columns = ["total_cases"]

# Change the prediction value type as 'int64'
predictions["total_cases"] = predictions["total_cases"].astype(np.int64)

# Drop the total_cases column from the submission dataframe
submission_format.drop(["total_cases"], axis=1, inplace=True)

# Concatenation of submission dataframe with the predicted column
submission_format = pd.concat([submission_format, predictions], axis=1)

# Produce the csv of the submission dataframe as 'out.csv'
submission_format.to_csv("out.csv", index=False)
