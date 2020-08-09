import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection, preprocessing
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import classification_report
#Loading the input data
input_data = 'traffic_data.txt'
data = []
with open(input_data, 'r') as f:
    for line in f.readlines():
        items = line[:-1].split(',')
        data.append(items)
data = np.array(data)

#Encoding the non-numerical values
label_encoder = []
X_encoded = np.empty(data.shape)
for i, item in enumerate(data[0]):
    if item.isdigit():
        X_encoded[:, i] = data[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(data[:, i])
X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)
#Spliting the data into train and test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state = 5)
#Training Extremely random forest classifier
params = {'n_estimators': 100, 'max_depth':4, 'random_state':0}
regressor = ExtraTreesRegressor(**params)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print("Mean absolute error: ", round(mean_absolute_error(y_test, y_pred), 2))
#Computing the output of encoded values
test_datapoint = ["Saturday", '10:20', 'no']
test_datapoint_encoded = [-1] * len(test_datapoint)
count = 0
test_datapoint = ['Saturday', '10:20', 'Atlanta', 'no']
test_datapoint_encoded = [-1] * len(test_datapoint)
count = 0
for i, item in enumerate(test_datapoint):
    if item.isdigit():
        test_datapoint_encoded[i] = int(test_datapoint[i])
    else:
        test_datapoint_encoded[i] = int(label_encoder[count].transform(test_datapoint[i]))
        count = count + 1

test_datapoint_encoded = np.array(test_datapoint_encoded)

print("Predict the traffic: ", int(regressor.predict([test_datapoint_encoded])[0]))


