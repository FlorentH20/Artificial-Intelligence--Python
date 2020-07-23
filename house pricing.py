import numpy as np
from sklearn import datasets
from sklearn.svm import SVR
from sklearn.metrics import explained_variance_score, mean_squared_error
from sklearn.utils import shuffle
#Load data
data = datasets.load_boston()
#Shuffle data
X, y = shuffle(data.data, data.target, random_state = 7)
#Split the data into training and testing
num_train = int(0.8*len(X))
X_train, y_train = X[:num_train], y[:num_train]
X_test, y_test = X[num_train:], y[num_train:]
#SVR
sv_regressor = SVR(epsilon=0.1, C=1.0, kernel='linear')
sv_regressor.fit(X_train,y_train)
y_test_pred = sv_regressor.predict(X_test)
#Evaluate
mse = mean_squared_error(y_test, y_test_pred)
evs = explained_variance_score(y_test, y_test_pred)
print("Mean squared error: ", round(mse, 2))
print("Explained variance score: ", round(evs, 2))

#Test
test_data = [3.7, 0, 18.4, 1, 0.87, 5.95, 91, 2.5052, 26, 666, 20.2, 351.34, 15.27]
print('\nPredicted Price: ', sv_regressor.predict([test_data])[0])