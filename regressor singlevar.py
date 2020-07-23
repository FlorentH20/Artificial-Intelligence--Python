import pickle
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import sklearn.metrics as sm

input_file = 'data_singlevar_regr.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]
num_train= int(0.8*len(X))
num_test = len(X) - num_train

X_train, y_train = X[:num_train], y[:num_train]
X_test, y_test = X[num_train:], y[num_train:]

regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)

y_test_pred = regressor.predict(X_test)

plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_test_pred, color='black', linewidth=4)
plt.xticks(())
plt.yticks(())
plt.show()

#Evaluating the model
print("Linear Regression Performance:")
print("Mean absolute error: ", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error: ", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error: ", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain variance score: ", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score", round(sm.r2_score(y_test, y_test_pred), 2))

#When the model is created we can save it to use it later

output_model_file = 'model.pkl'
with open(output_model_file, 'wb') as f:
    pickle.dump(regressor, f)
#load the model from the file on the disk and perform prediction
with open(output_model_file, 'rb') as f:
    regressor_model = pickle.load(f)

y_test_pred_new = regressor_model.predict(X_test)
print("\n\nNew mean absolute error: ", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))

