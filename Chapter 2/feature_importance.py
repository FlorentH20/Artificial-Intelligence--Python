import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn import model_selection
from sklearn.utils import shuffle

housing_data = datasets.load_boston()
X, y = shuffle(housing_data.data, housing_data.target, random_state=7)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=.2, random_state=7)
regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=7)
regressor.fit(X_train, y_train)
#Evaluate
y_pred = regressor.predict((X_test))
mse = mean_squared_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
print("\nADABOOSTREGRESSOR: ")
print("Mean Squared Error: ", round(mse, 2))
print("\nExplained Variance Score: ", round(evs, 2))
#Relative feature importance calculation
feature_importance = regressor.feature_importances_
feature_names = housing_data.feature_names
#Normalize the values
feature_importance = 100.0 * (feature_importance/max(feature_importance))
#Sort and flip values
index_sorted = np.flipud(np.argsort(feature_importance))
#Arrange Xticks
pos = np.arange(index_sorted.shape[0]) + 0.5
#Plot the bar graph
plt.figure()
plt.bar(pos, feature_importance[index_sorted], align='center')
plt.xticks(pos, feature_names[index_sorted])
plt.ylabel("Relative importances")
plt.title("Feature importances using ADABOOST")
plt.show()