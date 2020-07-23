import numpy as np
from sklearn import preprocessing
input_data = np.array([[5.1, -2.9, 3.3],
                       [-1.2, 7.8, -6.1],
                       [3.9, 0.4, 2.1],
                       [7.3, -9.9, -4.5]])

#1. BINARIZATION makes numerical values into bool.

data_binarized = preprocessing.Binarizer(threshold = 2.1).transform(input_data)
print("\nBinarized data:\n", data_binarized)
#val>=2.1 are 1 others 0

#2. MEAN REMOVAL removes mean to make mean vector value = 0
print('\nBefore:\n')
print('Mean= ', input_data.mean(axis=0))
print('std deviation = ', input_data.std(axis=0))

print('\nAfter:\n')
data_scaled = preprocessing.scale(input_data)
print('Mean Scaled= ', data_scaled.mean(axis=0))
print('Std deviation scaled= ', data_scaled.std(axis=0))

#3. SCALING- random values are scaled and become understandable for Machine learning to train
#Min Max scaling
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print('\nMin max scaled data: \n', data_scaled_minmax)

#4. NORMALIZATION
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm = 'l2')
print('\nL1 normalized data: \n', data_normalized_l1)
print('\nL2 normalized data: \n', data_normalized_l2)



