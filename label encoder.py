import numpy as np
from sklearn import preprocessing
input_labels = ['red', 'black', 'green', 'yellow', 'blue', 'black', 'white']
#1. LABEL ENCODER AND FITTING LABELS
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)
print('\nLabel maping: ')
for i, item in enumerate(encoder.classes_):
    print(item, '-->', i)
test_labels = ['red', 'green', 'blue']
encoded_values = encoder.transform(test_labels)
print('\nLabels= ', test_labels)
print('\nEncoded labels= ', list(encoded_values))
#DECODING
encoded_values = [3, 0, 4, 1]
decode_list = encoder.inverse_transform(encoded_values)
print('\nEncoded values = ', encoded_values)
print('\nDecoded values = ', decode_list)


