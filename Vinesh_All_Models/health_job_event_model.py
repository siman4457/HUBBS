import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import binary_crossentropy
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

correctly_predicted = 0
data = pd.read_csv("Non_uniform_data_Job_event_mgt_feature_final.csv", header=0)

# sns.countplot(x="event_mgt", data=data)
# plt.show()

cols = [0, 70, 71]  # non-feature columns
X = data.drop(data.columns[cols], axis=1)
y = data["event_mgt"]

# X_test = X.iloc[0:1, 0:69]
# X_train = X.drop(0)
# print(X_test)
for i in range(0, len(y)):
    X_test = X.iloc[i:(i+1), 0:69] # selects a single row as data frame
    X_train = X.drop([i])
    y_test = y[i]
    y_train = y.drop(i)
    model = Sequential([Dense(69, input_dim=69, activation='relu'),
                        Dense(1, activation="sigmoid")
                        ])
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=10, epochs=5, shuffle=True, verbose=1)
    # _, accuracy = model.evaluate(X_test, y_test)
    # print('Accuracy: %.2f' % (accuracy * 100))
    predictions = model.predict(X_test)
    for p in predictions:
        if p == y_test:
            correctly_predicted += 1
print(correctly_predicted)




