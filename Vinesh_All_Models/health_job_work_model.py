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

data = pd.read_csv("Non_uniform_data_Job_work_mgt_feature_final.csv", header=0)
sns.countplot(x="work_mgt", data=data)
plt.show()
cols = [0, 70, 71]  # non-feature columns
X = data.drop(data.columns[cols], axis=1)
y = data["work_mgt"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
model = Sequential([Dense(69, input_dim=69, activation='relu'),
                    Dense(1, activation="sigmoid")
                    ])
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=10, epochs=100, shuffle=True, verbose=1)
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy * 100))
