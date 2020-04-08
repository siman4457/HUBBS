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

data = pd.read_csv("Non_uniform_health_anxiety_mgt_feature_final.csv", header=0)

sns.countplot(x="anxiety_mgt", data=data)
plt.show()

cols = [0, 69, 70]  # non-feature columns
X = data.drop(data.columns[cols], axis=1)
y = data["anxiety_mgt"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
