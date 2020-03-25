import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

data = pd.read_csv("../tiles_data/Non_uniform_health_pos_af_mgt_feature_final.csv")
# sns.scatterplot(x="ID", y="sleep_mgt", data=data)
# sns.countplot(x = "sleep_mgt", data = data)
# plt.show()

cols = [0, 69]  # non-feature columns
X = data.drop(data.columns[cols], axis=1) #drop first column (row numbers)

num_columns = len(X.columns)
y = data[["pos_af_mgt", "ID"]]
ID = data["ID"]

unique = []

#------- unique ids -------
for i in range(0, len(ID)):
    if ID[i] not in unique:
        unique.append(ID.at[i])

#------- normalize all data-------
normalized_X = pd.DataFrame()
normalized_y = pd.DataFrame()
min_max_scaler = preprocessing.MinMaxScaler()

for i in range(0, num_columns - 1):
    col_name = X.columns[i]
    x_floats = X[[col_name]].values.astype(float)
    x_scaled = min_max_scaler.fit_transform(x_floats)  #---------------Not sure if min_max_scaler gets reset for every column
    normalized_values = pd.DataFrame(x_scaled)
    normalized_X[col_name] = normalized_values.iloc[:, 0]
normalized_X["ID"] = ID

for i in range(0, len(y.columns) - 1):
    col_name = y.columns[i]
    y_floats = y[[col_name]].values.astype(float)
    y_scaled = min_max_scaler.fit_transform(y_floats)
    normalized_values = pd.DataFrame(y_scaled)
    normalized_y[col_name] = normalized_values.iloc[:, 0]
normalized_y["ID"] = ID
#------------------------------------------

model = Sequential([Dense(69, input_dim=68, activation='relu'),
                    Dense(1, activation="sigmoid")
                    ])
model.compile(optimizer="adam", loss='mean_squared_error', metrics=['mean_squared_error'])


predictions = []
for i in range(0, len(unique)):
    X_train = normalized_X[normalized_X.ID != unique[i]]
    X_test = normalized_X[normalized_X.ID == unique[i]]
    y_train = normalized_y[normalized_y.ID != unique[i]]
    y_test = normalized_y[normalized_y.ID == unique[i]]
    X_train = X_train.drop(["ID"], axis=1)
    X_test = X_test.drop(["ID"], axis=1)
    y_train = y_train.drop(["ID"], axis=1)
    y_test = y_test.drop(["ID"], axis=1)
    model.fit(X_train, y_train, batch_size=10, epochs=15, shuffle=True, verbose=1)
    # _, accuracy = model.evaluate(X_test, y_test)
    # print('Accuracy: %.2f' % (accuracy * 100))
    prediction = model.predict(X_test)
    for x in range(0, len(prediction)):
        predictions.append(prediction[x][0])
    # _, accuracy = model.evaluate(X_test, y_test)
    # print('Accuracy: %.2f' % (accuracy * 100))

# get labels in a list
want = normalized_y.drop(["ID"], axis=1)
labels = []
for i in range(0, len(want)):
    labels.append(want["pos_af_mgt"].at[i])

r, p_value = stats.pearsonr(predictions, labels)

print(f'r = {r} p-value = {p_value}')








# ------ NOTES --------
'''
03/17/20

We are getting a p_value of 0. If this number is obtained from rounding down from a p-value of 0.00001 for example, we
can say that the data is statistically significant.

Statistical significance refers to whether any differences observed between groups being studied are "real" 
or whether they are simply due to chance. If it is unlikely enough that the difference in outcomes occurred by 
chance alone, the difference is pronounced "statistically significant."

If the results yield a p-value of .05, here is what the scientists are saying: "Assuming the two groups of people 
being compared were exactly the same from the start, there's a very good chance — 95 per cent — that the three-kg 
difference in weight loss would NOT be observed if the weight loss drug had no benefit whatsoever." From this finding, 
scientists would infer that the weight loss drug is indeed effective.

A p-value of 0.1 would mean there is an excellent chance — 99 per cent — that the difference in outcomes 
would NOT be observed if the intervention had no benefit whatsoever.

A low p-value (under 0.1) is statistically significant.

'''