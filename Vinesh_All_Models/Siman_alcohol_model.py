import tensorflow as tf
import pandas as pd
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


'''
This program trains and reports the accuracy of the alcohol data from TILES
using a FF Neural Network.

References: 

*** https://www.youtube.com/watch?v=T91fsaG2L0s
*** https://www.heatonresearch.com/2017/06/01/hidden-layers.html
'''



def dichotomize(data_outcome):
    dich = []
    try:
        for i in range(0, len(data_outcome)):
            value = float(data_outcome.values[i][0])
            if value > 0.0:
                dich.append(1.0)
            else:
                dich.append(0)
    except Exception as e:
        print("Error in dichotomize function.")
        print(e)
    data = {'labels': dich}
    return pd.DataFrame(data = data)

# baseline model
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(68, input_dim=68, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main():
    dataset_path = "Non_uniform_health_alcohol_mgt_feature_final.csv"

    df = pd.read_csv(dataset_path, header=0)


    #Train Set
    x = df.iloc[:4000,1:69]
    y = df.iloc[:4000,69:70]
    dichotomized_y = dichotomize(y)

    #Test Set
    x_test = df.iloc[4000:,1:69]
    y_test = df.iloc[4000:,69:70]
    dich_y_test = dichotomize(y_test)


    baseline_model = create_baseline()
    baseline_model.fit(x, dichotomized_y, epochs=100, batch_size=10)

    _, accuracy = baseline_model.evaluate(x_test, dich_y_test)
    print('Accuracy: %.2f' % (accuracy * 100))


    # evaluate baseline model with standardized dataset
    # estimators = []
    # estimators.append(('standardize', StandardScaler()))
    # estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)))
    # pipeline = Pipeline(estimators)
    # kfold = StratifiedKFold(n_splits=10, shuffle=True)
    # results = cross_val_score(pipeline, x, dichotomized_y, cv=kfold)
    # print("Standardized Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))


main()