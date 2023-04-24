from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import numpy as np
import matplotlib.pyplot as plt
scaler = MinMaxScaler()

def plot_feature_importances(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)
    plt.title(model)
    
model_list = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]

datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=337)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

for i in range(4):
    globals()['model'+str(i)] = model_list[i]
    globals()['model'+str(i)].fit(x_train, y_train)
    plt.subplot(2, 2, i+1)
    print(globals()['model'+str(i)].feature_importances_)
    plot_feature_importances(globals()['model'+str(i)])
    if i == 3:
        plt.title('XGBClassifier()')
plt.show()