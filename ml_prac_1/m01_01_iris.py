from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings(action='ignore')
data_list = [load_iris, load_breast_cancer, load_digits, load_wine]
model_list = [RandomForestClassifier(), LinearSVC(), DecisionTreeClassifier(), LogisticRegression()]

for i in data_list:
    x, y = i(return_X_y=True)
    for j in model_list:
        model = j
        model.fit(x, y)
        results = model.score(x, y)
        print(i.__name__, type(j).__name__, results)