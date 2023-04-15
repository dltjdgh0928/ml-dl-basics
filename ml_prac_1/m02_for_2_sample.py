from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings(action='ignore')
data_list = [load_iris, load_breast_cancer, load_digits, load_wine]
data_name = ['load_iris', 'load_breast_cancer', 'load_digits', 'load_wine']
model_list = [RandomForestClassifier(), LinearSVC(), DecisionTreeClassifier(), LogisticRegression()]
model_name = ['RandomForestClassifier', 'LinearSVC', 'DecisionTreeClassifier', 'LogisticRegression']

for i, v1 in enumerate(data_list):
    x, y = v1(return_X_y=True)
    for j, v2 in enumerate(model_list):
        model = v2
        model.fit(x, y)
        results = model.score(x, y)
        print(data_name[i], model_name[j], results)