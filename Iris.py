import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import model_selection

pd.options.display.max_columns = 5
pd.options.display.expand_frame_repr = False

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(r"C:\Users\User\Desktop\iris.data", names=names)
X = dataset.drop(['class'], axis=1)
Y = dataset['class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=13)
models = []
models.append(['KNN', KNeighborsClassifier()])
models.append(['LogReg', LogisticRegression(max_iter=5000)])
models.append(['SVM', LinearSVC(max_iter=5000)])
models.append(['Tree', DecisionTreeClassifier(max_depth=5000)])
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=43, shuffle=True)
    result = cross_val_score(model, X, Y, cv=kfold)
    print(f'{name}:{result.mean()}')

Log = LogisticRegression()
Log.fit(X_train, Y_train)
pred = Log.predict(X_test)
print(accuracy_score(Y_test, pred))