from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np 
iris= load_iris()
print(iris.feature_names)
print(iris.target_names)

#Now to test the algorithm, I'm gonna remove some random lines from the data we have and with the remaining data we will train the algorithm, if the real values from the data and algorithm matches then we can say that our algorithm is working right.
remove = [0,25,50,100]
new_target = np.delete(iris.target, remove)
new_data= np.delete(iris.data, remove, axis=0)

#now we will train our classifier

clf= tree.DecisionTreeClassifier()
clf= clf.fit(new_data, new_target)
prediction = clf.predict(iris.data[remove])
print("original lables as per data", iris.target[remove])
print("lables predicted as per algorithm", prediction) 
