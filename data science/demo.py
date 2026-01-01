from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

clf_tree = tree.DecisionTreeClassifier()
clf_svc = SVC()
clf_knn = KNeighborsClassifier()
clf_gnb = GaussianNB()
# CHALLENGE - create 3 more classifiers...
# 1 Support Vector Classifier
# 2 K-nearest Neighbour(KNN)
# 3 Gaussian Naive Bayes

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
clf_tree = clf_tree.fit(X, Y)
clf_svc=clf_svc.fit(X,Y)
clf_knn=clf_knn.fit(X,Y)
clf_gnb = clf_gnb.fit(X,Y)              
#Test
test_input=[190,70,43]

#predict
prediction_tree = clf_tree.predict([test_input])
prediction_svc = clf_svc.predict([test_input])
prediction_knn = clf_knn.predict([test_input])
prediction_gnb = clf_gnb.predict([test_input])

y_true=['male']

#calculate accuracy
results = {
    "Decision Tree": accuracy_score(y_true, prediction_tree),
    "SVC":           accuracy_score(y_true, prediction_svc),
    "KNN":           accuracy_score(y_true, prediction_knn),
    "Naive Bayes":   accuracy_score(y_true, prediction_gnb)
}
for i in results:
    print(i,results[i])
# CHALLENGE compare their reusults and print the best one!

