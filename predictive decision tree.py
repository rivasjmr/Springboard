# predictive model using a decision tree

import pandas as pd
#
# Training data
df_training = pd.read_csv('C:\\Users\\rivas\\OneDrive\\Documents\\JMR\\Education\\Springboard\\Projects\\Capstone1\\fashionmnisttrain.csv')
feature1 = df_training.iloc[:, 1:]
label1 = df_training.iloc[:, :1]

#
import sklearn
from sklearn import tree
#
# Process Decision Tree with training data
clf = tree.DecisionTreeClassifier()
clf.fit(feature1,label1)

#
# Test data
df_test = pd.read_csv('C:\\Users\\rivas\\OneDrive\\Documents\\JMR\\Education\\Springboard\\Projects\\Capstone1\\fashionmnisttest.csv')
feature2 = df_test.iloc[:, 1:]
label2 = df_test.iloc[:, :1]

#
# Predict using Test Data
clf_results = clf.predict(feature2)

#
# Find results against Test data Labels
compare_results = {}

for i in range(len(clf_results)):
    compare_results[i] = clf_results[i] == label2.iloc[i]
    
results_pct = 1* sum(compare_results.values())/len(compare_results)
print("Accurracy predicting from decsion tree model: "+"{:.2%}".format(results_pct[0]))

#
#







