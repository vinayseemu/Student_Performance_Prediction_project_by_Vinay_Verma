# Importing required libraries for data handling, visualization, machine learning, and warnings
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import time as t
import sklearn.utils as u
import sklearn.preprocessing as pp
import sklearn.tree as tr
import sklearn.ensemble as es
import sklearn.metrics as m
import sklearn.linear_model as lm
import sklearn.neural_network as nn
import numpy as np
import warnings as w

w.filterwarnings('ignore')  # To ignore warning messages

# Load the student dataset
data = pd.read_csv("Student_data_detail.csv")

# Menu-based graph plotting for data visualization
ch = 0
while(ch != 10):
    print("1.Marks Class Count Graph\t2.Marks Class Semester-wise Graph\n3.Marks Class Gender-wise Graph\t4.Marks Class Nationality-wise Graph\n5.Marks Class Grade-wise Graph\t6.Marks Class Section-wise Graph\n7.Marks Class Topic-wise Graph\t8.Marks Class Stage-wise Graph\n9.Marks Class Absent Days-wise\t10.No Graph\n")
    ch = int(input("Enter Choice: "))
    
    # Various graph options based on user input
    if (ch == 1):
        print("Loading Graph....\n")
        t.sleep(1)
        print("\tMarks Class Count Graph")
        axes = sb.countplot(x='Class', data=data, order=['L', 'M', 'H'])  # Low, Medium, High classes
        plt.show()
    
    # Each 'elif' block generates a graph for specific categories
    elif (ch == 2):
        ...
    elif (ch == 3):
        ...
    # Similar elifs for options 4 to 9
if(ch == 10):
    print("Exiting..\n")
    t.sleep(1)

# Drop columns that are not required for machine learning
columns_to_drop = ["gender", "StageID", "GradeID", "NationalITy", "PlaceofBirth",
                   "SectionID", "Topic", "Semester", "Relation", "ParentschoolSatisfaction",
                   "ParentAnsweringSurvey", "AnnouncementsView"]
data = data.drop(columns_to_drop, axis=1)

u.shuffle(data)  # Shuffle the data to ensure randomness

# Encode non-numeric data into numbers for ML models
for column in data.columns:
    if data[column].dtype == type(object):
        le = pp.LabelEncoder()
        data[column] = le.fit_transform(data[column])

# Splitting data into training (70%) and testing (30%) sets
ind = int(len(data) * 0.70)
feats = data.values[:, 0:4]  # Feature columns
lbls = data.values[:,4]      # Labels (target)

feats_Train = feats[0:ind]
feats_Test = feats[(ind+1):len(feats)]
lbls_Train = lbls[0:ind]
lbls_Test = lbls[(ind+1):len(lbls)]

# Apply Decision Tree Classifier
modelD = tr.DecisionTreeClassifier()
modelD.fit(feats_Train, lbls_Train)
lbls_predD = modelD.predict(feats_Test)
countD = sum(a==b for a,b in zip(lbls_Test, lbls_predD))
accD = countD/len(lbls_Test)
print("\nAccuracy measures using Decision Tree:")
print(m.classification_report(lbls_Test, lbls_predD))
print("Accuracy using Decision Tree: ", round(accD, 3))

# Apply Random Forest Classifier
modelR = es.RandomForestClassifier()
modelR.fit(feats_Train, lbls_Train)
lbls_predR = modelR.predict(feats_Test)
countR = sum(a==b for a,b in zip(lbls_Test, lbls_predR))
print("\nAccuracy Measures for Random Forest Classifier: \n")
print(m.classification_report(lbls_Test,lbls_predR))
accR = countR/len(lbls_Test)
print("Accuracy using Random Forest: ", round(accR, 3))

# Apply Perceptron
modelP = lm.Perceptron()
modelP.fit(feats_Train, lbls_Train)
lbls_predP = modelP.predict(feats_Test)
countP = sum(a==b for a,b in zip(lbls_Test, lbls_predP))
accP = countP/len(lbls_Test)
print("\nAccuracy measures using Linear Model Perceptron:")
print(m.classification_report(lbls_Test, lbls_predP))
print("Accuracy using Linear Model Perceptron: ", round(accP, 3))

# Apply Logistic Regression
modelL = lm.LogisticRegression()
modelL.fit(feats_Train, lbls_Train)
lbls_predL = modelL.predict(feats_Test)
countL = sum(a==b for a,b in zip(lbls_Test, lbls_predL))
accL = countL/len(lbls_Test)
print("\nAccuracy measures using Linear Model Logistic Regression:")
print(m.classification_report(lbls_Test, lbls_predL))
print("Accuracy using Linear Model Logistic Regression: ", round(accL, 3))

# Apply MLP Classifier (Neural Network)
modelN = nn.MLPClassifier(activation="logistic")
modelN.fit(feats_Train, lbls_Train)
lbls_predN = modelN.predict(feats_Test)
countN = sum(a==b for a,b in zip(lbls_Test, lbls_predN))
accN = countN/len(lbls_Test)
print("\nAccuracy measures using MLP Classifier:")
print(m.classification_report(lbls_Test, lbls_predN))
print("Accuracy using Neural Network MLP Classifier: ", round(accN, 3))

# Allow user to test the models with custom inputs
choice = input("Do you want to test specific input (y or n): ")
if(choice.lower()=="y"):
    # Taking inputs step by step from user
    gen = input("Enter Gender (M or F): ")
    nat = input("Enter Nationality: ")
    pob = input("Place of Birth: ")
    gra = input("Grade ID as (G-<grade>): ")
    ...
    absc = input("Enter No. of Abscenes(Under-7 or Above-7): ")
    ...
    
    # Prepare input for prediction using only selected 4 features
    arr = np.array([rai, res, dis, absc])

    # Predict class using all trained models
    predD = modelD.predict(arr.reshape(1, -1))
    predR = modelR.predict(arr.reshape(1, -1))
    predP = modelP.predict(arr.reshape(1, -1))
    predL = modelL.predict(arr.reshape(1, -1))
    predN = modelN.predict(arr.reshape(1, -1))

    # Convert predicted numeric values back to class labels
    predD = ['H', 'M', 'L'][int(predD)]
    predR = ['H', 'M', 'L'][int(predR)]
    predP = ['H', 'M', 'L'][int(predP)]
    predL = ['H', 'M', 'L'][int(predL)]
    predN = ['H', 'M', 'L'][int(predN)]

    # Display prediction results
    print("\nUsing Decision Tree Classifier: ", predD)
    print("Using Random Forest Classifier: ", predR)
    print("Using Linear Model Perceptron: ", predP)
    print("Using Linear Model Logisitic Regression: ", predL)
    print("Using Neural Network MLP Classifier: ", predN)
    print("\nExiting...")
    t.sleep(1)
else:
    print("Exiting..")
    t.sleep(1)
