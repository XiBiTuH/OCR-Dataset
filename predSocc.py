#!/usr/bin/env python
# coding: utf-8


import pandas as pd




df = pd.read_csv("P1.csv")
        



lst1 = list(df['AwayTeam']) + list(df['HomeTeam'])
res = [] 
[res.append(x) for x in lst1 if x not in res] 

dict1 = {

}

for k in range(len(res)):
    dict1[res[k]] = k
    
kst = []
for i in range(len(df.index)):
     kst.append(dict1[df['HomeTeam'][i]])
        

df.insert(2, "Home", kst, True) 

kst = []
for i in range(len(df.index)):
     kst.append(dict1[df['AwayTeam'][i]])
        
df.insert(3, "Away", kst, True) 



del df['AwayTeam']
del df['HomeTeam']
del df['Date']
del df['Div']
del df['Time']


f = pd.isnull(df).sum() > 0



for k in df.columns:
    if f[k] == True:
        df[k].fillna(df[k].mean(), inplace = True) 



#Encode columns HTR, REferee, FTR
df["FTR"] = df["FTR"].astype('category')
df["HTR"] = df["HTR"].astype('category')
#df["Referee"] = df["Referee"].astype('category')



df["FTR"] = df["FTR"].cat.codes
df["HTR"] = df["HTR"].cat.codes
#df["Referee"] = df["Referee"].cat.codes



from sklearn.model_selection import train_test_split
y = df.pop("FTR")
X = df
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)



from sklearn.decomposition import PCA
pca = PCA(0.95)
pca.fit(X_train)

X_train = pca.transform(X_train) 
X_test = pca.transform(X_test) 

X = pca.fit_transform(X)



from sklearn.model_selection import cross_val_score


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
print(cross_val_score(clf,X,y))
print(clf.fit(X,y))




#Tensorflow  Imports
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


# define NN model
def NNModel(features):

    # create model
    model = Sequential()
    model.add(Dense(16, input_dim=features, activation='relu'))

    model.add(Dense(16, activation='relu'))

    model.add(Dense(32, activation='relu'))

    model.add(Dense(64, activation='relu'))

    model.add(Dense(3, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



dummy_y = to_categorical(y)

from sklearn.metrics import accuracy_score
from sklearn import preprocessing

X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.3, stratify=y)
X_train, norm = preprocessing.normalize(X_train, norm="l1", axis=0, return_norm=True)


for k in range(len(norm)):
    X_test[:, k] = X_test[:, k] / norm[k]


model_nn = NNModel(len(X[1,]))
model_nn.fit(X_train,y_train, epochs=3000)
y_pred = model_nn.predict(X_test) > 0.5

print(accuracy_score(y_test,y_pred))



from sklearn import tree, preprocessing

tree_pred = tree.DecisionTreeClassifier()
tree_pred.fit(X_train,y_train)
print(cross_val_score(tree_pred,X,y))
print(tree_pred.fit(X,y))



# App 
import numpy as np

results = {
    '0' : "Away Wins",
    '1' : "Draw",
    '2' : "Home Wins"
}

print("Welcome to the PREDICTOR")
print("Made by Pedro Brito")

print("\n\n ------------- Teams ------------- \n\n")
for i in dict1.keys():
    print(str(i) + " : " + str(dict1[i]) + "\n")
    
    
home = input("Enter Home Team : ")

away = input("Enter Away Team : ")



to_predict = [home,away]
fd = pd.DataFrame(X_test)
for i in fd.columns[2:]:
    to_predict.append(fd[i].mean())
    
aux = np.array(to_predict)
y_pred_svm = clf.predict(aux.reshape(1,-1))
print( "SVM Prediction : " + results[str(y_pred_svm[0])])


y_pred_tree = tree_pred.predict([aux])
print( "Tree Prediction : " + results[str(y_pred_tree[0])])



y_nn_pred = model_nn.predict(aux)
print( "Tree Prediction : " + results[str(y_nn_pred[0])])




