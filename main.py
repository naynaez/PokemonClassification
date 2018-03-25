import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

usedAttributes = ["type1", "abilities",'against_dark', 'against_bug', 'base_total',
                'base_egg_steps','against_dragon','against_electric','against_fairy',
                  'against_fight','against_fire','against_flying','against_ghost','against_grass',
                  'against_ground','against_ice','against_psychic','against_rock','against_steel',
                  'against_water','type2']  # Define read attributes.
pokemonData = pd.read_csv('pokemon.csv', usecols=usedAttributes)  # Read CSV file.

    # Replace NaN object with "NULL" string
for i in range(usedAttributes.__len__()):   # attribute iterator
    for j in range(pokemonData.__len__()):      # row iterator

        if pokemonData[usedAttributes[i]][j] is pd.np.nan:   # Replace Nan Object with "NULL"
            pokemonData[usedAttributes[i]][j] = "NULL"

        if i == 1 :                 # Clean "abilities" value to one first ability
            str_index = 0
            for charac in pokemonData[usedAttributes[i]][j]:
                if charac == '\'' and str_index > 2 :
                    pokemonData[usedAttributes[i]][j] = pokemonData[usedAttributes[i]][j][2:str_index]
                    break
                str_index = str_index + 1

########### Encoded catagory/string  to number

type1_value = dict(zip(pokemonData['type1'].astype('category').cat.categories.tolist(), range(18)))
pokemonData['type1_encoded'] = pokemonData['type1'].map(type1_value)

type2_value = dict(zip(pokemonData['type2'].astype('category').cat.categories.tolist(), range(19)))
pokemonData['type2_encoded'] = pokemonData['type2'].map(type2_value)

#print(pokemonData['abilities'].astype('category').values)
abilities_value = dict(zip(pokemonData['abilities'].astype('category').cat.categories.tolist(), range(165)))
pokemonData['abilities_encoded'] = pokemonData['abilities'].map(abilities_value)

########### Data visualization

# for i in range(usedAttributes.__len__()):
#     if( i != 0):
#         plt.xlabel(usedAttributes[i]) # X label for scatter plot
#         plt.ylabel(usedAttributes[0]) # Y label for scatter plot
#         plt.scatter(pokemonData[usedAttributes[i]], pokemonData[usedAttributes[0]]) # Scatter Plot
#         plt.show()

########### Logistic Regression Classification

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import train_test_split

X = pokemonData[['type2_encoded', 'abilities_encoded','against_dark', 'against_bug', 'base_total',
                 'base_egg_steps','against_electric','against_fairy','against_fight','against_fire',
                 'against_flying','against_ghost','against_grass','against_ground','against_ice',
                 'against_psychic','against_rock','against_steel','against_water']]
y = pokemonData['type1_encoded']
clf = LogisticRegression()
kfold = model_selection.KFold(n_splits=10, random_state=7)
results = model_selection.cross_val_score(clf, X, y, cv=kfold, scoring= 'accuracy')
print("10-fold accuracy: %.3f" % (results.mean()))

print("\n")
########## SVM
from sklearn import svm
from sklearn.model_selection import KFold

kf = KFold(n_splits=10)
clf = svm.SVC()
i = 1
sum_acc = 0
for train, test in kf.split(X):
    X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    acc = np.sum(predictions == y_test)/len(predictions)
    sum_acc += acc
    print("round ", i, "  acc = ", acc)
    i = i+1

print("SVM Average Accuracy = ", sum_acc/10)


print("\n")
########## Final trained Model of Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(X,y)
clf = LogisticRegression()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
acc = np.sum(predictions == y_test)/len(predictions)
print("Logistic R. Accuracy = ", acc)
#print("Coeficient:\n",clf.coef_)
print("\nIntercept:\n",clf.intercept_)


########## result from Logistic Model

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, predictions)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, predictions,target_names=list(type1_value.keys())))