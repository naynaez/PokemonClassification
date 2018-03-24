import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

usedAttributes = ["type1", "type2", "against_bug", "abilities"]  # Define selected attributes.
pokemonData = pd.read_csv('pokemon.csv', usecols=usedAttributes)  # Read CSV file.

    # Replace NaN object with "NULL" string
for i in range(usedAttributes.__len__()):   # attribute iterator
    for j in range(pokemonData.__len__()):      # row iterator

        if pokemonData[usedAttributes[i]][j] is pd.np.nan:   # Replace Nan Object with "NULL"
            pokemonData[usedAttributes[i]][j] = "NULL"

        if i == 3 :                 # Clean "abilities" value to one first ability
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

# for i in range(4):
#     if( i != 0):
#         plt.xlabel(usedAttributes[i]) # X label for scatter plot
#         plt.ylabel(usedAttributes[0]) # Y label for scatter plot
#         plt.scatter(pokemonData[usedAttributes[i]], pokemonData[usedAttributes[0]]) # Scatter Plot
#         plt.show()

########### Logistic Regression Classification

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from sklearn import model_selection
from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix

kfold = model_selection.KFold(n_splits=10, random_state=7)
clf = LogisticRegression()
X = pokemonData[['against_bug', 'type2_encoded', 'abilities_encoded']]
y = pokemonData['type1_encoded']
#clf.fit(X, y)


# coeff_df = pd.DataFrame(clf.coef_, X.columns, columns=['Coefficient'])
# print(coeff_df)
# print("Intercept = " + str(clf.intercept_))