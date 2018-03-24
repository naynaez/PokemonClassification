import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

usedAttributes = ["type1", "type2", "capture_rate", "sp_defense"]  # Define selected attributes.
pokemonData = pd.read_csv('pokemon.csv', usecols=usedAttributes)  # Read CSV file.

    # Replace NaN object with "NULL" string
for i in range(usedAttributes.__len__()):
    for j in range(pokemonData.__len__()):
        if pokemonData[usedAttributes[i]][j] is pd.np.nan:
            pokemonData[usedAttributes[i]][j] = "NULL"

# for i in range(4):      # Data Visualization
#     if( i != 0):
#         plt.xlabel(usedAttributes[i]) # X label for scatter plot
#         plt.ylabel(usedAttributes[0]) # Y label for scatter plot
#         plt.scatter(pokemonData[usedAttributes[i]], pokemonData[usedAttributes[0]]) # Scatter Plot
#         plt.show()

### Convert type string to type number
#type1Value = dict(zip(pokemonData['type1'].astype('category').values, range(18)))
type1_value = dict(zip(pokemonData['type1'].astype('category').cat.categories.tolist(), range(18)))
type2_value = dict(zip(pokemonData['type2'].astype('category').cat.categories.tolist(), range(19)))

pokemonData['type1_encoded'] = pokemonData['type1'].map(type1_value)
pokemonData['type2_encoded'] = pokemonData['type2'].map(type2_value)
print(pokemonData['type2_encoded'])

### Logistic Regression Classification
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

clf = LogisticRegression()
X = pokemonData[[usedAttributes[1], usedAttributes[2], usedAttributes[3]]]
y = pokemonData[usedAttributes[0]]
clf.fit(X, y)
#
# coeff_df = pd.DataFrame(clf.coef_, X.columns, columns=['Coefficient'])
# print(coeff_df)
# print("Intercept = " + str(clf.intercept_))