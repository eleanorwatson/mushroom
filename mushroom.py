# -*- coding: utf-8 -*-

# %% -------------------------------------------------------------------------------------------------------------------
# File I/O
# ----------------------------------------------------------------------------------------------------------------------

file_path = r"/Users/eleanor/Documents/k_exercises_revisited/ML/mushroom/mush_git/mushroom/mushroom.csv"
# %% -------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
import pandas as pd
import sklearn 

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# %% -------------------------------------------------------------------------------------------------------------------
# Data Load
# ----------------------------------------------------------------------------------------------------------------------


#--------- to try ----------
# sudo port install py35-numpy py35-scipy py35-matplotlib p35-ipython 
# pip install scipy (should work and it isn't ... maybe ask on Stackoverflow.)
#  pip install scikit-learn
df = pd.read_csv(file_path)

print(df.head(20))
print(df.describe())

# Create regression dataframe
df_reg = df[['class']].copy()
df_reg['IsPoisonous'] = df_reg['class'].map({'p':1, 'e':0})
df_reg.drop(columns=['class'], inplace=True)

print("df_ref")
print(df_reg.head(20))
print(df_reg.describe())

for col in (set(df.columns) - set(['class', 'veil-type'])):
    dummies = pd.get_dummies(df[col])
    dummy_cols = {c:col+'_'+str(c) for c in dummies.columns}
    dummies.rename(columns=dummy_cols, inplace=True)
    df_reg = pd.concat([df_reg, dummies], axis=1)



# %% -------------------------------------------------------------------------------------------------------------------
# Train test split
# ----------------------------------------------------------------------------------------------------------------------
target = 'IsPoisonous'
features = set(df_reg.columns) - set([target])

# Split training and test
df_train, df_test = train_test_split(df_reg, test_size=0.2, random_state=99)

X_train = df_train[features]
y_train = df_train[target]

X_test  = df_train[features]
y_test  = df_train[target]



# %% -------------------------------------------------------------------------------------------------------------------
# Model fitting
# ----------------------------------------------------------------------------------------------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

y_hat = model.predict(X_test)
accuracy = accuracy_score(y_test, y_hat)
print('accuracy score: ', accuracy)





