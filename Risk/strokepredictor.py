# %%
import pandas as pd 
import numpy as np 
import altair as alt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


# %%
data = pd.read_csv('healthcare-dataset-stroke-data.csv')



# %%
strokData = pd.get_dummies(data = data)
# %%
stroke_subset = strokData.drop(columns= ['gender_Other', 'id', 'stroke', 'smoking_status_Unknown']).head(500).dropna()
stroke_subset_2 = strokData.filter(['age', 'hypertension','stroke', 'heart_disease', 'bmi', 'gender_Female', 'gender_Male', 'ever_married_No', 'ever_married_Yes']).dropna()
# my_subset = 
# %%
#sns.pairplot(stroke_subset_2, hue = 'heart_disease')

# %%
X_pred = stroke_subset_2.drop(columns = ['heart_disease','stroke'])
#y_pred = 
y_pred = stroke_subset_2.filter(['heart_disease', 'stroke'])
#X_train = 
#y_train = 
X_train, X_test, y_train, y_test = train_test_split(
    X_pred, 
    y_pred, 
    test_size = .3, 
    random_state = 70)

model = RandomForestClassifier()
model.fit(np.array(X_train), np.array(y_train))

# %%

prediction = model.predict(np.array(X_test))


# %%
metrics.accuracy_score(y_test, prediction)