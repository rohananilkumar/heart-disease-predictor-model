import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

df = pd.read_csv('heart_2020_cleaned.csv')


# sns.pairplot(df)
print(df.info())
df = df[(df['AgeCategory']=='25-29')]

df['HeartDisease'] = df['HeartDisease'].apply(lambda x: 1 if x=='Yes' else 0)
df['Smoking'] = df['Smoking'].apply(lambda x: 1 if x=='Yes' else 0)
df['AlcoholDrinking'] = df['AlcoholDrinking'].apply(lambda x: 1 if x=='Yes' else 0)
df['Stroke'] = df['Stroke'].apply(lambda x: 1 if x=='Yes' else 0)
df['Diabetic'] = df['Diabetic'].apply(lambda x: 1 if x=='Yes' else 0)
df['DiffWalking'] = df['DiffWalking'].apply(lambda x: 1 if x=='Yes' else 0)
df['Asthma'] = df['Asthma'].apply(lambda x: 1 if x=='Yes' else 0)
df['KidneyDisease'] = df['KidneyDisease'].apply(lambda x: 1 if x=='Yes' else 0)
df['SkinCancer'] = df['SkinCancer'].apply(lambda x: 1 if x=='Yes' else 0)
df['PhysicalActivity'] = df['PhysicalActivity'].apply(lambda x: 1 if x=='Yes' else 0)
df = df.drop(['GenHealth','Race', 'AgeCategory'], axis=1)


dfHeartDisease =  df[df.HeartDisease==1]
dfNotHeartDisease =  df[df.HeartDisease==0]
dfNotHeartDisease = dfNotHeartDisease.sample(300)
df = pd.concat([dfHeartDisease, dfNotHeartDisease])




# heartDisease = df[df.HeartDisease==1].groupby('SleepTime').count().reset_index()
# heartNotDisease = df[df.HeartDisease==0].groupby('SleepTime').count().reset_index()
# plt.plot(heartDisease.SleepTime, heartDisease.HeartDisease, label='Heart Diseased')
# plt.plot(heartNotDisease.SleepTime, heartNotDisease.HeartDisease, label='Heart not diseased')
# plt.legend()
# plt.show()

sex = pd.get_dummies(df['Sex'], drop_first=True)
# ageCategory = pd.get_dummies(df['AgeCategory'], drop_first=True)
df.drop('Sex', axis=1, inplace=True)
# df.drop('AgeCategory', axis=1, inplace=True)

df = pd.concat([df, sex], axis=1)

print(df.info())

X= df.drop('HeartDisease', axis=1)
y= df['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

logModel = LogisticRegression()
logModel.fit(X_train, y_train)

predictions = logModel.predict(X_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

sns.countplot(x='HeartDisease', data=df)

plt.show()
# plt.show()