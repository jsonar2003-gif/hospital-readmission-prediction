#!/usr/bin/env python
# coding: utf-8

# In[63]:


import pandas as pd
import numpy as np


# In[64]:


df = pd.read_csv('TrainingWiDS2021.csv')
df.head()


# In[65]:


df.shape
df.columns
df.info()
df.describe()
df.head(20)


# In[66]:


df['readmission_status'].value_counts()


# In[67]:


df.drop(['encounter_id', 'hospital_id', 'Unnamed: 0'], axis=1, inplace=True)


# In[68]:


df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
df['height'] = pd.to_numeric(df['height'], errors='coerce')


# In[69]:


df.fillna(df.median(numeric_only=True), inplace=True)


# In[70]:


df['readmission_status'] = df['readmission_status'].astype(str).str.strip().str.upper()


# In[71]:


# Age groups
df['age_group'] = pd.cut(df['age'],
                        bins=[0, 30, 50, 70, 100],
                        labels=['Young', 'Adult', 'Senior', 'Elderly'])


# In[72]:


df['readmission_status'].value_counts()


# In[75]:


df = pd.read_csv('diabetic_data.csv')

df.shape
df.columns.tolist()
df.head(10)


# In[76]:


df['readmitted'].value_counts()


# In[77]:


df['readmitted_flag'] = df['readmitted'].apply(
    lambda x: 1 if x == '<30' else 0
)


# In[78]:


df['readmitted_flag'].value_counts()


# In[79]:


df.replace('?', np.nan, inplace=True)


# In[80]:


df.drop(['weight', 'payer_code', 'medical_specialty'], axis=1, inplace=True)


# In[81]:


df.isnull().sum().sort_values(ascending=False).head(10)


# In[82]:


df['age_group'] = df['age']


# In[83]:


df['length_of_stay'] = df['time_in_hospital']


# In[84]:


df.groupby('age_group')['readmitted_flag'].mean().sort_values(ascending=False)


# In[85]:


df.groupby('gender')['readmitted_flag'].mean()


# In[86]:


df.groupby('num_medications')['readmitted_flag'].mean()


# In[87]:


df['med_group'] = pd.cut(df['num_medications'],
                        bins=[0, 5, 10, 20, 50, 100],
                        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])


# In[88]:


df.groupby('med_group')['readmitted_flag'].mean()


# In[89]:


import matplotlib.pyplot as plt

df.groupby('med_group')['readmitted_flag'].mean().plot(kind='bar')
plt.title('Readmission Rate by Medication Level')
plt.ylabel('Readmission Rate')
plt.show()


# In[29]:


df.groupby('length_of_stay')['readmitted_flag'].mean()


# In[30]:


import matplotlib.pyplot as plt

df.groupby('length_of_stay')['readmitted_flag'].mean().plot(marker='o')
plt.title('Readmission Rate vs Length of Stay')
plt.xlabel('Length of Stay (Days)')
plt.ylabel('Readmission Rate')
plt.grid()
plt.show()


# In[31]:


df.groupby('diag_1')['readmitted_flag'].mean().sort_values(ascending=False).head(10)


# In[32]:


df['diag_1']


# In[33]:


diag_counts = df['diag_1'].value_counts()

# Keep only diagnoses with > 100 patients
valid_diags = diag_counts[diag_counts > 100].index

# Filter dataset
df_filtered = df[df['diag_1'].isin(valid_diags)]


# In[34]:


df_filtered['diag_1']


# In[35]:


df_filtered.groupby('diag_1')['readmitted_flag'].mean().sort_values(ascending=False).head(10)


# In[ ]:





# In[36]:


df['diag_1'].value_counts().head(10)


# In[37]:


df_filtered.groupby('diag_1')['readmitted_flag'].mean().sort_values(ascending=False).head(10)


# In[38]:


df_filtered.groupby('diag_1')['readmitted_flag'].mean().sort_values(ascending=False).head(10).plot(kind='bar')

plt.title('Top High-Risk Diagnoses')
plt.ylabel('Readmission Rate')
plt.show()


# In[39]:


df.groupby(['age_group', 'gender'])['readmitted_flag'].mean()


# In[40]:


import matplotlib.pyplot as plt

df.groupby(['age_group', 'gender'])['readmitted_flag'].mean().unstack().plot(kind='bar')

plt.title('Readmission Rate by Age Group and Gender')
plt.ylabel('Readmission Rate')
plt.xticks(rotation=90)
plt.show()


# In[41]:


# Clean original column
df['readmitted'] = df['readmitted'].astype(str).str.strip()

# Create target variable
df['readmitted_flag'] = df['readmitted'].apply(
    lambda x: 1 if x == '<30' else 0
)


# In[42]:


df['readmitted_flag'].value_counts()


# In[43]:


y = df['readmitted_flag']


# In[44]:


features = [
    'time_in_hospital',
    'num_lab_procedures',
    'num_medications',
    'number_outpatient',
    'number_emergency',
    'number_inpatient',
    'gender'
]

X = df[features]
y = df['readmitted_flag']


# In[45]:


X = pd.get_dummies(X, drop_first=True)


# In[90]:


y_pred_rf = rf_model.predict(X_test)


# In[50]:


pip install --user imbalanced-learn


# In[54]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# In[ ]:





# In[51]:


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)


# In[55]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_res, y_train_res)


# In[58]:


y_probs = rf.predict_proba(X_test)[:,1]

y_pred_04 = (y_probs > 0.4).astype(int)
y_pred_05 = (y_probs > 0.5).astype(int)


# In[59]:


from sklearn.metrics import classification_report, confusion_matrix

y_pred = rf.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# In[57]:





# In[ ]:




