#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data_app=pd.read_csv(r'E:\diploma ML\New folder (2)\application_record.csv')


# In[3]:


data_app.head()


# In[4]:


data_record=pd.read_csv(r'E:\diploma ML\New folder (2)\credit_record.csv')


# In[5]:


data_record.head()


# In[6]:


f1_data=data_app.merge(data_record,on=['ID'],how='inner')
f1_data.head()


# In[7]:


f1_data.isna().sum()


# In[8]:


f1_data.shape


# In[9]:


f1_data.dropna(inplace=True)
f1_data.isna().sum()


# In[10]:


f1_data['STATUS'].value_counts()


# # preprocessing

# In[11]:


f1_data.status=f1_data.STATUS.map({'X':1,'C':1,'0':1,'1':0,'2':0,'3':0,'4':0,'5':0})


# In[12]:


f1_data.sample(4)


# In[13]:


data_app['ID'].nunique() 


# In[14]:


data_record['ID'].nunique() 


# In[15]:


f1_data = f1_data.drop_duplicates('ID', keep='last') 


# In[16]:


f1_data['CODE_GENDER'].value_counts()


# In[17]:


f1_data['FLAG_OWN_CAR'].value_counts()


# In[18]:


f1_data['FLAG_OWN_REALTY'].value_counts()


# In[19]:


f1_data['NAME_INCOME_TYPE'].value_counts()


# In[20]:


f1_data['NAME_EDUCATION_TYPE'].value_counts()


# In[21]:


f1_data['NAME_FAMILY_STATUS'].value_counts()


# In[22]:


f1_data['NAME_HOUSING_TYPE'].value_counts()


# In[23]:


f1_data['STATUS'].value_counts(normalize=True)


# In[24]:


f1_data.drop('OCCUPATION_TYPE', axis=1, inplace=True) 


# In[25]:


f1_data['CODE_GENDER'].replace('M',0,inplace=True) #male -> 0
f1_data['CODE_GENDER'].replace('F',1,inplace=True)#female -> 1
f1_data['FLAG_OWN_CAR'].replace('Y',0,inplace=True)
f1_data['FLAG_OWN_CAR'].replace('N',1,inplace=True)
f1_data['FLAG_OWN_REALTY'].replace('Y',0,inplace=True)
f1_data['FLAG_OWN_REALTY'].replace('N',1,inplace=True)
f1_data.head(4)


# In[26]:


quantile_hi = f1_data['CNT_CHILDREN'].quantile(0.999)
quantile_low = f1_data['CNT_CHILDREN'].quantile(0.001)
f1_data = f1_data[(f1_data['CNT_CHILDREN']>quantile_low) & (f1_data['CNT_CHILDREN']<quantile_hi)]


# In[27]:


quantile_hi = f1_data['AMT_INCOME_TOTAL'].quantile(0.999)
quantile_low = f1_data['AMT_INCOME_TOTAL'].quantile(0.001)
f1_data = f1_data[(f1_data['AMT_INCOME_TOTAL']>quantile_low) & (f1_data['AMT_INCOME_TOTAL']<quantile_hi)]


# In[28]:


quantile_hi = f1_data['CNT_FAM_MEMBERS'].quantile(0.999)
quantile_low = f1_data['CNT_FAM_MEMBERS'].quantile(0.001)
f1_data = f1_data[(f1_data['CNT_FAM_MEMBERS']>quantile_low) & (f1_data['CNT_FAM_MEMBERS']<quantile_hi)]


# In[29]:


data_app.head()


# In[30]:


f1_data.info()


# In[31]:


f1_data.NAME_INCOME_TYPE.value_counts().plot(kind='bar')


# In[32]:


def income_tans(df,col):
    result_col = []
    for i in df[col]:
        if i == 'Working':
            result_col.append('Working')
        elif i == 'Commercial associate':
            result_col.append('Commercial associate')
        elif i == 'State servant':
            result_col.append('State servant')
        else:
            result_col.append('others')
    df[col] = result_col
    return df


income_tans(f1_data,'NAME_INCOME_TYPE')
f1_data.NAME_INCOME_TYPE.value_counts()


# In[33]:


f1_data.AMT_INCOME_TOTAL = f1_data.AMT_INCOME_TOTAL/1000
f1_data.AMT_INCOME_TOTAL = f1_data.AMT_INCOME_TOTAL.astype('int')
f1_data.head(3)


# In[34]:


f1_data['Age']=-(f1_data['DAYS_BIRTH'])//365

f1_data['employee_from_years']=-(f1_data['DAYS_EMPLOYED'])//365

f1_data.drop(columns=('DAYS_BIRTH'), inplace=True)

f1_data.drop(columns=('DAYS_EMPLOYED'), inplace=True)
f1_data.head(4)


# In[35]:


f1_data.drop(columns=['ID','CNT_CHILDREN','FLAG_MOBIL','FLAG_WORK_PHONE','FLAG_EMAIL'], inplace=True)
f1_data.sample(10)


# In[36]:


plt.figure(figsize = (15,10))
sns.heatmap(f1_data.corr(), annot = True )
plt.title('Correlation ', size = 16)
plt.show()


# In[37]:


# one hot encoding
def one_hot_encode(df, columns):
    for col in columns:
        one_hot = pd.get_dummies(df[col], drop_first=True)
        df = pd.concat([df, one_hot], axis=1)
        df = df.drop([col], axis=1)
    return df

categories=['NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE']
f1_data = one_hot_encode(f1_data,categories)
f1_data.head(5)


# In[38]:


x = f1_data.drop(columns=('STATUS'),axis=1)
y = f1_data.STATUS


# In[39]:


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.2,random_state=2)
print('X_train :', train_x.shape)
print('X_test :', test_x.shape)
print('y_train :', train_y.shape)
print('y_test :', test_y.shape)


# # oversampling

# In[40]:


from imblearn.over_sampling import SMOTE
from collections import Counter
print("data before oversampling ",Counter(train_y))


# In[41]:


sm=SMOTE()
x_res,y_res=sm.fit_resample(train_x,train_y)
print("data before oversampling ",Counter(train_y))


# In[42]:


from imblearn.under_sampling import TomekLinks
print("data after oversampling ",Counter(y_res))
t1=TomekLinks(sampling_strategy='all')
x_res,y_res=sm.fit_resample(x_res,y_res)
print("data after undersampling ",Counter(y_res))


# # creation models

# In[43]:


from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
model.fit(x_res,y_res)


# In[44]:


y_pred=model.predict(test_x)


# In[45]:


from sklearn.metrics import accuracy_score
accuracy_score(test_y,y_pred)


# In[46]:


y_pred=model.predict(train_x)
accuracy_score(train_y,y_pred)


# In[47]:


from sklearn.impute import SimpleImputer

for col in f1_data.columns:
    if f1_data[col].dtype=='object':
        
        s=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
        f1_data[col]=s.fit_transform(f1_data.loc[:,col].values.reshape(-1,1))
        
    else:
        s=SimpleImputer(missing_values=np.nan,strategy='mean')
        f1_data[col]=s.fit_transform(f1_data.loc[:,col].values.reshape(-1,1))
        


# In[48]:


from sklearn.tree import DecisionTreeClassifier
model1=DecisionTreeClassifier()
model1.fit(train_x,train_y)


# In[49]:


from sklearn.metrics import accuracy_score

y_pred=model1.predict(test_x)
accuracy_score(y_pred,test_y)


# In[50]:


from sklearn.ensemble import RandomForestClassifier

model2 = RandomForestClassifier(max_features = 'auto',
                                   min_samples_split = 2,
                                   min_samples_leaf = 1,
                                   n_jobs=-1)
    
model2.fit(train_x, train_y)
model2.score(test_x, test_y)
y_pred = model2.predict(test_x)
print("accuracy_score : ",accuracy_score(test_y,y_pred))


# In[51]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
model3=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1))
model3.fit(train_x,train_y)


# In[52]:


AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1))


# In[53]:


y_pred=model3.predict(test_x)
print("accuracy_score : ",accuracy_score(test_y,y_pred))


# In[54]:


from sklearn.metrics import classification_report,confusion_matrix,f1_score,accuracy_score
print("classification_report\n : ",classification_report(test_y,y_pred))
print("confusion_matrix : ",confusion_matrix(y_pred,test_y))
print("f1_score : ",f1_score(test_y,y_pred))
print("accuracy_score : ",accuracy_score(test_y,y_pred))

