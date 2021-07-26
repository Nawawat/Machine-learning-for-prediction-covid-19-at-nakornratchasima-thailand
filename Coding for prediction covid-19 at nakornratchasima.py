#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# In[39]:


df=pd.read_excel('/Users/nawawatcharoensuk/Desktop/Book1.xlsx')


# In[78]:


df


# In[40]:


plt.subplots(figsize=(10,10))
sns.heatmap(df.corr(),annot=True)
sns.pairplot(df)
plt.show()


# In[71]:


x=df.a.values.reshape(-1,1)
y=df.F.values.reshape(-1,1)


# In[72]:


y


# In[73]:


poly_feature=PolynomialFeatures(degree=1)
x_poly=poly_feature.fit_transform(x)
model=LinearRegression()
model.fit(x_poly,y)


# In[74]:


y_poly_pred=model.predict(x_poly)
sns.set_style('darkgrid')
plt.scatter(x,y,color='b',s=20)
plt.plot(x,y_poly_pred,color='r')
plt.ylabel('Force')
plt.xlabel('accelaration')
plt.title('')
plt.show()


# In[75]:


from sklearn.metrics import r2_score


# In[76]:


print('R2={:.5f}'.format(r2_score(y,y_poly_pred)))


# In[77]:


predict=model.predict(x_poly)
predict


# In[ ]:




