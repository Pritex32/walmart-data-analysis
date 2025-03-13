import streamlit as st

PORTFOLIO_URL='https://pritex32.github.io/prisca.github.io/'


 # Replace with your actual portfolio link



 

# Inject CSS and HTML for the navbar

st.markdown("""
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
""", unsafe_allow_html=True)  # make sure to add this to connect the nav bar. copy it exactly



st.markdown(f"""
    <!-- Just an image -->
    <nav class="navbar navbar-dark bg-dark">
      <a class="nav-link" href="#">
        🏠 Home
        <img src="https://getbootstrap.com/docs/4.4/assets/brand/bootstrap-solid.svg" width="30" height="30" alt="Home">
      </a>
      <a class="nav-link" href="{PORTFOLIO_URL}" target="_blank">
        📞 Contact
        <img src="https://getbootstrap.com/docs/4.4/assets/brand/bootstrap-solid.svg" width="30" height="30" alt="Contact">
      </a>
    </nav>
    """, unsafe_allow_html=True)

    
# Add space to push content below the navbar
st.write("")  



st. markdown("""# Author : Prisca Ukanwa""")


# In[ ]:





st.markdown("""# PROJECT AIM:
#### - Summary statistics of customer purchases
#### - Distribution of Age, Gender, City_Category, Occupation
#### - Average purchase amount by customer segment
#### -Customer Segmentation:

#### - Cluster customers based on purchasing behavior
#### - Identify high-value customers using K-Means clustering
#### - Feature Engineering & Modeling:

#### - Build a predictive model to estimate Purchase based on customer attributes
#### - Train a Regression Model (Linear Regression, Decision Tree, or Random Forest)

# Application of Customer Behavior Analysis
### ✅ Personalized Marketing: Tailoring ads based on demographics and past purchases.
### ✅ Demand Forecasting: Predicting product demand to manage inventory better.
### ✅ Recommendation Systems: Suggesting products customers are likely to buy (like Amazon).
### ✅ Customer Segmentation: Grouping similar customers for targeted strategies.""")




st.markdown("""
# Exploratory Aspects of Customer Behavior Analysis
## 1️⃣ Demographic Insights
### 🔹 Gender: Do men and women have different purchasing patterns?
### 🔹 Age: Are younger customers spending more on specific products?
### 🔹 Marital Status: Do married individuals buy different products compared to singles?

## 2️⃣ Shopping Preferences
### 🛍️ Product Choices: What categories do customers prefer?
### 🔄 Purchase Frequency: How often do customers buy?
### 💰 Spending Patterns: How much do customers spend on average?

## 3️⃣ Influencing Factors
### 👔 Occupation: Does job type affect spending habits?
### 🏙️ City Category: Does a customer’s location (urban, semi-urban, rural) impact their purchase decisions?
### 🏡 Stay in Current City: Are long-term residents more loyal customers?

""", unsafe_allow_html=True)


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


import streamlit as st


# In[2]:




# In[3]:

df=pd.read_csv('walmart.csv')


# In[4]:




# In[5]:





# In[6]:


st.subheader(' 🔹 DATASET')
wal=df[['Gender', 'Age', 'Occupation', 'City_Category',
       'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category',
       'Purchase']]


st.write(wal)





# In[8]:


st.write ("Dataset size" , df.shape)


# In[9]:

# no null value

# In[10]:


# reconfirm null values
st.write('missing values',wal.isnull().sum())


# In[11]:


# duplicates
st.write('we have ',{wal.duplicated().sum()},'duplicates')
# we have 20,000 plus duplicates


# In[12]:


# remove the dupliactes
dup=wal.drop_duplicates(inplace=True)


# # Data cleaning

# In[13]:






# In[15]:


def convert_age(age):
    if pd.isna(age):  # Handle NaN values
        return None
    age = str(age)  # Convert to string to avoid errors
    
    if '+' in age:
        return int(age[:-1])  # Convert '55+' to 55
    elif '-' in age:
        start, end = map(int, age.split('-'))  # Convert '0-17' to (0+17)/2 = 8.5
        return (start + end) / 2
    else:
        return float(age)  # If it's a single numeric value, return it

# Apply the function to the DataFrame
wal['Age_Numeric'] = wal['Age'].apply(convert_age)


# In[16]:


wal['Stay_In_Current_City_Years']=wal['Stay_In_Current_City_Years'].str.replace('+','')



# In[17]:




st.subheader(' 🚀 CUSTOMER BEHAVIOURAL INSIGHT')

# In[18]:




# ## 1.Do men and women have different purchasing patterns?

# In[19]:

st.subheader('VISUALIZATIONS 👇')

st.markdown("## 1. Do men and women have different purchasing patterns?")

purchase=wal.groupby('Gender')['Purchase'].mean()
purchase.sort_values(ascending=False)


# In[20]:

fg=plt.figure(figsize=(5,5))
purchase.plot(kind='pie',autopct='%1.1f%%',labels=['female','male'],shadow=True,colors=['red','green'])
plt.title('male and women purchasing pattern')

st.pyplot(fg)
st.subheader("finding: Yes, From the chart above,both men and women have different purchasing pattern, Men purchase more by value of 9472 than women ")
st.markdown("#### 2.Are younger customers spending more on specific products?")

# In[21]:


fig4 = plt.figure(figsize=(8, 5))  # Set proper figure size
sns.barplot(x=wal['Age_Numeric'], y=wal['Product_Category'], color='brown', ci=None)
plt.title('Spending Pattern of Young Customers')

st.pyplot(fig4)  

# In[ ]:





# In[ ]:





st.subheader('findings: from the chart above, young people from the age of 8 to 30 years purcahase prouduct in the category of 5, product category of (5) is leading in young people, the higher the customer age the more product preferences changes')

# In[ ]:





st.markdown("#### 3.Do married individuals buy different products compared to singles?")

# In[22]:


from sklearn.preprocessing import LabelEncoder


# In[23]:


le=LabelEncoder()


# In[24]:


married_pple=wal.groupby('Marital_Status')['Product_Category'].mean()
st.write('Date:',married_pple)


# In[25]:

fig6=plt.figure(figsize=(10,10))
sns.barplot(married_pple,palette='magma')
plt.title('purchasing behaviour of married individuals')
plt.xlabel('product_Category')
plt.ylabel('marrital status')
st.pyplot(fig6)

st.subheader('Findings : there is no much difference on the purchasing pattern of married and single people, the purchases are same ')

# In[ ]:





st.subheader('4. How much do customers spend on average?')

# In[26]:


spending=wal.groupby('Age_Numeric')['Purchase'].mean()
spending.sort_values(ascending=False)
st.write('Data:',spending)

# In[27]:


st.subheader('Average spending habit of customer')
fig7=plt.figure(figsize=(10,10))
plt.title('Average spending ')
sns.barplot(spending,palette='viridis')
st.pyplot(fig7)
st.subheader('Findings: the spending among customers ranges from 8000 to 9500 dollars on average, people at the age of 53 spends the most from the chart above, young people spends less, no customer is spending up to $10,000 on average')

# In[ ]:





st.subheader('5.Does job type affect spending habits?')

# In[28]:


job=wal.groupby('Occupation')['Purchase'].agg(['sum','mean'])
df1=pd.DataFrame(job)

st.write('purchasing behaviour base on occupation', df1)
# In[29]:


df1.sort_values(by='sum',ascending=False)
## the following are the occupation and their total spending


# In[30]:


job=wal.groupby('Occupation')['Purchase'].sum()


# In[31]:

fig8=plt.figure(figsize=(10,10))
sns.barplot(job,palette='coolwarm')
plt.title('purchases base on occupations')
st.pyplot(fig8)

st.subheader('finding:from the chart above, customers that belong to  occupation 0 and 4 purchases more than other occupations,therefore, job type affect the customer spending')

# In[ ]:





st.markdown('### 6. Does a customer’s location (urban, semi-urban, rural) impact their purchase decisions?')

# In[32]:


location=wal.groupby('City_Category')['Purchase'].sum()
location.sort_values(ascending=False)
st.write('Data:',location)

# In[33]:

fig9=plt.figure(figsize=(10,10))
sns.lineplot(location, marker='o',markerfacecolor='red')
plt.title('Purchases by location')
st.pyplot(fig9)
# In[34]:

fig10=plt.figure(figsize=(10,10))
sns.barplot(location, label='location vs purchase',color='yellow')
plt.legend()
st.pyplot(fig10)


# In[ ]:





# In[35]:

fig12=plt.figure(figsize=(10,8))
plt.pie(location,labels=['A','B','C'],autopct='%1.1f%%',colors=['red','green','yellow'],shadow=True)
plt.title('Percentage of Purchases by Location')
st.pyplot(fig12)

st.subheader('findings: from the chart above, customers in city B , spends more by 41% than other cities, therefore cities affect buying power of customers')

# In[ ]:





st.markdown('### 7.Are long-term residents more loyal customers?')

# In[36]:


residents=wal.groupby('Stay_In_Current_City_Years')['Purchase'].mean()
st.write('Data:',residents)


# In[37]:

fig13=plt.figure(figsize=(10,10))
sns.barplot(residents,color='skyblue')
plt.title('long term resident purchasing habit')
st.pyplot(fig13)
st.subheader('finding: All resisdents are loyal and how long a residents stays in a location doesnt affect there spending pattern from the above charts')

# In[ ]:





st.markdown('# CLUSTERING (customer segmentation)')

# In[38]:





# In[39]:

st.subheader('segementation of customer base on the purchase behaviour')
x=wal[['Product_Category','Purchase','Age_Numeric']]
st.write('Data:',x)

# In[40]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# In[41]:


sc=StandardScaler()


# In[42]:


scaled_data=sc.fit_transform(x)


# In[43]:


# getting value of k


# In[44]:


error=[]
for i in range(1,15):
    kmeans=KMeans(n_clusters=i,random_state=0).fit(scaled_data)
    error.append(kmeans.inertia_)
    


# In[45]:

st.subheader('Determining the optimum number of k using the elbow hand method')
fig14=plt.figure(figsize=(8,8))
plt.plot(range(1,15),error,marker='d')
plt.grid(True)
st.pyplot(fig14)
st.subheader('k is chosen to be 5')


# In[46]:




# In[47]:


## kmeams is chosen to be clustered into 5


# In[48]:


km_model=KMeans(n_clusters=5)
km_model.fit(scaled_data)


# In[49]:


clusters=km_model.labels_



# In[ ]:





# In[50]:


## add clusters to dataframe


# In[51]:


wal['clusters']=clusters


# In[52]:


df_drop=wal.drop(['Age'],axis=1,inplace=True)


# In[53]:


wal['City_Category']=le.fit_transform(wal['City_Category'])
wal['Gender']=le.fit_transform(wal['Gender'])


# In[54]:


wal.head()


# In[55]:


wal['Stay_In_Current_City_Years']=wal['Stay_In_Current_City_Years'].astype(int)


# In[56]:


st.markdown("## lets get the summary of the clustering")
summary=wal.groupby(['clusters']).mean()
st.write(summary)

# # HERE THE THE CUSTOMER SEGMENTATION BASED ON THE PURCHASING PATTERN

# In[57]:




st.markdown("""
## Cluster Analysis Summary:

### Cluster 0 (Older, Urban, High-Spending Group)
- Mostly males with a high occupation index.
- Older customers (~45 years) with moderate spending (14,528).
- More urban-based shoppers.

### Cluster 1 (Younger, High-Spending Group)
- Mostly males, younger (~26 years), and high-spending (15,002).
- Likely urban customers but with lower marital rates.
- Buys from lower-numbered product categories.

### Cluster 2 (Middle-Aged, Niche Product Buyers)
- Age - 35 years, mixed marital status.
- Buys from higher-numbered product categories (16.36).
- Lower spending compared to Cluster 1 (10,858).

### Cluster 3 (Young, Low-Spending Group)
- Young (~26 years), low-spending (6,238).
- Mostly single individuals in smaller cities.
- Lower occupational index, possibly students or entry-level workers.

### Cluster 4 (Older, Low-Spending Group)
- Oldest group (~45 years), more married individuals.
- Low spending (6,379), likely budget-conscious shoppers.
""")
# In[ ]:





st.subheader(""" 
Business Implications:
#### Cluster 1 (Young High-Spending) and Cluster 0 (Older High-Spending) are most valuable and should be targeted for high-end product promotions.
#### Cluster 3 and 4 have the lowest spending and may respond better to discounts or budget-friendly products.
#### Cluster 2 buys from unique categories, so special category-specific marketing could be effective.""")

# In[ ]:







# In[58]:





# In[59]:





# In[60]:


x_df=wal[['Gender', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years',
       'Marital_Status', 'Product_Category', 'Purchase','Age_Numeric']]


# In[61]:
st.subheader('CORRELATIONS')

x_corr=x_df.corr()
st.write('Date:', x_df)

# In[62]:

fig16=plt.figure(figsize=(10,10))
st.subheader('Correlation map')
sns.heatmap(x_corr,annot=True,cmap='magma')
plt.title('Correlations among features')
st.pyplot(fig16)
