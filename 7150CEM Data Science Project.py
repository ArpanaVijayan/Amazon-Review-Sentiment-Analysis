#!/usr/bin/env python
# coding: utf-8

# # Amazon Consumer Electronics Review Analysis: Uncovering Insights from 1,500+ Reviews

# ## Importing dataset

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[2]:


data = pd.read_csv('dataset.csv')
data.head()


# In[3]:


data.info()


# In[4]:


data.shape


# ## Data Preprocessing

# In[5]:


data['reviews.rating'] = pd.to_numeric(data['reviews.rating'])


# In[6]:


data['dateAdded'] = pd.to_datetime(data['dateAdded'])
data['dateUpdated'] = pd.to_datetime(data['dateUpdated'])


# In[7]:


data.info()


# In[8]:


data.nunique()


# In[9]:


data.isnull().sum()


# In[10]:


data.isnull()


# In[11]:


data = data.drop_duplicates()
data.shape


# In[12]:


data['reviews.rating'].unique()


# In[13]:


#Option 1. Filling null values with the mean
data = data.fillna(value = data['reviews.rating'].mean())


# In[14]:


data = data.fillna('')


# In[15]:


for x in data.index:
  if data.loc[x, "reviews.rating"] == '':
    data.drop(x, inplace=True)


# ## Descriptive Statistics

# In[16]:


data.describe()


# # Exploratory Data Analysis

# In[17]:


data['reviews.rating'].unique()


# In[18]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x=data['reviews.rating'])
plt.title('Distribution of rating scores')
plt.xlabel('Rating')
plt.ylabel('Number of Reviews')
plt.show()


# In[19]:


def score(df):
  """ This function categories the reviews into positive, negative and neutral based on the overall rating given """
  if df['reviews.rating'] == 3.0 :
    review = 'neutral'
  elif df['reviews.rating'] > 3.0 :
    review = 'positive'
  elif df['reviews.rating'] < 3.0 :
    review = 'negative'
  else :
    review = -1
  return review


# In[20]:


#Applying the function in our new column
data['sentiment'] = data.apply(score, axis=1)
data.head()


# In[21]:


data['sentiment'].value_counts()


# In[22]:


sns.countplot(x=data['sentiment'])
plt.title('Distribution of rating scores')
plt.xlabel('Rating')
plt.ylabel('Number of Reviews')
plt.show()


# # Top 10 Most reviewed products

# In[23]:


top_reviews = data['id'].value_counts().head(10).index
print(top_reviews)


# In[24]:


top_reviews = data['name'].value_counts().head(10).index
print(top_reviews)


# In[25]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(9, 5))
sns.countplot(data=data, y='name', order=top_reviews)
plt.title('Top 10 most reviewed products')
plt.xlabel('Number of Reviews')
plt.ylabel('Products')
plt.show()


# In[26]:


sns.countplot(y='name',hue='reviews.rating',data=data, order=top_reviews)
sns.set(rc={'figure.figsize':(9,5)})
plt.title("Top 10 most reviewed products")
plt.xlabel('Number of Reviews')
plt.ylabel('Products')
plt.show()


# In[27]:


sns.countplot(y='name',hue='sentiment',data=data, order=top_reviews)
sns.set(rc={'figure.figsize':(9,5)})
plt.title("Top 10 most reviewed products")
plt.xlabel('Number of Reviews')
plt.ylabel('Products')
plt.show()


# # Quantitative Analysis

# In[28]:


intial_reviews = data['dateAdded']


# In[29]:


# Let's consider first product in Top 10 Review ie. "Amazon Tap - Alexa-Enabled Portable Bluetooth Speaker "

# Assuming 'id' is the unique identifier for each product
product_id = 'AVpfpK8KLJeJML43BCuD'  # Replace 'your_product_id_here' with the actual product ID
product_name = 'Amazon Tap - Alexa-Enabled Portable Bluetooth Speaker'
specific_product_data = data[data['id'] == product_id]

print('Product ID :', product_id)
print('Product Name :', product_name)
print(specific_product_data['reviews.date'])


# In[30]:


specific_product_data['reviews.date'] = pd.to_datetime(specific_product_data['reviews.date'])
specific_product_data = specific_product_data.dropna(subset=['reviews.date'])
specific_product_data = specific_product_data.sort_values(by='reviews.date')

# Group by 'date' and count the number of reviews for each date
reviews_count_per_date = specific_product_data.groupby('reviews.date').size().reset_index(name='num_reviews')
print(reviews_count_per_date)


# In[31]:


# Plotting review counts over time for the specific product
plt.figure(figsize=(10, 6))
plt.plot(reviews_count_per_date['reviews.date'], reviews_count_per_date['num_reviews'], marker='.')
plt.xlabel('Review Date')
plt.ylabel('Number of Reviews')
plt.title(f'Review Evolution Over Time for Product : {product_name}')
plt.grid(True)
plt.show()


# # Correlation between product attributes

# In[32]:


print(data['prices'][0])


# In[33]:


import pandas as pd
import json

# Convert JSON strings to Python lists
data['prices1'] = data['prices'].apply(lambda x: json.loads(x) if pd.notna(x) else [])

# Extract the relevant information from the list
data['amountMax'] = data['prices1'].apply(lambda x: x[0]['amountMax'] if x else None)
data['amountMin'] = data['prices1'].apply(lambda x: x[0]['amountMin'] if x else None)
data['currency'] = data['prices1'].apply(lambda x: x[0]['currency'] if x else None)


# In[54]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame, replace it with your actual DataFrame
# Drop irrelevant columns or preprocess data as needed

columns_required = ['reviews.numHelpful','dateAdded','dateUpdated','reviews.doRecommend',
                    'reviews.rating', 'amountMax','amountMin']

df = data[columns_required]
# Calculate the correlation matrix
correlation_matrix = df.corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[ ]:





# # Sentiment Analysis

# In[35]:


eg = data['reviews.text'][12]
print(eg)


# In[36]:


import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

all_stopwords = stopwords.words('english')
all_stopwords.remove('not')


# In[37]:


corpus=[]

for i in range(0, 1597):
  review = re.sub('[^a-zA-Z]', ' ', data['reviews.text'][i])
  review = review.lower()
  review = review.split()
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)


# In[38]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1420)


# In[39]:


df1 = data['reviews.rating']
X = cv.fit_transform(corpus).toarray()
y = df1.values

threshold = 4.0
y_train_discrete = (y > threshold).astype(int)


# In[40]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_train_discrete, test_size = 0.20, random_state = 0)


# In[41]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# In[42]:


# Exporting NB Classifier to later use in prediction
import joblib
joblib.dump(classifier, 'c2_Classifier_Sentiment_Model') 


# In[48]:


y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("Accuracy: ", accuracy_score(y_test, y_pred))


# In[49]:


print("classification report : ",classification_report(y_test, y_pred))


# In[52]:


from sklearn import metrics

cm = confusion_matrix(y_test,y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True])
cm_display.plot()
plt.show()
    


# # Applying Machine Learning Models to develop predictive models 

# ## Naives Bayes Classifier, Logistic Regression & Random Forest Classifier

# In[51]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn import metrics

# Convert ratings to binary labels (positive/negative sentiment)
threshold = 3  # Choose an appropriate threshold
data['sentiment'] = data['reviews.rating'].apply(lambda x: 1 if x >= threshold else 0)

# Tokenize and preprocess the review text
X = data['reviews.text']
y = data['sentiment']
y=y.astype('int')


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier()
}
accuracy = {}
conf_matrix = {}
class_report = {}
# Train and evaluate each model
for model_name, model in models.items():
    #Create a pipeline with TF-IDF vectorization and the current model
    pipeline = make_pipeline(TfidfVectorizer(), model)
    
    # Train the model
    pipeline.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)
    # Evaluate the model
    accuracy[model_name] = accuracy_score(y_test, y_pred)
    conf_matrix[model_name] = confusion_matrix(y_test, y_pred)
    class_report[model_name] = classification_report(y_test, y_pred)

    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy[model_name]}")
    print(f"Confusion Matrix:\n{conf_matrix[model_name]}")
    print(f"Classification Report:\n{class_report[model_name]}")
    cm = confusion_matrix(y_test,y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True])
    cm_display.plot()
    plt.show()
    print("="*50)


# In[ ]:





# In[ ]:




