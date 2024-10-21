# Twitter and Reddit Sentiment Analysis Project

## Introduction

This project demonstrates the steps to perform sentiment analysis on a dataset from Kaggle that contains Twitter and Reddit sentiment data. The project covers downloading the dataset, preprocessing, exploratory data analysis (EDA), and visualizing the results.

## Step 1: Install and Set Up Kaggle API

To begin, you need to set up the Kaggle API for accessing datasets. Follow these commands to install the Kaggle library and set up the API key (`kaggle.json`):

```bash
!pip install kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/  # Ensure your Kaggle API token is saved as 'kaggle.json'
!chmod 600 ~/.kaggle/kaggle.json
```

## Step 2: Download the Dataset from Kaggle

Download the Twitter and Reddit Sentiment Analysis dataset using the Kaggle API:

```bash
!kaggle datasets download -d cosmos98/twitter-and-reddit-sentimental-analysis-dataset
```

## Step 3: Extract the Dataset

Unzip the downloaded dataset into a specified directory:

```python
import zipfile

with zipfile.ZipFile('twitter-and-reddit-sentimental-analysis-dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('sentiment_data')
```

## Step 4: Load the Dataset

Load the extracted dataset into a Pandas DataFrame:

```python
import pandas as pd

df = pd.read_csv('sentiment_data/Twitter_Data.csv')  # Adjust the path if necessary
print(df.head())
```

## Step 5: Data Exploration

### 1. Check for Missing Values

Check if there are any missing values in the dataset:

```python
print(df.isnull().sum())
```

### 2. Handle Missing Values

Drop rows with missing values in critical columns:

```python
df = df.dropna(subset=['clean_text', 'category'])
```

## Step 6: Perform Sentiment Labeling and Analysis

### 1. Create Sentiment Labels

Create a new column to categorize sentiment labels as `positive`, `neutral`, or `negative` based on the values in the 'category' column:

```python
df['sentiment_label'] = df['category'].apply(lambda x: 'positive' if x == 1.0 else ('neutral' if x == 0.0 else 'negative'))
```

### 2. Check Distribution of Sentiment Labels

Print the distribution of sentiment labels to understand the dataset's composition:

```python
print(df['sentiment_label'].value_counts())
```

## Step 7: Data Visualization

### 1. Sentiment Distribution Pie Chart

Visualize the distribution of sentiment labels using a pie chart:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
df['sentiment_label'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#66c2a5','#fc8d62','#8da0cb'])
plt.title('Sentiment Distribution')
plt.ylabel('')  # Remove the default ylabel
plt.show()
```

### 2. Word Cloud for Positive and Negative Sentiments

Generate word clouds for positive and negative sentiments to visualize the most frequent words:

```python
from wordcloud import WordCloud

positive_words = ' '.join(df[df['sentiment_label'] == 'positive']['clean_text'])
negative_words = ' '.join(df[df['sentiment_label'] == 'negative']['clean_text'])

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(WordCloud(width=800, height=400).generate(positive_words), interpolation='bilinear')
plt.axis('off')
plt.title('Positive Sentiment Word Cloud')

plt.subplot(1, 2, 2)
plt.imshow(WordCloud(width=800, height=400).generate(negative_words), interpolation='bilinear')
plt.axis('off')
plt.title('Negative Sentiment Word Cloud')

plt.show()
```

### 3. Bar Plot of Sentiment Counts by Category

Create a bar plot to visualize the count of each sentiment label:

```python
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.countplot(x='sentiment_label', data=df, palette='viridis')
plt.title('Count of Sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
```

### 4. Box Plot of Text Length by Sentiment

Analyze the text length distribution for each sentiment category using a box plot:

```python
df['text_length'] = df['clean_text'].apply(len)

plt.figure(figsize=(8, 6))
sns.boxplot(x='sentiment_label', y='text_length', data=df, palette='Set3')
plt.title('Text Length by Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Text Length')
plt.show()
```

## Conclusion

This project outlines the steps for sentiment analysis using the Twitter and Reddit Sentiment dataset from Kaggle. It includes loading the data, performing basic preprocessing, sentiment labeling, and conducting visualizations to explore sentiment distributions. Further steps could involve building machine learning models to classify sentiment based on the provided data.
