import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

try:
    df = pd.read_csv('C:/Users/HARSH/OneDrive/Desktop/task_04/twitter_validation.csv')
    print("Original Columns in the DataFrame:", df.columns.tolist()) # Print all columns for verification
except FileNotFoundError:
    print("Error: 'twitter_validation.csv' not found. Please ensure the file is in the 'MultipleFiles' directory.")
    exit()



if 'Sentiment' not in df.columns:
    
    if len(df.columns) > 2:
        df.rename(columns={df.columns[2]: 'Sentiment'}, inplace=True)
        print(f"Renamed column '{df.columns[2]}' to 'Sentiment'.")
    else:
        print("Error: 'Sentiment' column not found and cannot be inferred from column index.")
        print("Please check your CSV file's column names and structure.")
        exit()

if 'Brand' not in df.columns:
    if len(df.columns) > 1:
        df.rename(columns={df.columns[1]: 'Brand'}, inplace=True)
        print(f"Renamed column '{df.columns[1]}' to 'Brand'.")
    else:
        print("Error: 'Brand' column not found and cannot be inferred from column index.")
        print("Please check your CSV file's column names and structure.")
        exit()

if 'text' not in df.columns:
    if len(df.columns) > 3:
        df.rename(columns={df.columns[3]: 'text'}, inplace=True)
        print(f"Renamed column '{df.columns[3]}' to 'text'.")
    else:
        print("Error: 'text' column not found and cannot be inferred from column index.")
        print("Please check your CSV file's column names and structure.")
        exit()


print("Columns after standardization:", df.columns.tolist())
print("First few rows of the DataFrame:")
print(df.head())
print("\n")

print("--- Sentiment Distribution ---")
sentiment_counts = df['Sentiment'].value_counts()
print(sentiment_counts)
print("\n")

plt.figure(figsize=(8, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
plt.title('Overall Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.show()

print("--- Sentiment Distribution by Topic/Brand ---")
sentiment_by_brand = df.groupby('Brand')['Sentiment'].value_counts().unstack(fill_value=0)
print(sentiment_by_brand)
print("\n")

sentiment_by_brand.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='coolwarm')
plt.title('Sentiment Distribution Across Different Brands/Topics')
plt.xlabel('Brand/Topic')
plt.ylabel('Number of Tweets')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Sentiment')
plt.tight_layout()
plt.show()

print("--- Most Frequent Words by Sentiment ---")

text_by_sentiment = df.groupby('Sentiment')['text'].apply(lambda x: ' '.join(x.astype(str))).to_dict() # Ensure text is string

def get_most_common_words(text, n=10):
    words = text.lower().split()
    words = [word.strip('.,!?"\'()[]{}') for word in words if word.isalpha()]
    stopwords = set(
        ['the', 'a', 'an', 'is', 'it', 'and', 'to', 'of', 'for', 'in', 'on', 'with', 'that', 'this', 'i', 'you', 'he', 'she', 'we', 'they', 'be', 'have', 'do', 'not', 'but', 'so', 'just', 'my', 'me', 'your', 'from', 'by', 'at', 'as', 'can', 'will', 'would', 'get', 'like', 'what', 'when', 'where', 'why', 'how', 'if', 'or', 'no', 'yes', 'out', 'up', 'down', 'go', 'see', 'one', 'all', 'about', 'there', 'their', 'their', 'some', 'any', 'very', 'much', 'more', 'than', 'then', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
    )
    words = [word for word in words if word not in stopwords and len(word) > 2] # Filter out short words too
    return Counter(words).most_common(n)

for sentiment, text in text_by_sentiment.items():
    print(f"Top 10 words for {sentiment} sentiment:")
    print(get_most_common_words(text))
    print("\n")
