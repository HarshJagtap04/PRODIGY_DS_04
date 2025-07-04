import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Load the dataset
try:
    df = pd.read_csv('C:/Users/HARSH/OneDrive/Desktop/task_04/twitter_validation.csv')
    print("Original Columns in the DataFrame:", df.columns.tolist()) # Print all columns for verification
except FileNotFoundError:
    print("Error: 'twitter_validation.csv' not found. Please ensure the file is in the 'MultipleFiles' directory.")
    exit()

# --- Identify and Standardize Column Names ---
# Based on the provided context (e.g., 3364,Facebook,Irrelevant,"I mentioned..."),
# the columns appear to be: [ID, Brand, Sentiment, Text]
# Let's assume the column at index 2 is 'Sentiment' and index 1 is 'Brand'.

# Check if 'Sentiment' column exists, if not, try to infer or rename
if 'Sentiment' not in df.columns:
    # If 'Sentiment' is not found, try to use the column at index 2
    # This is a common pattern if the header is missing or named differently
    if len(df.columns) > 2:
        # Assuming the third column (index 2) is the sentiment
        df.rename(columns={df.columns[2]: 'Sentiment'}, inplace=True)
        print(f"Renamed column '{df.columns[2]}' to 'Sentiment'.")
    else:
        print("Error: 'Sentiment' column not found and cannot be inferred from column index.")
        print("Please check your CSV file's column names and structure.")
        exit()

# Check if 'Brand' column exists, if not, try to infer or rename
if 'Brand' not in df.columns:
    # Assuming the second column (index 1) is the brand/topic
    if len(df.columns) > 1:
        df.rename(columns={df.columns[1]: 'Brand'}, inplace=True)
        print(f"Renamed column '{df.columns[1]}' to 'Brand'.")
    else:
        print("Error: 'Brand' column not found and cannot be inferred from column index.")
        print("Please check your CSV file's column names and structure.")
        exit()

# Ensure 'text' column exists, if not, try to infer or rename
if 'text' not in df.columns:
    # Assuming the fourth column (index 3) is the text
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

# --- 1. Analyze Sentiment Distribution ---
print("--- Sentiment Distribution ---")
sentiment_counts = df['Sentiment'].value_counts()
print(sentiment_counts)
print("\n")

# --- 2. Visualize Sentiment Distribution ---
plt.figure(figsize=(8, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
plt.title('Overall Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.show()

# --- 3. Analyze Sentiment by Topic/Brand ---
print("--- Sentiment Distribution by Topic/Brand ---")
sentiment_by_brand = df.groupby('Brand')['Sentiment'].value_counts().unstack(fill_value=0)
print(sentiment_by_brand)
print("\n")

# --- 4. Visualize Sentiment by Topic/Brand ---
sentiment_by_brand.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='coolwarm')
plt.title('Sentiment Distribution Across Different Brands/Topics')
plt.xlabel('Brand/Topic')
plt.ylabel('Number of Tweets')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Sentiment')
plt.tight_layout()
plt.show()

# --- 5. Identify Most Frequent Words for Each Sentiment (Basic Text Analysis) ---
print("--- Most Frequent Words by Sentiment ---")

# Combine all text for each sentiment
text_by_sentiment = df.groupby('Sentiment')['text'].apply(lambda x: ' '.join(x.astype(str))).to_dict() # Ensure text is string

# Function to get most common words
def get_most_common_words(text, n=10):
    words = text.lower().split()
    # Remove punctuation and common stopwords (you might need a more comprehensive list)
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
