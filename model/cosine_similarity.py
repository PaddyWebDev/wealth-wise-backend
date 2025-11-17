import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import joblib

# Replace with your actual file name


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "mutual_fund_dataset.csv")
df = pd.read_csv(DATA_PATH)

columns = ['min_sip', 'risk_level', 'category', 'returns_1yr', 'returns_3yr', 'returns_5yr']
df = df[columns].copy()

# Fill missing values

# Convert to string and lowercase
for col in columns:
    df[col] = df[col].astype(str).str.lower()

# Combine features into one string for vectorization
def combine_features(row):
    return ' '.join([
        row['min_sip'],
        row['risk_level'],
        row['category'],
        row['returns_1yr'],
        row['returns_3yr'],
        row['returns_5yr']
    ])

df['combined_features'] = df.apply(combine_features, axis=1)

# Vectorize combined features
cv = TfidfVectorizer(stop_words='english')
vector_matrix = cv.fit_transform(df['combined_features'])
# print(cosine_similarity(vector_matrix))



joblib.dump(cv, "model/vectorizer.pkl")
joblib.dump(vector_matrix, "model/vector_matrix.pkl")
joblib.dump(df, "model/dataFrame.pkl")