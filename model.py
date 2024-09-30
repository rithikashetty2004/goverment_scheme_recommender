import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import re
import pickle

# Load the dataset (replace with your actual dataset path)
df = pd.read_csv('government_schemes_dataset.csv')

# Predefined genres and keywords for classification
genres = {
    'Healthcare': ['health', 'medical', 'hospital', 'ayushman'],
    'Education': ['education', 'school', 'learning', 'beti bachao', 'skill'],
    'Employment': ['job', 'employment', 'work', 'mgnrega', 'kaushal'],
    'Social Security': ['pension', 'assistance', 'social', 'welfare'],
    'Housing': ['housing', 'home', 'shelter', 'awas'],
    'Digital Empowerment': ['digital', 'technology', 'internet', 'computer'],
    'Financial Inclusion': ['bank', 'finance', 'loan', 'jan dhan', 'mudra'],
    'Women Empowerment': ['women', 'girl', 'matru', 'mahila'],
    'Rural Development': ['rural', 'village', 'gram', 'panchayat'],
    'Urban Development': ['urban', 'city', 'municipal', 'smart city'],
    'Agriculture': ['farm', 'crop', 'agriculture', 'kisan'],
    'Environment': ['environment', 'climate', 'pollution', 'green'],
    'Entrepreneurship': ['entrepreneur', 'startup', 'business', 'stand up india'],
    'Sanitation': ['sanitation', 'toilet', 'hygiene', 'swachh']
}

# Function to determine the genre based on scheme name
def determine_genre(scheme_name):
    scheme_name = scheme_name.lower()
    for genre, keywords in genres.items():
        if any(keyword in scheme_name for keyword in keywords):
            return genre
    return "General Welfare"

# Apply genre classification to the dataset
df['Genre'] = df['Scheme Name'].apply(determine_genre)

# Function to generate a description based on the scheme name and genre
def generate_description(scheme_name, genre):
    words = re.findall(r'\w+', scheme_name.lower())
    if 'pradhan' in words and 'mantri' in words:
        scheme_type = "flagship"
    else:
        scheme_type = "government"
    
    templates = [
        f"A {scheme_type} scheme in the {genre} sector, aimed at improving the lives of citizens through targeted interventions.",
        f"This {genre} initiative focuses on enhancing the welfare of the population through various {scheme_type} measures.",
        f"A comprehensive {scheme_type} program designed to address key issues in the {genre} domain and promote overall development.",
        f"An innovative approach to tackling challenges in the {genre} sector, this {scheme_type} scheme aims to bring about positive change.",
        f"Targeting the {genre} aspect of societal development, this {scheme_type} initiative strives to create a meaningful impact."
    ]
    
    return np.random.choice(templates)

# Generate descriptions for each scheme in the dataset
df['Description'] = df.apply(lambda row: generate_description(row['Scheme Name'], row['Genre']), axis=1)

# Function to get eligibility criteria
def get_eligibility_criteria(scheme_row):
    age = scheme_row['Applicable Age']
    gender = scheme_row['Gender']
    income_range = scheme_row['Income Range']
    
    eligibility = f"Applicable Age: {age}, Gender: {gender}, Income Range: {income_range}"
    return eligibility

# Create a text classifier to match input scheme names with known schemes
tfidf = TfidfVectorizer(stop_words='english')
model = make_pipeline(tfidf, MultinomialNB())
model.fit(df['Scheme Name'], df['Scheme Name'])  # We are training it to match scheme names

# Save the model and DataFrame to a pickle file
with open('government_scheme_model.pkl', 'wb') as f:
    pickle.dump((model, df), f)

print("Model and DataFrame saved to 'government_scheme_model.pkl'")
