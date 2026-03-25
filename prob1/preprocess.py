import re
import nltk
import os
from PyPDF2 import PdfReader
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download necessary NLTK data
nltk.download('punkt')

def preprocess_text(text):
    # (iii) Lower-casing
    text = text.lower()
    
    # (i) Removal of boilerplate and formatting artifacts
    # Removes and Page markers found in your specific files
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'--- page \d+ ---', '', text)
    
    # (iv) Removal of non-textual content and excessive punctuation
    # Keeps only English letters and whitespace
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # (ii) Tokenization
    tokens = word_tokenize(text)
    
    # Filter out noise (single characters)
    tokens = [t for t in tokens if len(t) > 1]
    
    return tokens

# List of your 3 source files
pdf_files = [
    "data/1_Academic_Regulations_Final_03_09_2019.pdf",
    "data/PhysRevB.110.134422.pdf",
    "data/5. B.Tech AIDS.pdf"
]

all_tokens = []
total_docs = len(pdf_files)

print("--- PROCESSING FILES ---")
for file_path in pdf_files:
    if os.path.exists(file_path):
        print(f"Reading: {file_path}")
        reader = PdfReader(file_path)
        raw_text = ""
        for page in reader.pages:
            raw_text += page.extract_text()
        
        # Clean and add to the master list
        file_tokens = preprocess_text(raw_text)
        all_tokens.extend(file_tokens)
    else:
        print(f"Warning: {file_path} not found!")

# 3. Report Dataset Statistics (Required for Task-1)
clean_corpus = " ".join(all_tokens)
vocab = set(all_tokens)

print("\n--- TASK-1: DATASET STATISTICS ---")
print(f"Total Documents: {total_docs}")
print(f"Total Tokens: {len(all_tokens)}")
print(f"Vocabulary Size: {len(vocab)}")

# 4. Save Cleaned Corpus for Task-2 (Word2Vec)
with open("data/cleaned_corpus.txt", "w", encoding="utf-8") as f:
    f.write(clean_corpus)
print("\nSuccess: 'cleaned_corpus.txt' created for Task-2.")

# 5. Generate Word Cloud
wordcloud = WordCloud(width=1000, height=500, background_color='white').generate(clean_corpus)
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Combined IIT Jodhpur Corpus Word Cloud")
plt.show()