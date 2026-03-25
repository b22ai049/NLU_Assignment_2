from collections import Counter

# Load the cleaned corpus
with open("data/cleaned_corpus.txt", "r", encoding="utf-8") as f:
    words = f.read().split()

# Count the frequencies
word_counts = Counter(words)

# Get the top 10
top_10 = word_counts.most_common(10)

# Format for the report
output = ", ".join([f"{word}, {freq}" for word, freq in top_10])
print(output)