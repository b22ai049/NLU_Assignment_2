from gensim.models import Word2Vec

# 1. Load your model 
# (Ensure your Task-2 training used vector_size=300 for this specific task)
model = Word2Vec.load("data/final_word2vec.model")

# 2. Select a word from your vocabulary (other than 'jodhpur')
target_word = "research"

if target_word in model.wv:
    vector = model.wv[target_word]
    
    # 3. Format as a comma-separated list
    vector_str = ", ".join([f"{val:.4f}" for val in vector])
    
    print(f"{target_word} - {vector_str}")
    
    # Optional: Save to a text file for your report
    with open("data/word_vector_300d.txt", "w") as f:
        f.write(f"{target_word} - {vector_str}")
else:
    print(f"Word '{target_word}' not found in vocabulary.")