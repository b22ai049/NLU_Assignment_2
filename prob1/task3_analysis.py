from gensim.models import Word2Vec

# Load the best model from your Task-2 experiments
model = Word2Vec.load("data/final_word2vec.model")

# 1. Top 5 nearest neighbors (Task-3.1)
target_words = ['research', 'student', 'phd', 'exam']

print("--- TASK-3.1: SEMANTIC SIMILARITY (Top 5 Neighbors) ---")
for word in target_words:
    try:
        print(f"\nNeighbors for '{word}':")
        # Cosine similarity is used by most_similar() by default
        results = model.wv.most_similar(word, topn=5)
        for neighbor, score in results:
            print(f"  - {neighbor}: {score:.4f}")
    except KeyError:
        print(f"  - '{word}' not found in the vocabulary.")

# 2. Analogy Experiments (Task-3.2)
print("\n--- TASK-3.2: ANALOGY EXPERIMENTS ---")

def solve_analogy(w1, w2, w3):
    # Analogy: w1 : w2 :: w3 : ? (e.g., UG : BTech :: PG : ?)
    # Positive are the words to add (w2, w3), negative is the word to subtract (w1)
    try:
        result = model.wv.most_similar(positive=[w2, w3], negative=[w1], topn=1)
        print(f"Analogy: {w1} : {w2} :: {w3} : {result[0][0]}")
    except KeyError as e:
        print(f"Could not perform analogy: {e}")

# Requirement: UG : BTech :: PG : ?
solve_analogy('ug', 'btech', 'pg')

# Additional IITJ specific analogies
solve_analogy('course', 'credits', 'project')
solve_analogy('semester', 'exam', 'summer')