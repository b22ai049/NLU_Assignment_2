import gensim
from gensim.models import Word2Vec
import pandas as pd

# Load corpus
with open("data/cleaned_corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()
sentences = [text.split()]

# TASK-2: Hyperparameter Experimentation Grid
dimensions = [50, 100, 200]    # (i) Embedding dimensions
windows = [2, 5, 10]           # (ii) Context window sizes
neg_samples = [5, 15]          # (iii) Number of negative samples

results = []

print("--- TASK-2: RUNNING HYPERPARAMETER EXPERIMENTS ---")

for d in dimensions:
    for w in windows:
        for ns in neg_samples:
            # Training Skip-gram (sg=1) as the baseline for experiments
            model = Word2Vec(
                sentences, 
                vector_size=d, 
                window=w, 
                negative=ns, 
                sg=1, 
                min_count=1, 
                workers=4,
                epochs=10
            )
            
            # Simple evaluation: Similarity of a core pair
            try:
                score = model.wv.similarity('research', 'phd')
            except:
                score = 0
            
            results.append({
                "Dimension": d,
                "Window": w,
                "Neg_Samples": ns,
                "Res-PhD_Sim": round(score, 4)
            })
            print(f"Tested: Dim={d}, Win={w}, Neg={ns} -> Sim={score:.4f}")

# Save the best model for Task-3
model.save("data/final_word2vec.model")

# Create a Table for your Report
df = pd.DataFrame(results)
df.to_csv("data/task2_experiments.csv", index=False)
print("\nExperiment results saved to 'data/task2_experiments.csv'.")
