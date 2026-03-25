import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
import numpy as np

# 1. Load your best model
model = Word2Vec.load("data/final_word2vec.model")

# 2. Select words to visualize (Cluster them by category)
clusters = {
    'Degrees': ['btech', 'mtech', 'phd', 'msc', 'ug', 'pg'],
    'Academic': ['exam', 'quiz', 'credits', 'semester', 'course', 'registration'],
    'Research': ['research', 'laboratory', 'thesis', 'project', 'presentation'],
    'Departments': ['mechanical', 'electrical', 'computer', 'science', 'humanities']
}

all_words = [word for cluster in clusters.values() for word in cluster]
# Ensure words exist in vocab
words_in_vocab = [w for w in all_words if w in model.wv.key_to_index]

# 3. Get vectors and reduce to 2D
word_vectors = np.array([model.wv[w] for w in words_in_vocab])
tsne = TSNE(n_components=2, random_state=42, perplexity=len(words_in_vocab)-1)
vectors_2d = tsne.fit_transform(word_vectors)

# 4. Plot
plt.figure(figsize=(12, 8))
for i, word in enumerate(words_in_vocab):
    # Color coding based on our clusters
    color = 'black'
    for label, cluster_words in clusters.items():
        if word in cluster_words:
            if label == 'Degrees': color = 'blue'
            if label == 'Academic': color = 'red'
            if label == 'Research': color = 'green'
            if label == 'Departments': color = 'orange'
    
    plt.scatter(vectors_2d[i, 0], vectors_2d[i, 1], c=color)
    plt.annotate(word, (vectors_2d[i, 0] + 0.1, vectors_2d[i, 1] + 0.1))

plt.title("t-SNE Visualization of IIT Jodhpur Word Embeddings")
plt.grid(True)
plt.show()