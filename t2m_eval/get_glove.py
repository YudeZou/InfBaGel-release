import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

embeddings_dict = {}

with open("/mnt/petrelfs/zouyude/code/lingo-release/t2m_eval/glove.840B.300d.txt", 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector
np.save('./glove_data//wordsList', np.array(list(embeddings_dict.keys())))
np.save('./glove_data//wordVectors', np.array(list(embeddings_dict.values()), dtype='float32'))
