# Singular Value Decomposition (SVD)

from skimage.color import rgb2gray
from skimage import data
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd

from collections import Counter
import re

# Matriz término documento

np.random.seed(42) # Semilla para reproducibilidad

# Definimos las dimensiones. 

# Términos
terms = ["Kholin", "scholar", "soldier", "healer", "ancient"] 

# Documento: Personajes del Archivo de las Tormentas y sus órdenes.
documents = [
    "Kaladin Stormblessed, a Windrunner. Soldier and surgeon.",
    "Shallan Davar, a Lightweaver. Noblewoman and scholar.",
    "Dalinar Kholin, a Bondsmith. Soldier and highprince.",
    "Venli, a Willshaper. Listener and scholar.",
    "Szeth, a Skybreaker. Truthless and assassin.",
    "Lift, a Edgedancer. Adventurer and healer.",
    "Renarin Kholin, a Truthwatcher. Highprince and healer.",
    "Shalash, Herald of the Lightweavers. Ancient and mysterious.",
    "Taln, Herald of the Stonewards. Ancient and broken.",
    "Jasnah Kholin, a Elsecaller. Queen and scholar."
]

def preprocess(text):
    # Convertir a minúsculas y extraer palabras
    text = text.lower()
    words = re.findall(r'\b[a-z]+\b', text)
    return words

print("0. DOCUMENTOS PREPROCESADOS:")
for doc in documents:
    print(preprocess(doc))
print("\n")

# Creamos el vocabuilario empleando set() para obtener las palabras únicas.
vocabulary = set()
for doc in documents:
    vocabulary.update(preprocess(doc))

vocabulary = sorted(vocabulary)  # Ordenamos el vocabulario
print(f"1. VOCABULARIO EXTRAÍDO: ({len(vocabulary)} palabras):\n", vocabulary)

### 1. Construcción de la matriz término-documento

def build_term_doc_matrix(documents, vocabulary):
    # Inicializamos una matriz de ceros
    matrix = np.zeros((len(documents), len(vocabulary)), dtype=int)
    
    for doc_idx, doc in enumerate(documents):
        words = preprocess(doc)
        word_count = Counter(words)

        for term_idx, term in enumerate(vocabulary):
            matrix[doc_idx, term_idx] = word_count.get(term, 0)
            # print(f"matrix[{doc_idx}, {term_idx}] = {word_count.get(term, 0)}")

    return matrix

# Asignamos la matriz término-documento
term_doc_matrix = build_term_doc_matrix(documents, vocabulary)

# Mostramos la matriz término-documento
print("\n2. MATRIZ TÉRMINO-DOCUMENTO:")
print(f"Dimensiones: {term_doc_matrix.shape}(términos x documentos)")
print(f"Primeras 10 filas y 15 columnas:")
print(term_doc_matrix[:10, :15])  # Mostrar solo una parte de la matriz para brevedad

'''

print("U: ", U)
print("Singular array", singular)
print("V^{T}", V_transpose)

singular_inv = 1.0 / singular
s_inv = np.zeros(X.shape)
s_inv[0][0] = singular_inv[0]
s_inv[1][1] = singular_inv[1]
M = np.dot(np.dot(V_transpose.T, s_inv.T), U.T)
print(M)

cat = data.chelsea()
plt.imshow(cat)

gray_cat = rgb2gray(cat)

U, S, V_T = svd(gray_cat, full_matrices=False)
S = np.diag(S)
fig, ax = plt.subplots(5, 2, figsize=(8, 20))

curr_fig = 0
for r in [5, 10, 70, 100, 200]:
    cat_approx = U[:, :r] @ S[0:r, :r] @ V_T[:r, :]
    ax[curr_fig][0].imshow(cat_approx, cmap='gray')
    ax[curr_fig][0].set_title("k = " + str(r))
    ax[curr_fig, 0].axis('off')
    ax[curr_fig][1].set_title("Original Image")
    ax[curr_fig][1].imshow(gray_cat, cmap='gray')
    ax[curr_fig, 1].axis('off')
    curr_fig += 1
plt.show()
'''