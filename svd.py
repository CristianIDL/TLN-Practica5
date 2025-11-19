# Singular Value Decomposition (SVD)

from skimage.color import rgb2gray
from skimage import data
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd

import pandas as pd
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

df = pd.DataFrame(term_doc_matrix, columns=vocabulary)
df.to_csv('Matriz_termino_documento.csv', index=False, encoding='utf-8')

### 2. Aplicación de SVD

U, singular, V_transpose = svd(term_doc_matrix)

print("\n3. SVD:")
print("\nU (Matriz de Términos): \n", U.shape)
print("\n",U[:10, :5])
df = pd.DataFrame(U, columns=[f'Doc_{i}' for i in range(len(documents))])
df.to_csv('Matriz_de_terminos.csv', index=False, encoding='utf-8')

print("\nS (Valores Singulares): \n", singular.shape)
print("\n",singular[:10])
df = pd.DataFrame(singular, columns=["Valores Singulares"])
df.to_csv('Valores_singulares.csv', index=False, encoding='utf-8')

print("\nV^T (Matriz transpuesta): \n", V_transpose.shape)
print("\n",V_transpose)
df = pd.DataFrame(V_transpose, columns=vocabulary)
df.to_csv('Matriz_de_documentos_transpuesta.csv', index=False, encoding='utf-8')