"""
Üç yöntemin (word2vec, GloVe, fastText) ortak kullandığı küçük fonksiyonlar.
"""
import numpy as np


def sigmoid(x):
    """σ(x) = 1 / (1 + e^(-x))  ->  her sayıyı (0, 1) aralığına sıkıştırır."""
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def kosinus_benzerligi(a, b):
    """
    cos(a, b) = (a · b) / (|a| |b|)

    İç çarpımdan farkı: vektörün UZUNLUĞUNDAN etkilenmez, yalnızca YÖNÜNE bakar.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def one_hot(indeks, boyut):
    """Tek yerinde (indeks) 1, geri kalan her yerinde 0 olan vektör."""
    v = np.zeros(boyut)
    v[indeks] = 1.0
    return v
