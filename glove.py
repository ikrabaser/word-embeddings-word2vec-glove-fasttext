"""
GloVe için temel fonksiyonlar.

Fikir: word2vec metni pencerelerle gezip TAHMİN eder; GloVe önce bütün derlemde
kim kiminle kaç kez yan yana geçmiş SAYAR, sonra vektörleri bu sayılara uydurur:

    w_i · w~_j + b_i + b~_j  ≈  log X_ij
"""
import numpy as np


# --------------------------------------------------------------------------
# 1) Birlikte-görülme sayımları
# --------------------------------------------------------------------------
def birlikte_gorulme_matrisi(cumleler, pencere=1):
    """
    X[i, j]: j kelimesi, i kelimesinin penceresinde kaç kez geçti.
    Sayım cümle içinde yapılır, simetriktir (X[i, j] = X[j, i]).

    Döndürür: (sözlük listesi, X matrisi)
    """
    sozluk = sorted({k for c in cumleler for k in c.split()})
    indeks = {k: i for i, k in enumerate(sozluk)}
    X = np.zeros((len(sozluk), len(sozluk)))
    for cumle in cumleler:
        kelimeler = cumle.split()
        for i, merkez in enumerate(kelimeler):
            for j in range(max(0, i - pencere), min(len(kelimeler), i + pencere + 1)):
                if j != i:
                    X[indeks[merkez], indeks[kelimeler[j]]] += 1
    return sozluk, X


def olasilik_matrisi(X):
    """P(j | i) = X(i, j) / X(i),  X(i) = i satırının toplamı (MLE: sayım / toplam)."""
    satir_toplami = X.sum(axis=1, keepdims=True)
    return np.divide(X, satir_toplami, out=np.zeros_like(X), where=satir_toplami > 0)


def olasilik_orani(P, i, j, k):
    """
    P(k | i) / P(k | j)
      ≈ 1   -> k kelimesi ikisiyle de ilgili ya da ikisiyle de ilgisiz (ortak etki sadeleşir)
      >> 1  -> k, i'ye özgü
      << 1  -> k, j'ye özgü
    """
    return P[i, k] / P[j, k]


# --------------------------------------------------------------------------
# 2) Ağırlık fonksiyonu f(X_ij)
# --------------------------------------------------------------------------
def agirlik(x, x_max=100.0, alfa=0.75):
    """
    f(x) = (x / x_max)^alfa   x < x_max ise
         = 1                  aksi halde  (çok sık çiftlerin ağırlığı 1'de sabitlenir)
    f(0) = 0: hiç görülmemiş çift (log 0 tanımsız) toplamdan zaten çıkar.
    """
    x = float(x)
    if x <= 0:
        return 0.0
    return (x / x_max) ** alfa if x < x_max else 1.0


# --------------------------------------------------------------------------
# 3) Kayıp fonksiyonu
# --------------------------------------------------------------------------
def cift_kaybi(tahmin, x_ij):
    """
    Tek bir (i, j) çifti için ağırlıklı kare hata:
        f(X_ij) · ( w_i·w~_j + b_i + b~_j - log X_ij )²

    `tahmin`: modelin w_i·w~_j + b_i + b~_j değeri.
    X_ij = 0 ise None döner (çift atlanır).
    Döndürür: (hedef, fark, kare, f, kayıp)
    """
    if x_ij <= 0:
        return None
    hedef = float(np.log(x_ij))
    fark = tahmin - hedef
    f = agirlik(x_ij)
    return hedef, fark, fark ** 2, f, f * fark ** 2


def tahmin_skoru(w_i, w_j_tilde, b_i, b_j_tilde):
    """Modelin skoru: w_i · w~_j + b_i + b~_j"""
    return float(np.dot(w_i, w_j_tilde) + b_i + b_j_tilde)


def toplam_kayip(tahminler, x_degerleri):
    """J = Σ f(X_ij) (tahmin - log X_ij)²   (X_ij = 0 olan çiftler atlanır)"""
    toplam = 0.0
    for t, x in zip(tahminler, x_degerleri):
        sonuc = cift_kaybi(t, x)
        if sonuc is not None:
            toplam += sonuc[-1]
    return toplam
