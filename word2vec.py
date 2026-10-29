"""
word2vec (skip-gram + negatif örnekleme) için temel fonksiyonlar.

Fikir: Merkez kelimenin vektörü (w) ile bağlam kelimesinin vektörü (c) arasındaki
iç çarpım, "bu çift metinde gerçekten yan yana geçti mi?" sorusunun skorudur.
  - Gerçek çift   (pozitif): σ(c · w) -> 1 olmalı
  - Uydurma çift  (negatif): σ(c · w) -> 0 olmalı
"""
import numpy as np

from ortak import sigmoid


# --------------------------------------------------------------------------
# 1) Skip-gram çiftleri
# --------------------------------------------------------------------------
def skipgram_ciftleri(kelimeler, pencere=1):
    """
    Her kelimeyi merkez alıp solundaki ve sağındaki `pencere` kadar komşusuyla
    (merkez, bağlam) çifti kurar. Cümlenin uçlarındaki kelimelerin tek tarafı eksiktir.
    """
    ciftler = []
    n = len(kelimeler)
    for i, merkez in enumerate(kelimeler):
        for j in range(max(0, i - pencere), min(n, i + pencere + 1)):
            if j != i:
                ciftler.append((merkez, kelimeler[j]))
    return ciftler


# --------------------------------------------------------------------------
# 2) Maliyet karşılaştırması: tam softmax - negatif örnekleme
# --------------------------------------------------------------------------
def softmax(skorlar):
    """P_i = exp(y_i) / Σ_j exp(y_j)   (payda: sözlükteki HER kelimenin skoru)"""
    s = np.asarray(skorlar, dtype=float)
    e = np.exp(s - s.max())  # taşmayı önlemek için en büyüğü çıkarılır
    return e / e.sum()


def tam_softmax_maliyeti(sozluk_boyutu, vektor_boyutu):
    """Bir adımda sözlükteki her kelime için bir iç çarpım: |V| x d çarpma."""
    return sozluk_boyutu * vektor_boyutu


def negatif_ornekleme_maliyeti(k, vektor_boyutu):
    """Bir adımda (1 pozitif + k negatif) kelime için iç çarpım: (k + 1) x d çarpma."""
    return (k + 1) * vektor_boyutu


# --------------------------------------------------------------------------
# 3) Negatifleri çekme: ¾ kuralı ve alt örnekleme
# --------------------------------------------------------------------------
def uc_dort_dagilimi(frekanslar, us=0.75):
    """
    Negatif kelimeler ham sıklıkla değil, sıklığın ¾ kuvvetiyle çekilir:
        P(w) = f(w)^0.75 / Σ f(v)^0.75
    Nadir kelimelerin şansı artar, sık kelimelerinki biraz azalır.
    """
    f = np.asarray(frekanslar, dtype=float)
    f = f / f.sum()  # önce olasılığa çevir
    kuvvet = f ** us
    return kuvvet / kuvvet.sum()


def alt_ornekleme_atma_olasiligi(frekans, t=1e-5):
    """
    Çok sık kelimeler eğitim sırasında rastgele atlanır (hız hilesi):
        P(at) = 1 - sqrt(t / f(w))
    f(w): kelimenin göreli sıklığı, t: eşik (makalede ~1e-5).
    Sıklık eşikten küçükse atılmaz.
    """
    return max(0.0, 1.0 - np.sqrt(t / frekans))


# --------------------------------------------------------------------------
# 4) Negatif örnekleme kaybı, gradyanlar, güncelleme
# --------------------------------------------------------------------------
def negatif_ornekleme_kaybi(w, c_poz, c_negler):
    """
    L = -[ log σ(c_poz · w) + Σ_neg log σ(-c_neg · w) ]

    Döndürür: (toplam kayıp, ayrıntı listesi). Ayrıntı: (ad, iç çarpım, σ, ceza).
    """
    w = np.asarray(w, dtype=float)
    ayrinti = []

    s = float(np.dot(c_poz, w))
    sig = float(sigmoid(s))
    ayrinti.append(("pozitif", s, sig, -np.log(sig)))

    for i, c in enumerate(c_negler, start=1):
        s = float(np.dot(c, w))
        sig = float(sigmoid(-s))  # negatiflerde işaret ters çevrilir
        ayrinti.append((f"negatif {i}", s, sig, -np.log(sig)))

    return sum(a[3] for a in ayrinti), ayrinti


def negatif_ornekleme_gradyanlari(w, c_poz, c_negler):
    """
    dL/dc_poz  = (σ(c_poz · w) - 1) · w
    dL/dc_neg  = σ(c_neg · w) · w
    dL/dw      = (σ(c_poz · w) - 1) · c_poz + Σ σ(c_neg · w) · c_neg
    """
    w = np.asarray(w, dtype=float)
    c_poz = np.asarray(c_poz, dtype=float)

    sp = float(sigmoid(np.dot(c_poz, w)))
    g_c_poz = (sp - 1.0) * w
    g_w = (sp - 1.0) * c_poz

    g_c_negler = []
    for c in c_negler:
        c = np.asarray(c, dtype=float)
        sn = float(sigmoid(np.dot(c, w)))
        g_c_negler.append(sn * w)
        g_w = g_w + sn * c

    return g_w, g_c_poz, g_c_negler


def adim_at(w, c_poz, c_negler, ogrenme_orani):
    """
    Tek bir gradyan inişi adımı:  θ <- θ - η · gradyan
    Gradyanların hepsi ESKİ vektörlerle hesaplanır, sonra hep birlikte güncellenir.
    """
    g_w, g_c_poz, g_c_negler = negatif_ornekleme_gradyanlari(w, c_poz, c_negler)
    yeni_w = np.asarray(w, dtype=float) - ogrenme_orani * g_w
    yeni_c_poz = np.asarray(c_poz, dtype=float) - ogrenme_orani * g_c_poz
    yeni_negler = [np.asarray(c, dtype=float) - ogrenme_orani * g
                   for c, g in zip(c_negler, g_c_negler)]
    return yeni_w, yeni_c_poz, yeni_negler
