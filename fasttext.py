"""
fastText için temel fonksiyonlar.

Fikir: Kelimeyi tek bir sembol olarak görmek yerine karakter n-gram'larına böl.
Her n-gram'ın kendi vektörü vardır; kelimenin vektörü bu parçaların TOPLAMIdır.
Eğitim hâlâ negatif örneklemeli skip-gram'dır, tek fark merkez kelimenin vektörünün
bir tablo satırı değil, n-gram vektörlerinin toplamı olmasıdır.
"""
import numpy as np


# --------------------------------------------------------------------------
# 1) n-gram çıkarma
# --------------------------------------------------------------------------
def sinirli_kelime(kelime):
    """Başa '<', sona '>' konur: başta/sonda duran parçalar ortadakilerden ayırt edilir."""
    return f"<{kelime}>"


def ngramlar(kelime, min_n=3, max_n=6):
    """
    '<kelime>' üzerindeki bütün karakter n-gram'larını sırayla verir (n = min_n..max_n).
    Uzunluğu L olan dizide n-gram sayısı  L - n + 1  dir (n > L ise hiç yoktur).

    Not: Kısa kelimelerde n, '<kelime>' uzunluğuna eşit olabilir; o zaman parça
    '<kelime>'nin tamamıdır ve n-gram olarak yine sayılır (fastText ve gensim de öyle yapar).
    """
    w = sinirli_kelime(kelime)
    sonuc = []
    for n in range(min_n, max_n + 1):
        for i in range(len(w) - n + 1):
            sonuc.append(w[i:i + n])
    return sonuc


def ngram_sayisi(kelime, n):
    """Tek bir n için sayı: L - n + 1  (L = '<kelime>' uzunluğu)."""
    L = len(sinirli_kelime(kelime))
    return max(0, L - n + 1)


def vektor_sayisi(kelime, min_n=3, max_n=6):
    """Toplanacak vektör sayısı = n-gram sayısı + kelimenin kendisi."""
    return len(ngramlar(kelime, min_n, max_n)) + 1


# --------------------------------------------------------------------------
# 2) Kelime vektörü = parçaların toplamı
# --------------------------------------------------------------------------
def kelime_vektoru(kelime, ngram_tablosu, kelime_tablosu=None, min_n=3, max_n=6, ortalama=False):
    """
    Kelime vektörü = n-gram vektörlerinin toplamı (+ varsa kelimenin kendi vektörü).

    ngram_tablosu : {n-gram: vektör}  - eğitimde öğrenilmiş parçalar
    kelime_tablosu: {kelime: vektör}  - sözlükteki kelimenin kendi vektörü (isteğe bağlı)
    ortalama      : True ise toplam yerine ortalama (gensim böyle yapar; kosinüs
                    benzerliği vektör uzunluğundan etkilenmediği için fark etmez).

    Tabloda olmayan n-gram'lar atlanır. Hiçbir parça bulunamazsa None döner.
    """
    parcalar = [np.asarray(ngram_tablosu[g], dtype=float)
                for g in ngramlar(kelime, min_n, max_n) if g in ngram_tablosu]
    if kelime_tablosu is not None and kelime in kelime_tablosu:
        parcalar.append(np.asarray(kelime_tablosu[kelime], dtype=float))
    if not parcalar:
        return None
    toplam = np.sum(parcalar, axis=0)
    return toplam / len(parcalar) if ortalama else toplam


# --------------------------------------------------------------------------
# 3) Hash kovaları
# --------------------------------------------------------------------------
def fnv1a_hash(metin):
    """
    fastText'in n-gram'ları kovalara dağıtmak için kullandığı FNV-1a (32 bit).
    fastText baytları işaretli (int8) okur; ASCII dışı karakterlerde (ç, ş, ğ...) bu önemlidir.
    """
    h = 2166136261
    for bayt in metin.encode("utf-8"):
        isaretli = bayt - 256 if bayt > 127 else bayt
        h = (h ^ (isaretli & 0xFFFFFFFF)) & 0xFFFFFFFF
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def kova_indeksi(ngram, kova_sayisi=2_000_000):
    """n-gram'ın düştüğü kova. Farklı n-gram'lar aynı kovaya düşebilir (çakışma kabul edilir)."""
    return fnv1a_hash(ngram) % kova_sayisi


# --------------------------------------------------------------------------
# 4) Ortak n-gram'lar (yazım hatası / OOV örneği)
# --------------------------------------------------------------------------
def ortak_ngramlar(kelime_a, kelime_b, min_n=3, max_n=6):
    """İki kelimenin n-gram kümelerinin kesişimi."""
    return set(ngramlar(kelime_a, min_n, max_n)) & set(ngramlar(kelime_b, min_n, max_n))
