"""
Gömmelerin değerlendirilmesi: analoji (vektör aritmetiği) ve Spearman sıra korelasyonu.

1) Analoji:  "a : b :: c : ?"   (ör. ev : evler :: kitap : ?)
       sorgu = b - a + c      ->  kosinüsü sorguya en yakın kelime (girdi kelimeleri HARİÇ)

2) Spearman ρ: insanların kelime çiftlerine verdiği benzerlik puanlarının SIRALAMASI ile
   modelin kosinüs benzerliklerinin SIRALAMASI ne kadar uyuşuyor?  (+1 tam uyum, 0 ilişki yok,
   -1 tam ters)
"""
import numpy as np

from ortak import kosinus_benzerligi


# --------------------------------------------------------------------------
# 1) Analoji
# --------------------------------------------------------------------------
def analoji_sorgusu(a, b, c, normalize=False):
    """
    sorgu = b - a + c
    b - a, "a'dan b'ye geçiş yönü"dür (ör. çoğul yapma); bu yön c'ye eklenir.

    normalize=True: önce her vektör birim uzunluğa getirilir. gensim'in most_similar'ı
    böyle çalışır (uzun vektörler sorguyu domine etmesin diye).
    """
    a, b, c = (np.asarray(v, dtype=float) for v in (a, b, c))
    if normalize:
        a, b, c = (v / np.linalg.norm(v) for v in (a, b, c))
    return b - a + c


def kosinus_siralamasi(sorgu, vektorler, haric=()):
    """
    Sözlükteki her kelimenin sorguya kosinüs benzerliğini hesaplar, büyükten küçüğe sıralar.
    `haric`: sonuçtan atılacak kelimeler.
    Döndürür: [(kelime, kosinüs), ...]
    """
    sonuc = [(k, kosinus_benzerligi(sorgu, v)) for k, v in vektorler.items() if k not in haric]
    return sorted(sonuc, key=lambda x: -x[1])


def analoji_coz(a, b, c, vektorler, girdileri_at=True, normalize=False):
    """
    a : b :: c : ?   (a, b, c kelime; vektorler: {kelime: vektör})

    girdileri_at=True : a, b, c sonuçtan çıkarılır (gensim varsayılanı).
    girdileri_at=False: çıkarılmaz; sorgu girdilerden birine çok yakın kaldığı için çoğu zaman
                        ilk sıra bir girdi kelimesi olur (Nissim vd., 2020).
    Döndürür: (sorgu vektörü, sıralı [(kelime, kosinüs), ...])
    """
    sorgu = analoji_sorgusu(vektorler[a], vektorler[b], vektorler[c], normalize)
    haric = (a, b, c) if girdileri_at else ()
    return sorgu, kosinus_siralamasi(sorgu, vektorler, haric)


# --------------------------------------------------------------------------
# 2) Spearman sıra korelasyonu
# --------------------------------------------------------------------------
def siralar(degerler, buyukten_kucuge=True):
    """
    Her değerin sırasını verir (1 = en büyük). Eşit değerler SIRALARIN ORTALAMASINI alır:
    örn. 2. ve 3. sıradaki iki eşit değerin ikisi de 2.5 olur.
    """
    d = np.asarray(degerler, dtype=float)
    isaret = -d if buyukten_kucuge else d
    sirali_idx = np.argsort(isaret, kind="mergesort")
    sira = np.empty(len(d))
    i = 0
    while i < len(d):
        j = i
        while j + 1 < len(d) and isaret[sirali_idx[j + 1]] == isaret[sirali_idx[i]]:
            j += 1
        ortalama = (i + j) / 2 + 1  # i..j konumlarının (1'den başlayan) ortalama sırası
        sira[sirali_idx[i:j + 1]] = ortalama
        i = j + 1
    return sira


def pearson(x, y):
    """r = Σ(x - x̄)(y - ȳ) / sqrt( Σ(x - x̄)² · Σ(y - ȳ)² )"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm, ym = x - x.mean(), y - y.mean()
    return float(np.sum(xm * ym) / np.sqrt(np.sum(xm ** 2) * np.sum(ym ** 2)))


def spearman(x, y):
    """
    Spearman ρ = sıralar üzerinde hesaplanan Pearson korelasyonu.
    Eşitlikler varsa da doğru sonucu verir.
    """
    return pearson(siralar(x), siralar(y))


def spearman_d2(x, y):
    """
    Kısa formül:  ρ = 1 - 6 Σ d² / ( n (n² - 1) ),  d = iki sıranın farkı.
    YALNIZCA eşitlik (aynı değerli çift) yokken tam doğrudur.
    Döndürür: (ρ, d listesi, Σd²)
    """
    sx, sy = siralar(x), siralar(y)
    d = sx - sy
    n = len(d)
    toplam = float(np.sum(d ** 2))
    return 1.0 - 6.0 * toplam / (n * (n ** 2 - 1)), d, toplam


def cift_degerlendir(ciftler, insan_puanlari, vektor_getir):
    """
    AnlamVer benzeri değerlendirme:
      ciftler        : [(kelime1, kelime2), ...]
      insan_puanlari : her çift için insan benzerlik puanı
      vektor_getir   : kelime -> vektör, vektör yoksa None   (word2vec: sözlük araması,
                       fastText: n-gram toplamı)
    Vektörü bulunamayan çiftler atlanır. Döndürür: (ρ, kullanılan çift sayısı, toplam çift sayısı,
    kosinüs listesi)
    """
    kos, puan = [], []
    for (k1, k2), p in zip(ciftler, insan_puanlari):
        v1, v2 = vektor_getir(k1), vektor_getir(k2)
        if v1 is None or v2 is None:
            continue
        kos.append(kosinus_benzerligi(v1, v2))
        puan.append(p)
    rho = spearman(puan, kos) if len(kos) >= 2 else float("nan")
    return rho, len(kos), len(ciftler), kos
