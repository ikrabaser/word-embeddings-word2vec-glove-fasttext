"""
README'deki el hesabı sayılarının kodla tutup tutmadığını denetler.

Çalıştırma:   python test_el_hesaplari.py        (ya da:  pytest)
gensim kuruluysa fastText n-gram ve hash değerleri gensim ile de karşılaştırılır.
"""
import numpy as np

import degerlendirme as dg
import fasttext as ft
import glove
import word2vec as w2v
from ortak import kosinus_benzerligi, one_hot

W = np.array([1.0, 1.0])
C_POZ = np.array([0.5, -0.5])
C_NEGLER = [np.array([0.5, 0.5]), np.array([-1.0, 0.0])]


# ------------------------------- word2vec ------------------------------------
def test_one_hot_ic_carpim_sifir():
    assert np.dot(one_hot(0, 5), one_hot(1, 5)) == 0
    assert np.dot(one_hot(0, 5), one_hot(2, 5)) == 0


def test_skipgram_cift_sayisi():
    ciftler = w2v.skipgram_ciftleri(["sabah", "sıcak", "çay", "içtim"], pencere=1)
    assert len(ciftler) == 6


def test_maliyet_orani():
    tam = w2v.tam_softmax_maliyeti(300_000, 300)
    neg = w2v.negatif_ornekleme_maliyeti(5, 300)
    assert tam == 90_000_000 and neg == 1_800 and tam / neg == 50_000


def test_uc_dort_kurali():
    p = w2v.uc_dort_dagilimi([0.99, 0.01])
    assert abs(p[0] - 0.97) < 0.005 and abs(p[1] - 0.03) < 0.005
    assert 3.0 < p[1] / 0.01 < 3.2


def test_negatif_ornekleme_kaybi():
    L, ayrinti = w2v.negatif_ornekleme_kaybi(W, C_POZ, C_NEGLER)
    assert abs(ayrinti[0][3] - 0.693) < 1e-3
    assert abs(ayrinti[1][3] - 1.313) < 1e-3
    assert abs(ayrinti[2][3] - 0.313) < 1e-3
    assert abs(L - 2.32) < 5e-3


def test_gradyan_sayisal_turevle_ayni():
    """Analitik gradyan, sonlu farklarla bulunan türevle tutmalı."""
    g_w, _, _ = w2v.negatif_ornekleme_gradyanlari(W, C_POZ, C_NEGLER)
    eps = 1e-6
    for i in range(2):
        d = np.zeros(2)
        d[i] = eps
        sayisal = (w2v.negatif_ornekleme_kaybi(W + d, C_POZ, C_NEGLER)[0]
                   - w2v.negatif_ornekleme_kaybi(W - d, C_POZ, C_NEGLER)[0]) / (2 * eps)
        assert abs(sayisal - g_w[i]) < 1e-6
    assert abs(g_w[0] - (-0.153)) < 1e-3 and abs(g_w[1] - 0.616) < 1e-3


def test_tek_adim_guncelleme():
    yw, yc, yn = w2v.adim_at(W, C_POZ, C_NEGLER, 0.5)
    assert abs(np.dot(yc, yw) - 0.63) < 5e-3          # çay · içtim
    assert abs(np.dot(yn[0], yw) - 0.24) < 5e-3       # çay · masa
    L1, _ = w2v.negatif_ornekleme_kaybi(yw, yc, yn)
    assert abs(L1 - 1.48) < 5e-3


# --------------------------------- GloVe -------------------------------------
def test_glove_cift_kaybi():
    hedef, fark, kare, f, kayip = glove.cift_kaybi(4.2, 100)
    assert abs(hedef - 4.605) < 1e-3 and abs(fark - (-0.405)) < 1e-3
    assert abs(kayip - 0.164) < 1e-3 and f == 1.0


def test_glove_x_sifir_atlanir():
    assert glove.cift_kaybi(1.0, 0) is None
    assert glove.toplam_kayip([1.0, 4.2], [0, 100]) == glove.cift_kaybi(4.2, 100)[-1]


def test_glove_agirlik():
    assert abs(glove.agirlik(1) - 0.03) < 5e-3
    assert abs(glove.agirlik(10) - 0.18) < 5e-3
    assert glove.agirlik(100) == 1.0 and glove.agirlik(5000) == 1.0 and glove.agirlik(0) == 0.0


def test_glove_olasilik_satir_toplami_bir():
    sozluk, X = glove.birlikte_gorulme_matrisi(["sabah sıcak çay içtim", "sıcak çay içtim"])
    P = glove.olasilik_matrisi(X)
    assert np.allclose(P.sum(axis=1), 1.0)


# -------------------------------- fastText -----------------------------------
def test_ngram_sayilari():
    assert len(ft.ngramlar("kitap")) == 14 and ft.vektor_sayisi("kitap") == 15
    assert len(ft.ngramlar("evlerimizden")) == 42 and ft.vektor_sayisi("evlerimizden") == 43
    assert ft.ngramlar("evlerimizden", 3, 3) == [
        "<ev", "evl", "vle", "ler", "eri", "rim", "imi", "miz", "izd", "zde", "den", "en>"]


def test_kucuk_ornek_toplam():
    tablo = {"<ev": [1.0, 0.0], "ev>": [0.0, 1.0]}
    v = ft.kelime_vektoru("ev", tablo, {"ev": [1.0, 1.0]}, 3, 3)
    assert np.allclose(v, [2.0, 2.0])


def test_yazim_hatasi_ortak_parcalar():
    assert len(ft.ngramlar("bilgisyar")) == 30
    assert len(ft.ortak_ngramlar("bilgisyar", "bilgisayar")) == 17


def test_oov_icin_vektor_uretilir():
    tablo = {g: np.ones(3) for g in ft.ngramlar("bilgisayar")}
    assert ft.kelime_vektoru("bilgisyar", tablo) is not None      # sözlükte yok ama vektör var
    assert ft.kelime_vektoru("xqzw", tablo) is None               # hiç ortak parça yoksa yok


def test_kosinus_uzunluktan_bagimsiz():
    a = np.array([1.0, 2.0])
    assert abs(kosinus_benzerligi(a, 5 * a) - 1.0) < 1e-12


# ------------------------------ değerlendirme --------------------------------
VEK = {
    "ev": np.array([2.0, 1.0]), "evler": np.array([2.0, 3.0]), "kitap": np.array([5.0, 1.0]),
    "kitaplar": np.array([4.5, 3.5]), "masa": np.array([-1.0, 2.0]), "kalem": np.array([6.0, 0.5]),
}


def test_analoji_sorgusu():
    q = dg.analoji_sorgusu(VEK["ev"], VEK["evler"], VEK["kitap"])
    assert np.allclose(q, [5.0, 3.0])


def test_analoji_girdileri_atmak():
    _, atilmis = dg.analoji_coz("ev", "evler", "kitap", VEK, girdileri_at=True)
    _, atilmamis = dg.analoji_coz("ev", "evler", "kitap", VEK, girdileri_at=False)
    assert atilmis[0][0] == "kitaplar" and abs(atilmis[0][1] - 0.9927) < 5e-4
    assert atilmamis[0][0] == "ev" and abs(atilmamis[0][1] - 0.9971) < 5e-4   # ilk sıra girdi
    assert all(k not in ("ev", "evler", "kitap") for k, _ in atilmis)


def test_siralar_esitlikte_ortalama():
    assert np.allclose(dg.siralar([3.0, 7.0, 7.0, 1.0]), [3.0, 1.5, 1.5, 4.0])
    assert np.allclose(dg.siralar([3.0, 7.0, 7.0, 1.0], buyukten_kucuge=False), [2.0, 3.5, 3.5, 1.0])


def test_spearman_el_hesabi():
    insan = [9.2, 7.0, 6.5, 6.0, 3.1, 0.4]
    model = [0.88, 0.79, 0.81, 0.62, 0.46, 0.05]
    rho, d, toplam = dg.spearman_d2(insan, model)
    assert toplam == 2.0 and abs(rho - (1 - 12 / 210)) < 1e-12 and abs(rho - 0.9429) < 5e-5
    assert abs(dg.spearman(insan, model) - rho) < 1e-12          # esitlik yokken iki yol aynı
    assert abs(dg.spearman(insan, [10 * m for m in model]) - rho) < 1e-12   # ölçekten bağımsız
    assert dg.spearman(insan, [-i for i in insan]) == -1.0


def test_spearman_scipy_ile_capraz_kontrol():
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return  # scipy yoksa bu kontrol atlanır
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.integers(0, 6, size=12).astype(float)    # bol eşitlik içerir
        y = rng.normal(size=12)
        assert abs(dg.spearman(x, y) - spearmanr(x, y)[0]) < 1e-12
    assert abs(dg.spearman([3, 7, 7, 1], [2, 9, 8, 1]) - spearmanr([3, 7, 7, 1], [2, 9, 8, 1])[0]) < 1e-12


def test_cift_degerlendir_kapsam():
    ciftler = [("ev", "evler"), ("ev", "yokkelime"), ("kitap", "kitaplar"), ("masa", "kalem")]
    rho, kul, toplam, _ = dg.cift_degerlendir(ciftler, [8, 5, 9, 1], VEK.get)
    assert (kul, toplam) == (3, 4)          # vektörü olmayan çift atlanır


def test_gensim_analoji_capraz_kontrol():
    try:
        from gensim.models import KeyedVectors
    except ImportError:
        return
    kv = KeyedVectors(vector_size=2)
    kv.add_vectors(list(VEK), np.array(list(VEK.values())))
    gs = kv.most_similar(positive=["evler", "kitap"], negative=["ev"], topn=10)
    _, benim = dg.analoji_coz("ev", "evler", "kitap", VEK, girdileri_at=True, normalize=True)
    assert [k for k, _ in gs] == [k for k, _ in benim]
    for (k1, c1), (k2, c2) in zip(gs, benim):
        assert abs(c1 - c2) < 1e-6, (k1, c1, c2)


def test_gensim_ile_capraz_kontrol():
    try:
        from gensim.models.fasttext import ft_ngram_hashes
    except ImportError:
        return  # gensim yoksa bu kontrol atlanır
    for kelime in ["kitap", "evlerimizden", "bilgisyar", "bilgisayar", "ev", "çay", "göz"]:
        referans = sorted(ft_ngram_hashes(kelime, 3, 6, 2_000_000))
        benim = sorted(ft.kova_indeksi(g) for g in ft.ngramlar(kelime))
        assert referans == benim, kelime


if __name__ == "__main__":
    testler = [(ad, f) for ad, f in sorted(globals().items()) if ad.startswith("test_")]
    for ad, f in testler:
        f()
        print(f"ok  {ad}")
    print(f"\n{len(testler)} test geçti.")
