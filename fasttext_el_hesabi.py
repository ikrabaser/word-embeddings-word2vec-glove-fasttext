"""
fastText - ELLE TAKİP EDİLEBİLEN SAYISAL ÖRNEK

Çalıştırma:  python fasttext_el_hesabi.py
"""
import numpy as np

import fasttext as ft
from ortak import kosinus_benzerligi

np.set_printoptions(precision=4, suppress=True)


def baslik(metin):
    print("\n" + "=" * 70)
    print(metin)
    print("=" * 70)


def bolum_ngram_sayisi():
    baslik("1) n-gram sayısı:  L - n + 1  ('<' ve '>' dahil)")
    for kelime in ["kitap", "evlerimizden"]:
        w = ft.sinirli_kelime(kelime)
        print(f"\n  {w}  ->  {len(w)} karakter")
        toplam = 0
        for n in range(3, 7):
            say = ft.ngram_sayisi(kelime, n)
            toplam += say
            print(f"    n = {n}: {len(w)} - {n} + 1 = {say}")
        print(f"    Toplam n-gram = {toplam}  (+ kelimenin kendisi = {toplam + 1} vektör)")
        assert toplam == len(ft.ngramlar(kelime))
        assert toplam + 1 == ft.vektor_sayisi(kelime)

    print("\n  '<evlerimizden>' için sadece n = 3 alınırsa:")
    print("   ", ", ".join(ft.ngramlar("evlerimizden", 3, 3)))


def bolum_kucuk_ornek():
    baslik("2) Kelime vektörü = parçaların toplamı (boyut 2, yalnızca n = 3)")
    kelime = "ev"
    parcalar = ft.ngramlar(kelime, 3, 3)
    print(f"  {ft.sinirli_kelime(kelime)} 4 karakter -> 3-gramlar: {parcalar}  "
          f"(4 - 3 + 1 = {len(parcalar)} tane)")
    print(f"  Kelimenin kendisi de eklenince {len(parcalar) + 1} vektör toplanır.")

    # Örnek (uydurma) vektörler
    ngram_tablosu = {"<ev": [1.0, 0.0], "ev>": [0.0, 1.0]}
    kelime_tablosu = {"ev": [1.0, 1.0]}
    print("\n  Parça        Vektör")
    print(f"  <ev          {ngram_tablosu['<ev']}")
    print(f"  ev>          {ngram_tablosu['ev>']}")
    print(f"  <ev> (kelime){kelime_tablosu['ev']}")
    v = ft.kelime_vektoru(kelime, ngram_tablosu, kelime_tablosu, 3, 3)
    print(f"\n  Kelime vektörü = (1; 0) + (0; 1) + (1; 1) = {v}")

    print("\n  Paylaşım: '<evim>' için 3-gramlar:", ft.ngramlar("evim", 3, 3))
    ortak = ft.ortak_ngramlar("ev", "evim", 3, 3)
    print(f"  'ev' ile ortak parça: {sorted(ortak)}  -> 'evim', 'ev'den öğrenilmiş bir parçayı kullanır.")


def bolum_hash():
    baslik("3) Hash kovaları (2 milyon kova)")
    for g in ["<ev", "ev>", "evl", "<kitap>"]:
        print(f"  kova('{g}') = {ft.kova_indeksi(g)}")
    print("  n-gram sayısı çok fazla olduğu için her birine ayrı satır tutulmaz;")
    print("  hash ile kovaya atılır, farklı n-gram'ların aynı kovaya düşmesi (çakışma) kabul edilir.")


def bolum_oov():
    baslik("4) Yazım hatası / sözlükte olmayan kelime (OOV)")
    dogru, yanlis, ilgisiz = "bilgisayar", "bilgisyar", "masa"
    g_yanlis = ft.ngramlar(yanlis)
    g_dogru = ft.ngramlar(dogru)
    ortak = ft.ortak_ngramlar(yanlis, dogru)
    print(f"  '{yanlis}'  : {len(g_yanlis)} n-gram")
    print(f"  '{dogru}' : {len(g_dogru)} n-gram")
    print(f"  Ortak n-gram sayısı: {len(ortak)}  ({len(ortak)}/{len(g_yanlis)} parça '{dogru}' ile ortak)")
    print("  word2vec'te sözlükte olmayan kelime -> KeyError; fastText parçalardan vektör üretir.")

    # Gösterim için rastgele (ama sabit tohumlu) n-gram vektörleri: eğitimde öğrenilmiş sayılır.
    rng = np.random.default_rng(0)
    tablo = {}
    for kelime in (dogru, ilgisiz):          # eğitim kümesindeki kelimeler
        for g in ft.ngramlar(kelime):
            tablo.setdefault(g, rng.normal(size=8))

    v_dogru = ft.kelime_vektoru(dogru, tablo)
    v_yanlis = ft.kelime_vektoru(yanlis, tablo)   # 'bilgisyar' eğitimde hiç görülmedi
    v_ilgisiz = ft.kelime_vektoru(ilgisiz, tablo)
    print("\n  (Gösterim: n-gram vektörleri rastgele, sabit tohumlu 8 boyutlu vektörlerdir.)")
    print(f"  cos({yanlis}, {dogru})  = {kosinus_benzerligi(v_yanlis, v_dogru):.3f}   <- ortak parçalar sayesinde yüksek")
    print(f"  cos({yanlis}, {ilgisiz}) = {kosinus_benzerligi(v_yanlis, v_ilgisiz):.3f}   <- ortak parça az")


if __name__ == "__main__":
    bolum_ngram_sayisi()
    bolum_kucuk_ornek()
    bolum_hash()
    bolum_oov()
