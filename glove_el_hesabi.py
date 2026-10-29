"""
GloVe - ELLE TAKİP EDİLEBİLEN SAYISAL ÖRNEK

Çalıştırma:  python glove_el_hesabi.py
"""
import numpy as np

import glove

np.set_printoptions(precision=4, suppress=True)


def baslik(metin):
    print("\n" + "=" * 70)
    print(metin)
    print("=" * 70)


def bolum_sayim():
    baslik("1) Birlikte-görülme sayımları X(i, j) ve olasılıklar P(j | i)")
    cumleler = ["sabah sıcak çay içtim",
                "akşam sıcak kahve içtim",
                "sıcak çay içtim"]
    print("  Derlem:")
    for c in cumleler:
        print(f"    - {c}")
    sozluk, X = glove.birlikte_gorulme_matrisi(cumleler, pencere=1)
    P = glove.olasilik_matrisi(X)

    print(f"\n  Sözlük: {sozluk}")
    print("\n  X(i, j): j kelimesi i'nin penceresinde kaç kez geçti (pencere = 1)")
    print("  " + " " * 8 + "".join(f"{k:>8s}" for k in sozluk) + "   | X(i)")
    for i, k in enumerate(sozluk):
        print(f"  {k:8s}" + "".join(f"{X[i, j]:8.0f}" for j in range(len(sozluk)))
              + f"   | {X[i].sum():.0f}")

    i = sozluk.index("çay")
    print("\n  P(j | çay) = X(çay, j) / X(çay):")
    for j, k in enumerate(sozluk):
        if X[i, j] > 0:
            print(f"    P({k:6s} | çay) = {X[i, j]:.0f} / {X[i].sum():.0f} = {P[i, j]:.3f}")

    kahve = sozluk.index("kahve")
    print("\n  Oran örneği: P(k | çay) / P(k | kahve)")
    for k_ad in ["sıcak", "içtim", "sabah", "akşam"]:
        k = sozluk.index(k_ad)
        if P[kahve, k] > 0:
            oran = glove.olasilik_orani(P, i, kahve, k)
            print(f"    k = {k_ad:6s}: {P[i, k]:.3f} / {P[kahve, k]:.3f} = {oran:.2f}")
        else:
            print(f"    k = {k_ad:6s}: P(k | kahve) = 0, oran tanımsız")
    print("  (Derlem çok küçük olduğu için oranlar ya 1 ya da tanımsız çıkıyor; ")
    print("   'sıcak' ve 'içtim' iki kelimeyle de ilgili olduğundan oran 1'dir.)")


def bolum_oran_makale():
    baslik("2) Neden ham olasılığa değil ORANA bakıyoruz?")
    print("  GloVe makalesindeki 'ice' (buz) ve 'steam' (buhar) örneği, P(k|ice) / P(k|steam):")
    print("  (6 milyar sözcüklük İngilizce derlem; değerler makaleden alınmıştır)\n")
    tablo = [("solid (katı)", 8.9, "buza özgü"),
             ("gas (gaz)", 0.085, "buhara özgü"),
             ("water (su)", 1.36, "ikisiyle de ilgili -> ~1"),
             ("fashion (moda)", 0.96, "ikisiyle de ilgisiz -> ~1")]
    for ad, oran, yorum in tablo:
        print(f"    {ad:16s} {oran:7.3f}   {yorum}")
    print("\n  'water' hem buzla hem buharla sık geçer; ham olasılığa bakınca ayırt edici")
    print("  görünmez. Oran ortak etkiyi sadeleştirir, geriye ayırt edici kelimeler kalır.")


def bolum_cift_kaybi():
    baslik("3) Tek bir çiftin kaybı (elle hesap)")
    x_ij = 100
    tahmin = 4.2
    hedef, fark, kare, f, kayip = glove.cift_kaybi(tahmin, x_ij)
    print(f"  X_ij = {x_ij}  (çift 100 kez birlikte geçmiş)")
    print(f"  Modelin tahmini w_i·w~_j + b_i + b~_j = {tahmin}")
    print(f"  Hedef            : ln {x_ij} = {hedef:.4f}")
    print(f"  Fark             : {tahmin} - {hedef:.4f} = {fark:.4f}")
    print(f"  Fark²            : {kare:.4f}")
    print(f"  Ağırlık f({x_ij})   : {f:.0f}")
    print(f"  Çiftin kaybı     : {f:.0f} x {kare:.4f} = {kayip:.4f}")

    baslik("3b) Vektörlerden skora: w_i · w~_j + b_i + b~_j")
    w_i = np.array([1.0, 0.5])
    w_j = np.array([0.8, 1.2])
    b_i, b_j = 0.3, -0.1
    skor = glove.tahmin_skoru(w_i, w_j, b_i, b_j)
    print(f"  w_i = {w_i},  w~_j = {w_j},  b_i = {b_i},  b~_j = {b_j}")
    print(f"  w_i·w~_j = 1.0·0.8 + 0.5·1.2 = {np.dot(w_i, w_j):.2f}")
    print(f"  skor     = {np.dot(w_i, w_j):.2f} + {b_i} + ({b_j}) = {skor:.2f}")
    print("  (Örnek değerlerdir; gerçek eğitimde bunlar gradyan inişiyle öğrenilir.)")


def bolum_agirlik():
    baslik("4) Ağırlık fonksiyonu f(X_ij),  x_max = 100, α = 0.75")
    for x in [0, 1, 10, 50, 100, 1000]:
        print(f"  f({x:>4d}) = {glove.agirlik(x):.4f}")
    print("\n  - X = 0 : ağırlık 0 (log 0 tanımsız; çift toplamdan atlanır)")
    print("  - Nadir çiftler az, sık çiftler tam ağırlıkla katkı verir")
    print("  - x_max'tan sonra ağırlık 1'de sabitlenir (çok sık çiftler eğitimi domine etmesin)")


def bolum_toplam_kayip():
    baslik("5) Birkaç çiftten oluşan toplam kayıp J (X = 0 olan çift atlanır)")
    cift_adlari = ["(çay, sıcak)", "(çay, içtim)", "(çay, akşam)", "(çay, kahve)"]
    x_degerleri = [100, 10, 1, 0]
    tahminler = [4.2, 2.0, 0.5, 1.0]   # örnek model skorları
    print("  Çift            X_ij   tahmin   log X_ij    fark²      f      f·fark²")
    for ad, x, t in zip(cift_adlari, x_degerleri, tahminler):
        sonuc = glove.cift_kaybi(t, x)
        if sonuc is None:
            print(f"  {ad:14s} {x:5d}   {t:6.2f}   (log 0 tanımsız)  ->  ATLANDI")
        else:
            hedef, _, kare, f, kayip = sonuc
            print(f"  {ad:14s} {x:5d}   {t:6.2f}   {hedef:8.4f}   {kare:7.4f}   {f:6.4f}   {kayip:8.4f}")
    print(f"\n  Toplam kayıp J = {glove.toplam_kayip(tahminler, x_degerleri):.4f}")


if __name__ == "__main__":
    bolum_sayim()
    bolum_oran_makale()
    bolum_cift_kaybi()
    bolum_agirlik()
    bolum_toplam_kayip()
