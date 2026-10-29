"""
word2vec (skip-gram + negatif örnekleme) - ELLE TAKİP EDİLEBİLEN SAYISAL ÖRNEK

Bu dosya, hesabın her adımını ekrana yazar. Sayılar elle de hesaplanabilecek kadar
küçüktür (vektör boyutu 2).

Çalıştırma:  python word2vec_el_hesabi.py
"""
import numpy as np

import word2vec as w2v
from ortak import kosinus_benzerligi, one_hot, sigmoid

np.set_printoptions(precision=4, suppress=True)

# --- Örnek veriler (kodun tamamında aynı) -------------------------------------
CUMLE = ["sabah", "sıcak", "çay", "içtim"]

W = np.array([1.0, 1.0])            # merkez kelime: "çay"
C_POZ = np.array([0.5, -0.5])       # gerçek bağlam: "içtim"
C_NEGLER = [np.array([0.5, 0.5]),   # uydurma bağlam 1: "masa"
            np.array([-1.0, 0.0])]  # uydurma bağlam 2: "koştu"
NEG_ADLARI = ["masa", "koştu"]
OGRENME_ORANI = 0.5


def baslik(metin):
    print("\n" + "=" * 70)
    print(metin)
    print("=" * 70)


def bolum_one_hot():
    baslik("1) One-hot vektörler neden benzerlik bilgisi taşımaz?")
    sozluk = ["kedi", "köpek", "masa", "çay", "kahve"]
    vektorler = {k: one_hot(i, len(sozluk)) for i, k in enumerate(sozluk)}
    for k, v in vektorler.items():
        print(f"  {k:6s} -> {v.astype(int)}")
    print("\n  İç çarpımlar (1'ler hiçbir zaman aynı konuma gelmediği için hep 0):")
    print(f"  kedi · köpek = {np.dot(vektorler['kedi'], vektorler['köpek']):.0f}")
    print(f"  kedi · masa  = {np.dot(vektorler['kedi'], vektorler['masa']):.0f}")
    print("  => 'kedi' ile 'köpek' arasındaki mesafe, 'kedi' ile 'masa' arasındakiyle aynı.")


def bolum_ciftler():
    baslik("2) Skip-gram çiftleri (pencere = 1)")
    ciftler = w2v.skipgram_ciftleri(CUMLE, pencere=1)
    print(f"  Cümle: {' '.join(CUMLE)}")
    for merkez, baglam in ciftler:
        print(f"  ({merkez}, {baglam})")
    n = len(CUMLE)
    print(f"\n  Toplam {len(ciftler)} çift. Kısa yol: 2(n-1) = 2·({n}-1) = {2 * (n - 1)}")
    print("  (Uçtaki kelimelerin yalnızca tek komşusu vardır; 8 bulmak sık yapılan hata.)")


def bolum_maliyet():
    baslik("3) Maliyet: tam softmax - negatif örnekleme")
    V, d, k = 300_000, 300, 5
    tam = w2v.tam_softmax_maliyeti(V, d)
    neg = w2v.negatif_ornekleme_maliyeti(k, d)
    print(f"  Tam softmax        : |V| x d     = {V:,} x {d} = {tam:,} çarpma")
    print(f"  Negatif örnekleme  : (k+1) x d   = {k + 1} x {d} = {neg:,} çarpma")
    print(f"  Oran               : {tam:,} / {neg:,} = {tam / neg:,.0f} kat")
    print("  (Bu, çarpma sayısı oranıdır. Ölçülen süre oranı donanıma ve uygulamaya bağlıdır.)")


def bolum_uc_dort():
    baslik("4) Negatifleri çekerken ¾ kuralı")
    ham = np.array([0.99, 0.01])
    yeni = w2v.uc_dort_dagilimi(ham)
    print("  Ham olasılık : a = 0.99, b = 0.01")
    print(f"  a^0.75 = {0.99 ** 0.75:.4f}    b^0.75 = {0.01 ** 0.75:.4f}    "
          f"toplam = {0.99 ** 0.75 + 0.01 ** 0.75:.4f}")
    print(f"  Normalize   : a = {yeni[0]:.4f} (~0.97)   b = {yeni[1]:.4f} (~0.03)")
    print(f"  Nadir kelimenin şansı {yeni[1] / ham[1]:.2f} kat arttı; sık kelimeninki biraz azaldı.")

    baslik("4b) Alt örnekleme (sık kelimeleri atlama)")
    for ad, f in [("bir/ve/bu gibi çok sık kelime", 0.01), ("orta sıklıkta kelime", 1e-4),
                  ("nadir kelime", 1e-6)]:
        p = w2v.alt_ornekleme_atma_olasiligi(f)
        print(f"  f = {f:<8g} ({ad:30s}) -> atılma olasılığı = {p:.3f}")
    print("  ¾ kuralı negatif ÇEKERKEN, alt örnekleme metinden kelime ATLARKEN uygulanır.")


def bolum_kayip():
    baslik("5) Negatif örnekleme kaybı (tek adım, elle hesap)")
    print(f"  Merkez  w = 'çay'    = {W}")
    print(f"  Pozitif c = 'içtim'  = {C_POZ}")
    for ad, c in zip(NEG_ADLARI, C_NEGLER):
        print(f"  Negatif c = '{ad}'{' ' * (6 - len(ad))}= {c}")

    L, ayrinti = w2v.negatif_ornekleme_kaybi(W, C_POZ, C_NEGLER)
    adlar = ["içtim (pozitif)"] + [f"{a} (negatif)" for a in NEG_ADLARI]
    print("\n  Terim            c·w      σ(±c·w)    ceza = -log σ")
    for ad, (_, s, sig, ceza) in zip(adlar, ayrinti):
        print(f"  {ad:15s} {s:6.2f}   {sig:8.4f}   {ceza:8.4f}")
    print(f"\n  Toplam kayıp L = {L:.4f}")
    print("  Not: negatiflerde σ(-c·w) alınır. 'masa' için c·w = 1 olduğundan ağır ceza yer;")
    print("       model uydurma çifti 'gerçek' sanıyor.")
    return L


def bolum_guncelleme(L0):
    baslik("6) Gradyanlar ve tek güncelleme adımı (η = 0.5)")
    g_w, g_c_poz, g_c_negler = w2v.negatif_ornekleme_gradyanlari(W, C_POZ, C_NEGLER)
    print(f"  dL/dc_içtim = (σ-1)·w = {g_c_poz}")
    for ad, g in zip(NEG_ADLARI, g_c_negler):
        print(f"  dL/dc_{ad:6s}= σ·w     = {g}")
    print(f"  dL/dw       = pozitif + negatiflerin toplamı = {g_w}")

    yw, yc, yn = w2v.adim_at(W, C_POZ, C_NEGLER, OGRENME_ORANI)
    print("\n  Güncelleme: θ <- θ - η·gradyan")
    print(f"  w      : {W} -> {yw}")
    print(f"  içtim  : {C_POZ} -> {yc}")
    for ad, eski, yeni in zip(NEG_ADLARI, C_NEGLER, yn):
        print(f"  {ad:6s} : {eski} -> {yeni}")

    print("\n  Skorlar (iç çarpım) önce -> sonra:")
    print(f"  çay · içtim : {np.dot(C_POZ, W):.2f} -> {np.dot(yc, yw):.2f}   (gerçek çift yaklaştı)")
    for ad, eski, yeni in zip(NEG_ADLARI, C_NEGLER, yn):
        print(f"  çay · {ad:6s}: {np.dot(eski, W):.2f} -> {np.dot(yeni, yw):.2f}   (uydurma çift uzaklaştı)")

    L1, _ = w2v.negatif_ornekleme_kaybi(yw, yc, yn)
    print(f"\n  Kayıp: {L0:.2f} -> {L1:.2f}")


def bolum_cok_adim():
    baslik("7) Aynı adımı tekrarlarsak kayıp nasıl düşer?")
    w, cp, cn = W.copy(), C_POZ.copy(), [c.copy() for c in C_NEGLER]
    for adim in range(0, 31):
        if adim in (0, 1, 2, 5, 10, 20, 30):
            L, _ = w2v.negatif_ornekleme_kaybi(w, cp, cn)
            print(f"  adım {adim:2d}: L = {L:.4f}   çay·içtim = {np.dot(cp, w):6.3f}   "
                  f"cos(çay, içtim) = {kosinus_benzerligi(w, cp):.3f}")
        w, cp, cn = w2v.adim_at(w, cp, cn, OGRENME_ORANI)


if __name__ == "__main__":
    bolum_one_hot()
    bolum_ciftler()
    bolum_maliyet()
    bolum_uc_dort()
    L0 = bolum_kayip()
    bolum_guncelleme(L0)
    bolum_cok_adim()
