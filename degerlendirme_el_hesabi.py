"""
Değerlendirme (analoji + Spearman) - ELLE TAKİP EDİLEBİLEN SAYISAL ÖRNEK

Vektörler ve insan puanları ÖRNEK (uydurma) değerlerdir; amaç hesabın nasıl yapıldığını
göstermek. Gerçek değerlendirmede vektörler eğitilmiş bir modelden, puanlar insanlardan gelir.

Çalıştırma:  python degerlendirme_el_hesabi.py
"""
import numpy as np

import degerlendirme as dg
import fasttext as ft
from ortak import kosinus_benzerligi

np.set_printoptions(precision=4, suppress=True)

# Analoji örneği (boyut 2). Son dört kelime "dağıtıcı": cevabı karıştırabilecek kelimeler.
VEKTORLER = {
    "ev": np.array([2.0, 1.0]),
    "evler": np.array([2.0, 3.0]),
    "kitap": np.array([5.0, 1.0]),
    "kitaplar": np.array([4.5, 3.5]),   # gerçekte tam (5, 3) olmaz; model gürültülüdür
    "masa": np.array([-1.0, 2.0]),
    "kalem": np.array([6.0, 0.5]),
}

# Spearman örneği: (çift, insan puanı 0-10, modelin kosinüsü)
CIFTLER = [
    ("araba - otomobil", 9.2, 0.88),
    ("kedi - köpek", 7.0, 0.79),
    ("kahve - çay", 6.5, 0.81),
    ("sıcak - ılık", 6.0, 0.62),
    ("kahve - fincan", 3.1, 0.46),
    ("masa - koştu", 0.4, 0.05),
]


def baslik(metin):
    print("\n" + "=" * 70)
    print(metin)
    print("=" * 70)


def bolum_analoji():
    baslik("1) Analoji:  ev : evler :: kitap : ?")
    for k, v in VEKTORLER.items():
        print(f"  {k:9s} = {v}")
    a, b, c = VEKTORLER["ev"], VEKTORLER["evler"], VEKTORLER["kitap"]
    print(f"\n  evler - ev          = {b} - {a} = {b - a}     <- 'çoğul yapma' yönü")
    sorgu = dg.analoji_sorgusu(a, b, c)
    print(f"  evler - ev + kitap  = {b - a} + {c} = {sorgu}")

    print("\n  Her kelimenin sorguya kosinüs benzerliği:")
    print(f"    cos(x, q) = x·q / (|x| |q|),   |q| = sqrt({sorgu[0]:.0f}² + {sorgu[1]:.0f}²) = "
          f"{np.linalg.norm(sorgu):.4f}")
    tum = dg.kosinus_siralamasi(sorgu, VEKTORLER)
    for k, kos in tum:
        v = VEKTORLER[k]
        print(f"    {k:9s} x·q = {np.dot(v, sorgu):6.2f}   |x| = {np.linalg.norm(v):.4f}   cos = {kos:.4f}")

    print("\n  a) Girdi kelimeleri (ev, evler, kitap) HARİÇ tutulursa:")
    _, sonuc = dg.analoji_coz("ev", "evler", "kitap", VEKTORLER, girdileri_at=True)
    for sira, (k, kos) in enumerate(sonuc, 1):
        print(f"     {sira}. {k:9s} {kos:.4f}" + ("   <- cevap" if sira == 1 else ""))

    print("\n  b) Girdi kelimeleri atılmazsa:")
    _, sonuc2 = dg.analoji_coz("ev", "evler", "kitap", VEKTORLER, girdileri_at=False)
    for sira, (k, kos) in enumerate(sonuc2, 1):
        print(f"     {sira}. {k:9s} {kos:.4f}" + ("   <- ilk sıra bir GİRDİ kelimesi!" if sira == 1 else ""))
    print("  Sorgu girdilerden birine hâlâ çok yakın olduğu için atmazsak ilk sıra hep girdi olur.")

    print("\n  c) Normalize (gensim biçimi): her vektör önce birim uzunluğa getirilir:")
    _, sonuc3 = dg.analoji_coz("ev", "evler", "kitap", VEKTORLER, girdileri_at=True, normalize=True)
    for sira, (k, kos) in enumerate(sonuc3, 1):
        print(f"     {sira}. {k:9s} {kos:.4f}")


def bolum_spearman():
    baslik("2) Spearman ρ: insan sıralaması ile kosinüs sıralaması")
    adlar = [c[0] for c in CIFTLER]
    insan = [c[1] for c in CIFTLER]
    model = [c[2] for c in CIFTLER]
    s_insan = dg.siralar(insan)
    s_model = dg.siralar(model)
    rho, d, toplam = dg.spearman_d2(insan, model)

    print("  Çift               insan  kosinüs   sıra(insan)  sıra(model)    d    d²")
    for ad, i, m, si, sm, dd in zip(adlar, insan, model, s_insan, s_model, d):
        print(f"  {ad:17s} {i:6.1f}   {m:6.2f}   {si:9.0f}   {sm:11.0f}   {dd:+3.0f}   {dd ** 2:3.0f}")
    n = len(CIFTLER)
    print(f"\n  Σd² = {toplam:.0f},  n = {n}")
    print(f"  ρ = 1 - 6·Σd² / (n(n²-1)) = 1 - 6·{toplam:.0f} / ({n}·{n ** 2 - 1}) "
          f"= 1 - {6 * toplam:.0f}/{n * (n ** 2 - 1)} = {rho:.4f}")
    print(f"  Aynısı sıralar üzerinde Pearson ile: {dg.spearman(insan, model):.4f}")

    print("\n  Spearman yalnızca SIRAYA bakar, ölçeğe bakmaz:")
    print(f"    kosinüsler 10 ile çarpılınca ρ = {dg.spearman(insan, [10 * m for m in model]):.4f}")
    print(f"    kosinüslerin karesi alınınca  ρ = {dg.spearman(insan, [m ** 2 for m in model]):.4f}"
          "   (sırayı bozmayan her dönüşüm aynı sonucu verir)")

    print("\n  Uç durumlar:")
    print(f"    model sırayı tamamen ters çevirirse (kosinüs = -insan puanı): ρ = "
          f"{dg.spearman(insan, [-i for i in insan]):+.4f}")
    print(f"    model puanları insanla aynı sırada ise                      : ρ = "
          f"{dg.spearman(insan, [2 * i + 1 for i in insan]):+.4f}")


def bolum_esitlik():
    baslik("3) Eşit değerler: ortalama sıra")
    x = [3.0, 7.0, 7.0, 1.0]
    print(f"  Değerler: {x}")
    print(f"  Sıralar (1 = en büyük): {dg.siralar(x)}")
    print("  7 ve 7 eşit: 1. ve 2. sıraya denk geliyorlar, ikisi de (1 + 2) / 2 = 1.5 alır.")
    y = [2.0, 9.0, 8.0, 1.0]
    rho_kisa, _, _ = dg.spearman_d2(x, y)
    print(f"\n  İnsan = {x},  model = {y}")
    print(f"  Kısa formül (eşitlikte yaklaşık): {rho_kisa:.4f}")
    print(f"  Sıralar üzerinde Pearson (doğru)  : {dg.spearman(x, y):.4f}")


def bolum_kapsam():
    baslik("4) Kapsam: sözlükte olmayan kelime içeren çiftler")
    ciftler = [("kitap", "kalem"), ("ev", "evler"), ("kitap", "kitaplar"),
               ("bilgisayar", "dizüstü"), ("masa", "bilgisyar")]
    puanlar = [6.0, 8.5, 9.0, 7.0, 5.0]
    # word2vec benzeri: sözlükte olmayan kelime için vektör yok (None)
    sozluk = {k: v for k, v in VEKTORLER.items()}
    sozluk.update({"bilgisayar": np.array([3.0, 4.0]), "dizüstü": np.array([2.5, 3.5])})

    def w2v(kelime):
        return sozluk.get(kelime)

    # fastText benzeri: n-gram'lardan vektör üretir
    rng = np.random.default_rng(1)
    tablo = {}
    for kelime in list(sozluk):
        for g in ft.ngramlar(kelime):
            tablo.setdefault(g, rng.normal(size=4))

    def fasttext(kelime):
        return ft.kelime_vektoru(kelime, tablo)

    for ad, f in [("word2vec benzeri (sözlük araması)", w2v), ("fastText benzeri (n-gram toplamı)", fasttext)]:
        rho, kul, toplam, _ = dg.cift_degerlendir(ciftler, puanlar, f)
        print(f"  {ad:36s}: {kul}/{toplam} çift değerlendirilebildi")
    print("\n  'bilgisyar' (yazım hatası) sözlükte yok: word2vec bu çifti atlar, fastText n-gram'larından")
    print("  vektör üretip çifti değerlendirebilir. Bu yüzden fastText daha fazla çifti kapsar.")
    print("  (Burada vektörler rastgele; yalnızca KAPSAM farkı gösterilmek isteniyor, ρ değerleri anlamsızdır.)")


if __name__ == "__main__":
    bolum_analoji()
    bolum_spearman()
    bolum_esitlik()
    bolum_kapsam()
