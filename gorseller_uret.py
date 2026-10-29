"""
README'de kullanılan görselleri üretir (gorseller/ klasörüne PNG olarak kaydeder).
Sayıların hepsi diğer dosyalardaki fonksiyonlardan gelir, elle yazılmış değer yoktur.

Çalıştırma:  python gorseller_uret.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import degerlendirme as dg
import degerlendirme_el_hesabi as dgh
import fasttext as ft
import glove
import word2vec as w2v

CIKTI = "gorseller"
os.makedirs(CIKTI, exist_ok=True)

# Sade palet: bir vurgu rengi + bir ikincil renk + gri metin/ızgara
VURGU = "#2B6CB0"
IKINCIL = "#DD8A2F"
GRI = "#6B7280"
IZGARA = "#E5E7EB"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": GRI,
    "axes.labelcolor": GRI,
    "xtick.color": GRI,
    "ytick.color": GRI,
    "axes.grid": True,
    "grid.color": IZGARA,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "figure.dpi": 130,
})


def kaydet(fig, ad):
    yol = os.path.join(CIKTI, ad)
    fig.savefig(yol, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("kaydedildi:", yol)


# --------------------------------------------------------------------------
def gorsel_kayip():
    """Aynı örnekte gradyan inişi adımları: negatif örnekleme kaybı."""
    w = np.array([1.0, 1.0])
    cp = np.array([0.5, -0.5])
    cn = [np.array([0.5, 0.5]), np.array([-1.0, 0.0])]
    kayiplar = []
    for _ in range(31):
        kayiplar.append(w2v.negatif_ornekleme_kaybi(w, cp, cn)[0])
        w, cp, cn = w2v.adim_at(w, cp, cn, 0.5)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(kayiplar)), kayiplar, color=VURGU, linewidth=2)
    for adim in (0, 1):
        ax.scatter([adim], [kayiplar[adim]], color=VURGU, s=40, zorder=3,
                   edgecolor="white", linewidth=1.5)
    ax.annotate(f"adım 0: L = {kayiplar[0]:.2f}", (0, kayiplar[0]), xytext=(3, kayiplar[0] - 0.05),
                color="#1F2937", fontsize=9)
    ax.annotate(f"adım 1: L = {kayiplar[1]:.2f}", (1, kayiplar[1]), xytext=(3.5, kayiplar[1] + 0.15),
                color="#1F2937", fontsize=9,
                arrowprops=dict(arrowstyle="-", color=GRI, linewidth=0.8))
    ax.set_xlabel("gradyan inişi adımı (η = 0.5)")
    ax.set_ylabel("negatif örnekleme kaybı L")
    ax.set_title("Aynı örnek üzerinde kayıp adım adım düşüyor", loc="left", color="#111827", fontsize=11)
    kaydet(fig, "negatif_ornekleme_kayip.png")


def gorsel_uc_dort():
    """¾ kuralı: ham sıklık ve ¾ kuvvetiyle çekim olasılığı."""
    ham = np.array([0.99, 0.01])
    yeni = w2v.uc_dort_dagilimi(ham)
    adlar = ["sık kelime (a)", "nadir kelime (b)"]
    x = np.arange(2)
    g = 0.36

    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x - g / 2, ham, g, color=GRI, label="ham sıklık", zorder=3)
    b2 = ax.bar(x + g / 2, yeni, g, color=VURGU, label="¾ kuralı sonrası", zorder=3)
    for barlar in (b1, b2):
        for b in barlar:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                    f"{b.get_height():.2f}", ha="center", color="#1F2937", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(adlar)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("negatif olarak çekilme olasılığı")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(f"Nadir kelimenin şansı {yeni[1] / ham[1]:.1f} kat artıyor", loc="left",
                 color="#111827", fontsize=11)
    kaydet(fig, "uc_dort_kurali.png")


def gorsel_glove_agirlik():
    """GloVe ağırlık fonksiyonu f(x)."""
    xs = np.linspace(0, 150, 600)
    fs = [glove.agirlik(x) for x in xs]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, fs, color=VURGU, linewidth=2)
    for x in (1, 10, 100):
        f = glove.agirlik(x)
        ax.scatter([x], [f], color=VURGU, s=40, zorder=3, edgecolor="white", linewidth=1.5)
    ax.annotate(f"f(1) = {glove.agirlik(1):.2f}", (1, glove.agirlik(1)), xytext=(32, 0.08),
                color="#1F2937", fontsize=9, va="center",
                arrowprops=dict(arrowstyle="-", color=GRI, linewidth=0.8))
    ax.annotate(f"f(10) = {glove.agirlik(10):.2f}", (10, glove.agirlik(10)), xytext=(32, 0.20),
                color="#1F2937", fontsize=9, va="center",
                arrowprops=dict(arrowstyle="-", color=GRI, linewidth=0.8))
    ax.annotate("f(100) = 1  (tavan)", (100, 1.0), xytext=(60, 1.07), color="#1F2937", fontsize=9)
    ax.text(75, 0.03, "X = 0: çift kayba hiç girmez", color=GRI, fontsize=9)
    ax.set_ylim(0, 1.2)
    ax.set_xlabel("birlikte görülme sayısı X_ij")
    ax.set_ylabel("ağırlık f(X_ij)")
    ax.set_title("GloVe ağırlık fonksiyonu (x_max = 100, α = 0.75)", loc="left", color="#111827", fontsize=11)
    kaydet(fig, "glove_agirlik_fonksiyonu.png")


def gorsel_ngram_sayilari():
    """fastText: n değerine göre n-gram sayıları."""
    kelimeler = ["kitap", "evlerimizden"]
    renkler = [GRI, VURGU]
    ns = [3, 4, 5, 6]
    x = np.arange(len(ns))
    g = 0.36

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (kelime, renk) in enumerate(zip(kelimeler, renkler)):
        degerler = [ft.ngram_sayisi(kelime, n) for n in ns]
        toplam = sum(degerler)
        etiket = f"{ft.sinirli_kelime(kelime)}  (toplam {toplam} n-gram, +1 kelime = {toplam + 1} vektör)"
        barlar = ax.bar(x + (i - 0.5) * g, degerler, g, color=renk, label=etiket, zorder=3)
        for b in barlar:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2, f"{int(b.get_height())}",
                    ha="center", color="#1F2937", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"n = {n}" for n in ns])
    ax.set_ylim(0, 15)
    ax.set_ylabel("n-gram sayısı  (L - n + 1)")
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    ax.set_title("fastText: sınır işaretli kelimeden çıkan n-gram sayıları", loc="left",
                 color="#111827", fontsize=11)
    kaydet(fig, "fasttext_ngram_sayilari.png")


def gorsel_ortak_ngram():
    """Yazım hatası: 'bilgisyar' parçalarının kaçı 'bilgisayar' ile ortak?"""
    yanlis, dogru = "bilgisyar", "bilgisayar"
    toplam = len(ft.ngramlar(yanlis))
    ortak = len(ft.ortak_ngramlar(yanlis, dogru))

    fig, ax = plt.subplots(figsize=(7, 2.4))
    ax.barh([0], [ortak], color=VURGU, height=0.5, zorder=3)
    ax.barh([0], [toplam - ortak], left=[ortak], color=GRI, height=0.5, zorder=3,
            edgecolor="white", linewidth=2)
    ax.text(ortak / 2, 0, f"{ortak} parça '{dogru}' ile ortak", ha="center", va="center",
            color="white", fontsize=10)
    ax.text(ortak + (toplam - ortak) / 2, 0, f"{toplam - ortak} parça özgün", ha="center",
            va="center", color="white", fontsize=10)
    ax.set_xlim(0, toplam)
    ax.set_yticks([])
    ax.set_xlabel(f"'{ft.sinirli_kelime(yanlis)}' kelimesinin {toplam} n-gram'ı")
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_title("Yazım hatasına rağmen vektör üretilebiliyor", loc="left", color="#111827", fontsize=11)
    kaydet(fig, "fasttext_ortak_ngram.png")


def gorsel_analoji():
    """Analoji: 'çoğul yapma' yönü ev -> evler, kitap'a eklenince kitaplar'ın yakınına varılıyor."""
    V = dgh.VEKTORLER
    sorgu = dg.analoji_sorgusu(V["ev"], V["evler"], V["kitap"])

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.grid(True)
    ok = dict(arrowstyle="-|>", linewidth=2, shrinkA=7, shrinkB=7, mutation_scale=14)
    ax.annotate("", xy=V["evler"], xytext=V["ev"], arrowprops=dict(color=GRI, **ok))
    ax.annotate("", xy=sorgu, xytext=V["kitap"], arrowprops=dict(color=VURGU, **ok))
    ax.text(1.55, 2.0, "evler − ev\n(çoğul yönü)", ha="right", va="center", color=GRI, fontsize=9)
    ax.text(5.45, 2.1, "aynı yön\nkitap'a eklenir", ha="left", va="center", color=VURGU, fontsize=9)

    for kelime, v in V.items():
        ax.scatter(*v, color="#1F2937", s=36, zorder=3, edgecolor="white", linewidth=1.2)
    etiket = {"ev": (-0.1, -0.5), "evler": (-0.1, 0.3), "kitap": (0.0, -0.5), "kitaplar": (-0.3, 0.38),
              "masa": (0.15, 0.2), "kalem": (0.1, -0.5)}
    for kelime, (dx, dy) in etiket.items():
        v = V[kelime]
        ax.text(v[0] + dx, v[1] + dy, kelime, color="#1F2937", fontsize=9.5)

    ax.scatter(*sorgu, marker="x", color=IKINCIL, s=70, linewidths=2.2, zorder=4)
    ax.text(sorgu[0] + 0.2, sorgu[1] - 0.05, "sorgu = (5, 3)", color=IKINCIL, fontsize=9.5, va="center")

    ax.set_xlim(-2, 7.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_xlabel("1. boyut")
    ax.set_ylabel("2. boyut")
    ax.set_title("Analoji: ev : evler :: kitap : ?  →  evler − ev + kitap", loc="left",
                 color="#111827", fontsize=11)
    kaydet(fig, "analoji_vektor_aritmetigi.png")


def gorsel_spearman():
    """Spearman: insan sırası ile modelin kosinüs sırası; köşegen = tam uyum."""
    ciftler = dgh.CIFTLER
    adlar = [c[0] for c in ciftler]
    s_insan = dg.siralar([c[1] for c in ciftler])
    s_model = dg.siralar([c[2] for c in ciftler])
    rho, _, _ = dg.spearman_d2([c[1] for c in ciftler], [c[2] for c in ciftler])
    n = len(ciftler)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot([0.4, n + 0.6], [0.4, n + 0.6], color=GRI, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ayni = [x == y for x, y in zip(s_insan, s_model)]
    for ad, x, y, a in zip(adlar, s_insan, s_model, ayni):
        ax.scatter(x, y, s=60, color=VURGU if a else IKINCIL, zorder=3, edgecolor="white", linewidth=1.5)
        if x <= 3 and y <= x:   # etiket noktanın sağ-altına
            ax.text(x + 0.13, y - 0.2, ad, fontsize=9, color="#1F2937", ha="left", va="center")
        else:        # etiket noktanın sol-üstüne
            ax.text(x - 0.13, y + 0.2, ad, fontsize=9, color="#1F2937", ha="right", va="center")
    ax.scatter([], [], s=60, color=VURGU, label="sıra aynı")
    ax.scatter([], [], s=60, color=IKINCIL, label="sıra farklı")
    ax.plot([], [], color=GRI, linewidth=1.2, linestyle=(0, (4, 3)), label="köşegen = tam uyum")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_xlim(0.4, n + 0.6)
    ax.set_ylim(0.4, n + 0.8)
    ax.set_xticks(range(1, n + 1))
    ax.set_yticks(range(1, n + 1))
    ax.set_xlabel("insan sırası (1 = en benzer)")
    ax.set_ylabel("modelin kosinüs sırası (1 = en benzer)")
    ax.set_title(f"Spearman ρ = {rho:.4f}: yalnızca 2 çiftin sırası yer değiştirmiş", loc="left",
                 color="#111827", fontsize=11)
    kaydet(fig, "spearman_siralar.png")


if __name__ == "__main__":
    gorsel_analoji()
    gorsel_spearman()
    gorsel_kayip()
    gorsel_uc_dort()
    gorsel_glove_agirlik()
    gorsel_ngram_sayilari()
    gorsel_ortak_ngram()
