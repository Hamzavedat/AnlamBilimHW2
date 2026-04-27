# HW2 - LLM Trajectory Analysis

Bu proje, iki dil modelinin metin uretimi sirasinda olusan trajectory'lerini karsilastirmak icin hazirlandi.

## Ne yapiyor?

- `src/collect_data.py`
  - `gpt2` ve `EleutherAI/pythia-70m` modellerinden veri toplar.
  - Elle tanimlanmis `10 kategori x 10 topic x 10 soru sablonu = 1000 benzersiz prompt` kullanir.
  - Her model icin `1000` trajectory uretir.
  - Her trajectory en az `50` token uzunlugundadir.
  - Prompt bank'i deterministik olarak uretir ve `tables/prompt_bank.csv` icine yazar.
  - Token secimini greedy `argmax` ile yaptigi icin uretim deterministiktir.
  - Prompt metni dogrudan modele verilir; ek `Question:` veya `Answer clearly...` sarmalayicisi kullanilmaz.
  - Metadata icinde hem yalnizca modelin urettigi `generated_text` hem de tam `full_text` saklanir.
- `src/extract_features.py`
  - trajectory dosyalarindan ozellik cikarir.
- `src/predict.py`
  - iki tahmin gorevi kurar:
    - model identity
    - prompt category
- `src/visualize.py`
  - sonuclar icin grafik uretir.

## Calistirma

Gerekli paketleri kurmak icin:

```bash
python -m pip install -r requirements.txt
```

Tum pipeline'i bastan sona calistirmak icin:

```bash
python src/run_all.py
```

Not:
- Kullanilan kategoriler:
  - History, Physics, Biology, Technology, Economics, Psychology, Mathematics, Culture, Art, Sports
- Her kategoride 10 topic ve tum kategorilerde ortak 10 soru sablonu vardir.
- Grouped split tarafinda `prompt_id = topic_id` kullanilir; yani ayni topic'in farkli soru varyasyonlari train ve testte ayni anda bulunmaz.
- Ilk calistirmada modeller Hugging Face uzerinden indirilebilir.
- Modeller cache'te varsa offline calistirmak icin `HF_HUB_OFFLINE=1` ayarlanabilir.
- Bu ortamda build CPU uzerinden calistigi icin uzun surebilir.

## Neden bu iki model?

- `gpt2`
  - klasik ve iyi bilinen bir causal language model
  - hidden size daha buyuk, cevaplari daha yayvan ve degisken olabiliyor
- `EleutherAI/pythia-70m`
  - daha kucuk bir causal language model
  - farkli aileden geldigi icin GPT-2 ile anlamli bir karsilastirma veriyor

Bu secimle hem iki model de yerelde calisabilir kaliyor hem de trajectory'lerde boyut, mimari ve uretim davranisi farki gorulebiliyor.

## Beklenen ciktilar

Build tamamlandiginda su klasorde sonuc olusur:

```text
runs/hw2_full_seed42/
```

Bu klasor icinde:
- `tables/prompt_bank.csv`
- `tables/features.csv`
- `tables/*_metadata.csv`
- `trajectories/*.npy`
- `results/metrics.json`
- `results/report.md`
- `plots/`

## Sunum

Sunum dosyalari:
- `hw2_trajectory_presentation.tex`
- `hw2_trajectory_presentation.pdf`
