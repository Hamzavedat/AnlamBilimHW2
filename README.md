# HW2 - LLM Trajectory Analysis

Bu proje, iki dil modelinin metin uretimi sirasinda olusan trajectory'lerini karsilastirmak icin hazirlandi.

## Ne yapiyor?

- `src/collect_data.py`
  - `gpt2` ve `EleutherAI/pythia-70m` modellerinden veri toplar.
  - ELI5 category dataset'inden secilen `10 kategori x 100 soru = 1000 prompt` kullanir.
  - Her model icin `1000` trajectory uretir.
  - Her trajectory en az `50` token uzunlugundadir.
  - Sorulari her kategoride `q_id`, sonra `title` sirasina gore dizer ve ilk `100` kaydi alir.
  - Token secimini greedy `argmax` ile yaptigi icin uretim deterministiktir.
  - Orijinal soru metni ve answer alanlarini metadata icinde saklar.
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
- Dataset dosyalari su klasor yapisinda beklenir:
  - `eli5_category/train.jsonl` veya `eli5_category/train.json.gz`
  - `eli5_category/valid.jsonl` veya `eli5_category/valid.json.gz`
  - `eli5_category/valid2.jsonl` veya `eli5_category/valid2.json.gz`
  - `eli5_category/test.jsonl` veya `eli5_category/test.json.gz`
- Kullanilan kategoriler:
  - Biology, Chemistry, Culture, Earth Science, Economics, Mathematics, Other, Physics, Psychology, Technology
- `Engineering` ve `Repost` dahil edilmedi; cunku bunlar sadece `test` veya `valid2` tarafinda gorunuyor.
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
