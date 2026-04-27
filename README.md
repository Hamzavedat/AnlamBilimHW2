# HW2 - LLM Trajectory Analysis

Bu proje, iki dil modelinin metin uretimi sirasinda olusan trajectory'lerini karsilastirmak icin hazirlandi.

## Ne yapiyor?

- `src/collect_data.py`
  - `gpt2` ve `EleutherAI/pythia-70m` modellerinden veri toplar.
  - Her model icin `1000` trajectory uretir.
  - Her trajectory en az `50` token uzunlugundadir.
- `src/extract_features.py`
  - trajectory dosyalarindan ozellik cikarir.
- `src/predict.py`
  - iki tahmin gorevi kurar:
    - model identity
    - prompt category
- `src/visualize.py`
  - sonuclar icin grafik uretir.

## Calistirma

Tum pipeline'i bastan sona calistirmak icin:

```bash
bash run_all.sh
```

Not:
- Script offline calisacak sekilde ayarlanmistir.
- Bu ortamda build CPU uzerinden calistigi icin uzun surebilir.

## Beklenen ciktilar

Build tamamlandiginda su klasorde sonuc olusur:

```text
runs/hw2_full_seed42/
```

Bu klasor icinde:
- `data/features.csv`
- `results/metrics.json`
- `results/report.md`
- `plots/`

## Sunum

Sunum dosyalari:
- `hw2_trajectory_presentation.tex`
- `hw2_trajectory_presentation.pdf`
