# CHANGES

Bu dokuman, pipeline icin yapilan kod ve metodoloji duzeltmelerini acceptance kriterlerine gore ozetler.

## 1) Reproducibility ve dataset boyutu

- `src/collect_data.py` yeniden tasarlandi.
- Yeni CLI parametreleri eklendi:
	- `--num-samples`
	- `--batch-size`
	- `--max-new-tokens`
	- `--output-dir`
	- `--run-name`
	- `--smoke` / `--full`
	- `--seed`
	- `--overwrite`
- Varsayilan davranis artik tek bir `data/` klasorunu sessizce overwrite etmez.
- Her kosu ayri bir run klasorune yazilir: `runs/<run_name>/...`
- `--overwrite` verildiginde ayni `run_name` altindaki onceki klasor agaci tamamen silinir ve temiz bir run klasoru yeniden olusturulur.
- Boylece onceki kosudan kalan `.npy`, `results/*`, `plots/*` dosyalari yeni kosuyla karismaz.
- `run_config.json` ile kosu parametreleri kaydedilir.

## 2) Seed kullanimi ve tekrar-uretilebilirlik

- Metadata'da seed sadece yazilmiyor, generation sirasinda fiilen kullaniliyor.
- Her sample icin `seed_used = base_seed + sample_index` uygulanir.
- Generation per-sample sequential calistirilir (batch hizindan bilincli feragat).
- Her sample oncesi global RNG durumu set edilerek deterministik sampling akisi uygulanir.
- Reproducibility dogrulamasi eklendi:
	- Ayni model + ayni prompt + ayni seed ile ayni output/trajectory kontrol edilir.
	- Basarisizlik durumunda pipeline hata verir.

## 3) Prompt-template leakage duzeltmesi

- Tek stem + sayi yapisi kaldirildi.
- Her kategori icin coklu template bankasi eklendi.
- Metadata'ya su alanlar eklendi:
	- `template_id`
	- `prompt_id`
	- `topic_id`
- Category prediction artik grouped split ile calisir:
	- `GroupShuffleSplit(groups=prompt_id)`
- Eger train ve testte tum siniflari kapsayan grouped split bulunamazsa sessiz fallback yapilmaz.
	- Varsayilan politika: `error` (calisma durur).
	- Opsiyonel politika: `warn` (acik uyari verir ve coverage eksigi metriklere yazilir).
- `metrics.json` ve `report.md` icinde grouped coverage durumu acikca raporlanir (`coverage_satisfied`, eksik sinif listeleri, uyari mesaji).

## 4) Model identity leakage duzeltmesi

- `src/extract_features.py` icine trajectory normalization eklendi.
- Iki feature ailesi acikca ayrildi:
	- `raw_geo_*` (baseline, scale-sensitive)
	- `norm_geo_*` + `si_*` (normalized/scale-invariant, main)
- Model-spesifik PCA feature'lari yeniden adlandirildi:
	- `within_model_pca_*`
- Bu PCA feature'lari cross-model classification'da kullanilmiyor.
- `src/predict.py` iki deney raporlar:
	- raw baseline
	- normalized/scale-invariant main

## 5) Gorsellestirme gecerliligi

- Ayni eksende farkli PCA bazlarini ust uste bindirme kaldirildi.
- Yeni gorseller:
	- `trajectories_model_specific_pca.png`
		- model-basina ayri panel
		- panel eksenlerinin dogrudan karsilastirilamaz oldugu acikca belirtilir
	- `common_feature_space_pca.png`
		- ortak trajectory-level feature uzayinda PCA
		- eksenler modeller arasi karsilastirilabilir

## Ek teknik iyilestirmeler

- `src/predict.py` icinde Logistic Regression icin `StandardScaler` eklendi.
- Metrikler sadece stdout'a yazilmiyor:
	- `results/metrics.json`
	- `results/report.md`
- Split politikalari acik:
	- model identity: stratified split
	- category: grouped split
- `run_all.sh` mode bilgisini (`smoke` / `full`) tum adimlara tasir.
- `run_all.sh` grouped coverage politikasini acikca `predict.py`'a gecer.
	- Varsayilan: smoke icin `warn`, full/custom icin `error`.

## Dosya bazli degisiklik ozeti

- `src/collect_data.py`: reproducible data collection, run isolation, template bank, seed verification.
- `src/extract_features.py`: normalized/scale-invariant features, within-model PCA isolation.
- `src/predict.py`: leakage-aware experiments, grouped split, persisted metrics/report.
- `src/visualize.py`: analytically valid visualization strategy.
- `run_all.sh`: explicit, reproducible, mode-aware entrypoint.
