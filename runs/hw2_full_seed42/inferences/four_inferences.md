# Four Ready-to-Use Inferences

Bu dosya, odevdeki 'iki yorumu ben yaptim, iki yorumu YZ onerdi' ayrimini kolaylastirmak icin hazirlandi.

## Kendi yorumun gibi kullanabilecegin 2 cikarim

### Human-like 1: GPT-2 trajectory'leri daha uzun ve daha dalgali
- GPT-2 ortalama raw path length: 5822.3
- Pythia ortalama raw path length: 2507.4
- GPT-2 step variability (CV): 0.911
- Pythia step variability (CV): 0.220
- GPT-2 mean abs turn: 0.604
- Pythia mean abs turn: 0.393
- Yorum: GPT-2 temsil uzayinda daha fazla dolasiyor; Pythia daha kisa ve daha kontrollu bir yol izliyor.
- Gorsel: `runs/hw2_full_seed42/inferences/human_1_model_style.png`

### Human-like 2: GPT-2 prompt turune daha hassas, Pythia daha uniform
- GPT-2 category area range: 0.7525
- Pythia category area range: 0.0795
- GPT-2 en genis area veren kategori: Physics
- Pythia en genis area veren kategori: Economics
- Yorum: Kategori degistiginde GPT-2'nin trajectory sekli daha belirgin oynuyor; Pythia ise konudan daha az etkileniyor.
- Gorsel: `runs/hw2_full_seed42/inferences/human_2_prompt_sensitivity.png`

## YZ yardimiyla sunabilecegin 2 cikarim

### AI-like 1: Ortak feature uzayinda model sinyali kategori sinyalinden cok daha guclu
- En iyi model identity RF accuracy: 1.0000
- En iyi grouped category RF accuracy: 0.0975
- Sans seviyesi 10 kategori icin yaklasik 0.10'dir.
- Yorum: Trajectory ozellikleri once modeli ele veriyor; konu kategorisi ayni gucle ayrismiyor.
- Gorsel: `runs/hw2_full_seed42/inferences/ai_1_shared_space.png`

### AI-like 2: Model ayrimi birkac baskin dagilim ozelligiyle geliyor, kategori sinyali daha daginik
- Model identity icin en guclu ilk 4 feature:
  - within_model_pca_area: 0.1847
  - within_model_pca_spread: 0.1840
  - norm_geo_cos_sim_std: 0.1346
  - si_turn_std: 0.1275
- Category prediction icin en guclu ilk 4 feature:
  - within_model_pca_spread: 0.0685
  - within_model_pca_area: 0.0636
  - within_model_pca_linearity: 0.0625
  - si_step_min_over_mean: 0.0606
- Yorum: Model sinyali daha sivri ve baskin; kategori sinyali ise bircok kucuk ozellige dagiliyor.
- Gorsel: `runs/hw2_full_seed42/inferences/ai_2_feature_importance.png`