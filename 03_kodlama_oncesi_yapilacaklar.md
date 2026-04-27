# Kodlama Oncesi Yapilacaklar

Bu dosya, kod yazmaya baslamadan once netlestirilmesi gereken kararlarin kontrol listesidir.

## 1. Arastirma Tasarimini Netlestir
- Odevin cekirdek sorusunu bir cumle ile yaz.
- Iki modelin neden secildigini belirle.
- En az iki ozellik kumesinin neden anlamli oldugunu yaz.
- Tahmin edilecek iki hedefin neden secildigini acikla.

## 2. Model Secimi
- Hidden state erisimi olan acik kaynak modeller sec.
- Modellerin ayni aileden mi farkli ailelerden mi olacagina karar ver.
- Model boyutlarinin donanimda calisip calismayacagini kontrol et.
- Tokenizer farklarinin analizi nasil etkileyecegini not et.

## 3. Prompt Tasarimi
- Ortak prompt havuzu kullanilip kullanilmayacagina karar ver.
- Prompt'lari kategorilere ayir.
- Her kategori icin yeterli sayida prompt oldugundan emin ol.
- Prompt'larin cok kisa veya cok dengesiz olmamasina dikkat et.

## 4. Uretim Protokolu
- Her trajectory icin minimum token uzunlugunu `50` olarak sabitle.
- `temperature`, `top_p`, `max_new_tokens`, `seed` gibi ayarlari onceden belirle.
- Iki modelde de ayni uretim protokolunun kullanilacagini yazili hale getir.
- Durma kriterlerini belirle.

## 5. Trajectory Tanimi
- Hangi katmandan veri alinacagini netlestir.
- Her adimda sadece `son token` temsilinin alinacagini not et.
- Hidden state'in `lm_head` oncesinden geldigini dogrula.
- Trajectory saklama formatina karar ver:
  - `npy`
  - `pt`
  - `parquet`
  - `json metadata + binary tensor`

## 6. Etiketleme Karari
- Ilk tahmin hedefi olarak `model kimligi` secilecek mi?
- Ikinci tahmin hedefi olarak ne secilecek:
  - prompt kategorisi
  - tekrar var/yok
  - kalite etiketi
  - cesitlilik seviyesi
- Etiketlerin otomatik uretilebilir olmasina dikkat et.

## 7. Ozellik Kumesi Karari
- Birinci ozellik kumesi olarak geometrik ozellikleri secip secmeyecegine karar ver.
- Ikinci ozellik kumesi olarak PCA/UMAP tabanli ozellikleri secip secmeyecegine karar ver.
- Ozelliklerin yorumlanabilir olmasina oncelik ver.

## 8. Degerlendirme Plani
- Siniflandirma metriklerini belirle:
  - accuracy
  - F1
  - confusion matrix
- Veri bolme stratejisini belirle.
- Baseline modelleri onceden sec.
- Sonuc tablosu formatini bastan planla.

## 9. Gorsellestirme Plani
- Hangi trajectory grafiklerinin alinacagini belirle.
- 2B indirgeme icin hangi yontemin kullanilacagini sec.
- Sunumda kullanilacak sabit grafik setini bastan tanimla.

## 10. Klasorleme ve Dosyalama
- Ham veri klasoru
- Islenmis veri klasoru
- Ozellik tabloları klasoru
- Grafik klasoru
- Notlar/sunum klasoru

Onerilen temel yapi:

```text
project/
  data_raw/
  data_processed/
  features/
  figures/
  notebooks/
  src/
  presentation/
```

## 11. Riskler
- Model hidden state cikarma kisminin beklenenden zor olmasi
- Uzun uretimlerin yavas olmasi
- Bellek kullanimini patlatan trajectory kayitlari
- Tokenizer farklarindan dogan dengesizlikler
- Etiketlerin zayif veya anlamsiz kalmasi

## 12. Kodlamaya Baslamadan Once Son Kontrol
- Iki model secildi mi?
- Prompt havuzu hazir mi?
- Uretim ayarlari sabitlendi mi?
- Trajectory tanimi net mi?
- Iki ozellik kumesi secildi mi?
- Iki tahmin hedefi secildi mi?
- Sonuc metrikleri belirlendi mi?
- Sunum iskeleti kabaca cizildi mi?

## Onerilen Pratik Karar Seti
Karar veremiyorsan, baslangic icin su set en guvenli secimdir:

- 2 acik kaynak causal language model
- ortak prompt havuzu
- minimum 50 token
- ozellik kumesi 1: mesafe + aci + yol uzunlugu
- ozellik kumesi 2: PCA veya UMAP tabanli 2B ozetler
- tahmin 1: model kimligi
- tahmin 2: prompt kategorisi
- baseline: Logistic Regression + Random Forest

## Sonuc
Bu liste tamamlanmadan kodlamaya baslamak, proje icinde surekli karar degistirmeye neden olur. O yuzden once deney tasarimini kilitlemek, sonra implementasyona gecmek en dogru yaklasimdir.
