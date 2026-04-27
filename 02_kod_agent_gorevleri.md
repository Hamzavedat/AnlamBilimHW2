# Kod Agenti Icin Gorev Listesi

Bu dosya, konuyu bilmeyen bir kod agentinin odevi teknik olarak yurutmesi icin hazirlanmis net gorev listesidir.

## Genel Amac
Iki farkli dil modelinin metin uretimi sirasinda olusan son-katman son-token temsillerini topla, bu temsillerden trajectory veri seti olustur, trajectory ozellikleri cikar, bu ozelliklerle iki farkli tahmin problemi kur ve sonuclari gorsellestir.

## Ana Ciktilar
- Trajectory toplama kodu
- Ozellik cikarim kodu
- Etiketleme/tahmin pipeline'i
- Gorsellestirme kodu
- Sonuc tablolari
- Sunumda kullanilabilecek grafikler

## Adim Adim Teknik Gorevler
1. Calisma ortamini kur.
2. Hidden state erisimi olan 2 uygun acik kaynak model sec.
3. Her iki model icin ortak bir uretim arabirimi yaz.
4. Prompt veri setini hazirla veya kategori bazli prompt havuzu olustur.
5. Uretim parametrelerini sabitle:
   - `max_new_tokens >= 50`
   - sabit `temperature`
   - sabit `top_p`
   - sabit `seed` politikasi
6. Uretim sirasinda her adimda `son katman -> son token -> hidden state` vektorunu yakala.
7. Her calisma icin asagidaki bilgileri kaydet:
   - model adi
   - prompt metni
   - prompt kategorisi
   - seed
   - olusan metin
   - token sayisi
   - trajectory tensor veya numpy dizisi
8. Her model icin en az `1000` trajectory topla.
9. Her trajectory'nin en az `50 token` uzunlukta oldugunu dogrula.
10. Ham veriyi tekrar uretilebilir ve kolay okunur formatta sakla.

## Ozellik Kumesi Gorevleri
En az 2 farkli trajectory ozellik kumesi cikar.

### Onerilen Ozellik Kumesi 1: Geometrik/Istatistiksel
- ArdIsik noktalar arasi Oklid mesafesi
- ArdIsik noktalar arasi kosinus benzerligi
- Yon degisimi / aci degisimi
- Toplam yol uzunlugu
- Baslangic-son nokta uzakligi
- Mesafe serisinin ortalama/std/max/min degerleri

### Onerilen Ozellik Kumesi 2: Dusuk Boyutlu Gosterim Tabanli
- PCA / UMAP / t-SNE ile 2B izdusum
- 2B noktalarin yayilimi
- 2B alansal dagilim ozetleri
- Sikisiklik / daginiklik olculeri
- 2B uzayda hareketin dogrusal veya kivrimli olma olculeri

## Tahmin Gorevleri
En az 2 farkli tahmin problemi kur.

### Onerilen Tahmin Problemi 1
Bir trajectory'nin hangi model tarafindan uretildigini tahmin et.

### Onerilen Tahmin Problemi 2
Asagidakilerden birini sec:
- prompt kategorisini tahmin et
- uretimde tekrar olup olmadigini tahmin et
- cikti kalitesine dair basit bir etiketi tahmin et
- entropi/cesitlilik seviyesini tahmin et

## Modelleme Gorevleri
1. Ozellik tablosu olustur.
2. Train/validation/test veya train/test ayrimi yap.
3. Basit ve aciklanabilir baseline modeller kur:
   - Logistic Regression
   - Random Forest
   - XGBoost veya benzeri
4. Her ozellik kumesi icin ayri deney kos.
5. Sonuclari tablo halinde karsilastir.

## Gorsellestirme Gorevleri
1. Ornek trajectory'leri 2B uzayda ciz.
2. Iki modelin trajectory dagilimlarini ayni grafik ailesinde karsilastir.
3. Tahmin performanslarini bar chart veya tablo ile goster.
4. Yoruma yardimci olacak ornek trajectory vakalari sec.

## Kalite Kontrol Gorevleri
1. Trajectory uzunlugu filtrelemesini kontrol et.
2. Hidden state'in dogru katmandan alindigini test et.
3. Son token temsilinin boyutlarini logla.
4. Iki model icin de ayni veri toplama protokolunun uygulandigini dogrula.
5. Eksik veya bozuk trajectory'leri ayikla.

## Dokumantasyon Gorevleri
1. Kullanilan modelleri ve neden secildiklerini yaz.
2. Veri toplama protokolunu yaz.
3. Ozellik tanimlarini acikla.
4. Iki tahmin probleminin mantigini yaz.
5. Sinirliliklari ve riskleri not et.

## Sunum Icin Uretilecek Maddeler
- Problem tanimi
- Yontem
- Kullanilan modeller
- Trajectory tanimi
- Ozellik kumeleri
- Tahmin gorevleri
- Sonuclar
- Yorumlar
- Insan cozumu ve YZ cozumu karsilastirmasi

## Kod Agentina Verilebilecek Kisa Prompt
Asagidaki prompt, ayri bir kod agentina verilebilir:

`Iki acik kaynak causal language model sec. Metin uretimi sirasinda her adimda son katmandaki son-token hidden state vektorunu toplayarak en az 1000'er trajectory olustur. Her trajectory en az 50 token olsun. Trajectory'lerden en az iki farkli ozellik kumesi cikar: bir geometrik/mesafe/aci tabanli, bir de 2B indirgeme tabanli. Bu ozelliklerle iki tahmin problemi kur: model kimligi ve prompt kategorisi. Sonuclari tablo ve grafiklerle raporla. Kod moduler, tekrar uretilebilir ve kayit/log dostu olsun.`

## Kisa Not
Bu odevde agentin isi sadece kod yazmak degil; deney tasarimi, veri saklama duzeni, etiketleme ve yorumlamayi da sistematik hale getirmektir.
