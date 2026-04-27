# Konudan Habersiz Yazilimci Icin Rehber

Bu dokuman, NLP veya LLM ic yapilari hakkinda hic bilgisi olmayan ama yazilim bilen bir kisinin bu odevin ne oldugunu anlamasi icin yazildi.

Amac su:

- "Token ne?"
- "Yorunge ne?"
- "Bu kod ne yapiyor?"
- "Bu projede benim islevim ne?"
- "Sonuclar ne anlatiyor?"

sorularini basit bir dille cevaplamak.

## 1. En Kisa Ozet

Bu odevde iki farkli dil modeli seciliyor.

Bu modellerden metin uretmeleri isteniyor.

Metin uretirken, modelin her adimdaki ic sayisal durumu kaydediliyor.

Bu kayitlar sirali bir vektor dizisi olusturuyor.

Bu diziye `yorunge` deniyor.

Sonra bu yorungelerden sayisal ozellikler cikariliyor ve su gibi sorular soruluyor:

- Bu yorunge hangi modelden geldi?
- Bu yorunge hangi prompt kategorisinden geldi?

Yani odevin ozeti su:

`Modelin yazdigi metne degil, yazarken iceride nasil hareket ettigine bakiyoruz.`

## 2. Token Nedir

Dil modeli metni harf harf veya kelime kelime degil, `token` denilen parcalar halinde isler.

Token bazen tam kelime olabilir.

Bazen kelimenin bir parcasi olabilir.

Bazen noktalama veya bosluk da token olabilir.

Ornek:

- `"Istanbul"` tek token olabilir
- ya da `"Ist" + "anbul"` gibi iki parcaya bolunebilir
- `"hello,"` bazen `"hello"` ve `","` diye ayrilabilir

Yani token, modelin metni islerken kullandigi temel parcadir.

## 3. Hidden State Nedir

Model bir token gordugunde veya yeni bir token urettiginde, iceride bir suru sayisal hesap yapar.

Bu hesaplarin sonunda elinde bir vektor olur.

Bu vektor, modelin o ana kadar ne anladigini ve bir sonraki tokeni nasil sececegini temsil eden ic durumdur.

Buna kabaca `hidden state` denir.

Bu odevde ozellikle ilgilendigimiz sey:

- `son katman`
- `son token`
- `hidden state`

Yani:

`Model tam karar vermeden once, en son gordugu/urettigi token icin hangi vektor durumundaydi?`

## 4. Yorunge Nedir

Model bir kere metin uretmeye basladiginda tek bir karar vermez.

Adim adim ilerler.

Mesela 50 token uretirse, 50 farkli anda 50 farkli internal durum olusur.

Eger her adimda bu son hidden state vektorunu alirsan, elinde su olur:

- adim 1 vektoru
- adim 2 vektoru
- adim 3 vektoru
- ...
- adim 50 vektoru

Bu vektor dizisine `yorunge` denir.

Yani yazilimci diliyle:

- her trajectory bir `time series`
- her zaman adiminda bir `vector snapshot`
- tum seri birlikte modelin ic hareketini temsil ediyor

Bir baska benzetme:

- GPS ile bir arabanin nerelerden gectigini kaydetmek gibi
- burada araba modelin ic durumu
- konum yerine yuksek boyutlu vektorler var

## 5. Bu Odevde Tam Olarak Ne Yapiyoruz

Bu proje su adimlari yapiyor:

1. Iki model seciyor.
2. Her modele ayni tipte promptlar veriyor.
3. Her modelden metin urettiriyor.
4. Uretim sirasinda her adimda son-token hidden state'ini kaydediyor.
5. Her ornek icin bir trajectory dosyasi olusturuyor.
6. Bu trajectory'lerden sayisal ozellikler cikariyor.
7. Bu ozelliklerle siniflandirma deneyi yapiyor.
8. Sonuclari raporluyor ve gorsellestiriyor.

## 6. Bu Kodun Icinde Hangi Dosya Ne Is Yapiyor

Projede ana dosyalar sunlar:

- `src/collect_data.py`
- `src/extract_features.py`
- `src/predict.py`
- `src/visualize.py`
- `run_all.sh`

Rolleri:

### `src/collect_data.py`

Bu dosya veri toplar.

Yani:

- modeli yukler
- promptlari uretir
- metin uretimini calistirir
- her adimda hidden state toplar
- trajectory dosyalarini `.npy` olarak kaydeder
- metadata CSV yazar

### `src/extract_features.py`

Bu dosya ham trajectory'leri ozet sayilara cevirir.

Cunku makine ogrenmesi asamasinda 50x768 veya 50x512 gibi ham tensorlerle degil, daha kompakt ozelliklerle calismak daha kolaydir.

Ornek ozellikler:

- ardIsik adimlar arasi mesafe ortalamasi
- yol uzunlugu
- linearity
- normalized geometric feature'lar
- scale-invariant feature'lar

### `src/predict.py`

Bu dosya siniflandirma deneyini yapar.

Iki ana soru sorar:

- Bu trajectory hangi modelden geldi?
- Bu trajectory hangi kategoriye ait prompttan geldi?

Sonra accuracy ve F1 gibi metrikleri hesaplar.

### `src/visualize.py`

Bu dosya cizimler uretir.

Amac:

- trajectory mantigini gorsel olarak gostermek
- modellerin dagilimini ortak ozellik uzayinda gormek

### `run_all.sh`

Tum pipeline'i tek komutla kosar:

1. data toplama
2. feature cikarimi
3. prediction
4. visualization

## 7. Feature Nedir, Neden Lazim

Trajectory ham halde buyuk bir dizidir.

Mesela:

- 50 adim
- her adimda 768 boyut

Bu dogrudan yorumlamak zor.

O yuzden bunu ozet sayilara ceviriyoruz.

Buna `feature extraction` denir.

Yazilimci gozunden dusun:

- ham trajectory = detayli log stream
- feature = bu login ozet istatistikleri

Ornek:

- ortalama adim buyuklugu
- en buyuk sicrama
- ne kadar duz gidiyor
- ne kadar zikzak yapiyor

## 8. Bu Projede Iki Tahmin Gorevi Ne

### Gorev 1: Model Kimligi

Soru:

`Bu trajectory gpt2 mi, yoksa pythia mi?`

Eger feature'lar bu iki modeli iyi ayiriyorsa, modelin ic hareket tarzi farkli demektir.

### Gorev 2: Prompt Kategorisi

Soru:

`Bu trajectory History mi, Science mi, Technology mi, Art mi, Sports mu?`

Eger category tahmini iyi cikarsa, modelin ic hareketi prompt tipine gore de anlamli sekilde degisiyor olabilir.

## 9. Mevcut Calismanin Sonuclari Ne Diyor

Su an `runs/hw2_full_seed42/results/report.md` altinda tam kosu raporu var.

Oradaki temel sayilar:

- toplam trajectory sayisi: `2000`
- model sayisi: `2`
- kategori sayisi: `5`

Model kimligi tahmini:

- normalized RF accuracy: `0.9975`
- normalized LR accuracy: `1.0000`

Prompt kategorisi tahmini:

- grouped normalized RF accuracy: `0.2143`
- grouped normalized LR accuracy: `0.2071`

Ilk sonuc ne diyor:

- iki modelin trajectory ozellikleri birbirinden cok farkli
- yani `hangi model oldugunu` anlamak kolay

Ikinci sonuc ne diyor:

- kategori tahmini neredeyse sans seviyesine yakin
- 5 kategori varsa rasgele tahmin kabaca `0.20` civarindadir
- yani mevcut feature'larla prompt kategorisi belirgin sekilde ayristirilmamis

## 10. Bu Sonuclar Yazilimci Gozunden Ne Anlama Geliyor

Su sekilde dusun:

- Iki farkli servisin runtime davranis loglarini aliyorsun
- log ozelliklerinden hangi servis oldugunu cok iyi tahmin edebiliyorsun
- ama istegin hangi is sinifina ait oldugunu cok iyi tahmin edemiyorsun

Bu projede de benzer durum var:

- `model farki` cok belirgin
- `prompt kategorisi farki` pek belirgin degil

## 11. Peki Sorun Neydi, Neden Review Yapildi

Bu kisim onemli.

Ilk versiyonlarda su sorunlar vardi:

- ayni prompt saploni tekrar tekrar kullaniliyordu
- grouped split dogru degildi
- seed gercekten uygulanmiyordu
- overwrite eski dosyalari karistirabiliyordu
- bazi feature'lar modeli haksiz sekilde ele verebiliyordu
- bazi gorseller ayni koordinat sisteminde karsilastirilamazdi

Yani sorun "kod calismiyor" degildi.

Sorun su idi:

`Kod calisiyor olabilir ama deney metodolojik olarak dogru mu?`

Review bunun icin yapildi.

Simdi kod daha guvenilir hale getirildi.

## 12. Sen Bu Projede Ne Yaptin Aslinda

Senin yaptigin sey bir bakima su:

- iki modeli bir kara kutu gibi aldin
- sadece ciktisina degil, ic hareketine de baktin
- bu ic hareketi sayisal ozelliklere cevirdin
- sonra bu ozelliklerin ne kadar bilgi tasidigini test ettin

Bu, klasik "text classification" degil.

Bu daha cok:

`representation analysis`

veya

`model behavior analysis`

gibi dusunulebilir.

## 13. Sunumda Bunu Nasil Anlatmalisin

Asagidaki hikaye en temiz anlatim olur:

1. Iki dil modeli secildi.
2. Her modelden 1000 ornek trajectory toplandi.
3. Her trajectory en az 50 tokenlik uretim sirasinda olusturuldu.
4. Her adimda son-token son-layer hidden state kaydedildi.
5. Bu trajectory'lerden geometric ve normalized/scale-invariant feature'lar cikarildi.
6. Iki tahmin gorevi kuruldu:
   - model identity
   - prompt category
7. Sonuclar gosterdi ki model kimligi cok guclu sekilde tahmin edilebiliyor.
8. Buna karsin prompt category mevcut feature set ile iyi ayristirilmiyor.
9. Bu da trajectory'lerin model spesifik imzalar tasidigini, ama category bilgisinin daha zayif oldugunu gosteriyor olabilir.

## 14. Hocaya Soylenebilecek Basit Yorum

Su yorum guvenli ve anlasilir:

"Bu calismada modelin sadece urettigi metne degil, uretim sirasindaki ic temsil hareketine baktik. Iki modelin trajectory ozellikleri birbirinden cok net ayrildi. Buna karsin prompt kategorileri bu ozelliklerle kolayca ayristirilamadi. Bu durum, trajectory'lerin once model mimarisi veya modelin genel dinamikleri tarafindan daha guclu sekillendigini dusundurebilir."

## 15. Bu Sonuclari Abartmadan Nasil Yorumlamalisin

Sunlari demek guvenli:

- model trajectory'leri farkli davraniyor gibi gorunuyor
- secilen feature'lar model ayrimi icin bilgi tasiyor
- category ayrimi icin secilen feature'lar yeterince guclu degil
- farkli feature'lar veya farkli modellerle sonuclar degisebilir

Sunlari demek fazla iddiali olur:

- "LLM'lerin dusunme seklini kesin cozdum"
- "Bu trajectory dogrudan anlamsal kategoriyi temsil ediyor"
- "Bu sonuc butun modeller icin genellenir"

## 16. Sen Simdi Pratikte Ne Yapacaksin

Eger bu projeyi teslim ve sunum acisindan dusunuyorsan, senin gorevin artik su:

1. Kodun ne yaptigini yukaridaki mantikla anla.
2. Sonuclari ezberleme, mantigini anla.
3. Sunumda teknik terimi basit dile cevir.
4. Ozellikle su mesaji ver:
   - "Biz texti degil, internal representation hareketini inceledik."
5. Model identity neden yuksek, category neden dusuk bunu sade dille acikla.
6. Limitasyonlari durustce soyle.

## 17. Tek Cumlelik Final Ozet

Bu proje, dil modellerinin metin uretirken iceride olusturdugu vektor hareketlerini kaydedip bunlardan anlamli bilgi cikarilabilir mi sorusuna bakiyor.

Su anki sonuc:

- `hangi model oldugu` bu hareketlerden neredeyse tam bulunabiliyor
- `hangi konu kategorisinden geldigi` ise bu feature'larla cok iyi bulunamiyor

Bu da projeyi "LLM ic davranis analizi" olarak dusunmen gerektigini gosteriyor.
