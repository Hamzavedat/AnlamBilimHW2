# Hesaplamali Anlambilim 2. Odev Analizi

## Kaynak
- PDF basligi: `Hesaplamali Anlambilim Dersi 2.Odevi`
- Konu: `Yorungeler`
- Teslim edilecekler: `Kod, sunum`
- Son teslim: `21 Nisan 2026, 09:30`

## Odev Ne Istiyor
Bu odev klasik bir "tek bir algoritma yaz" odevinden cok, kucuk capli bir arastirma ve deney tasarimi istiyor.

Istenen sey su:

1. Iki farkli dil modeli secilecek.
2. Her model icin en az `1000` adet yorunge toplanacak.
3. Her yorunge en az `50 token` uzunlugunda olacak.
4. Uretim sirasinda, her adimda modelin `son token` icin `son katmandaki temsil vektoru` alinacak.
5. Bu vektorler sirayla kaydedilecek ve boylece bir `yorunge` olusacak.
6. Bu yorungeler en az `2 farkli ozellik kumesi` ile temsil edilecek.
7. Bu ozelliklerle `2 farkli sey` tahmin edilecek.
8. Hangi ozelliklerin kullanilacagi ve neyin tahmin edilecegi ogrencinin arastirma karari olacak.
9. Sonucta yorum agirlikli bir sunum hazirlanacak.
10. Ayrica "once siz cozun, sonra YZ'ye cozdurun, 2 cozumu karsilastirin" beklentisi var.

## Bu Odevde Kodlama Var Mi
Evet, var.

Bu cunku:

- Teslim edilecekler arasinda dogrudan `kod` yaziyor.
- Uretim sirasinda hidden state/toplam temsil toplamak pratikte kod gerektirir.
- En az `2 model x 1000 yorunge` gibi bir olcek manuel yapilamaz.
- Ozellik cikarimi, tahmin, gorsellestirme ve raporlama icin programlama gerekir.

Kisacasi bu odevin teknik cekirdegi yazilimdir; uzerine arastirma yorumu ve sunum eklenir.

## Hic Konuyu Bilmeyen Yazilimci Icin Sade Anlatim
Bu odevi soyle dusun:

- Bir dil modeli metin uretirken iceride bir suru sayisal vektor hesaplar.
- Her yeni token uretildiginde, modelin o andaki "ic durumu" son katmanda bir vektor olarak temsil edilir.
- Sen bu vektoru her adimda alip sakliyorsun.
- Boylece zaman icinde degisen bir vektor dizisi elde ediyorsun.
- Bu diziye `yorunge` deniyor.

Yani burada incelenen sey metnin kendisi degil, modelin metni uretirken ic temsillerinin nasil hareket ettigi.

## Teknik Kavramlarin Sade Karsiliklari
- `Token`: Metnin modele gore parcasi. Her zaman tam kelime degildir.
- `Son token`: Modelin o adimda en son isledigi/urettigi token.
- `Son katman temsili`: Modelin karar vermeden hemen onceki son ic sayisal temsili.
- `Yorunge`: Bu temsillerin zaman icindeki sirali listesi.
- `Ozellik cikarimi`: Ham vektor dizisini daha kucuk, ozet ve ogrenilebilir sayilara donusturme islemi.
- `Tahmin`: Bu ozetlerden sinif veya hedef degisken cikarmaya calismak.

## Odevin Acik Biraktigi Arastirma Kararlari
Odev su noktalarin hepsini ogrenciye birakiyor:

- Hangi 2 model kullanilacak
- Prompt'lar nasil secilecek
- Yorungelerden hangi ozellikler cikarilacak
- Hangi 2 seyin tahmin edilecegi
- Hangi gorsellestirme yontemlerinin kullanilacagi
- Sonuclarin nasil yorumlanacagi

Bu nedenle odevin en kritik kismi sadece kod yazmak degil, dogru deney tasarimi kurmak.

## En Guvenli Teknik Yorum
Bu odev icin en mantikli yol, hidden state'e erisebilecegin acik kaynak modellerle calismaktir.

Bunun sebebi:

- Son katman temsili lazim.
- Bu temsil `lm_head` veya cikis olasiliklari oncesinde alinmali.
- Kapali servisler/API'ler bunu genelde dogrudan vermez.
- Acik kaynak `transformers` tipi modeller bu isi daha kolay hale getirir.

## Odevin Bekledigi Olasi Cikti Turleri
- Ham trajectory verisi
- Trajectory ozellik tablolari
- 2 tahmin problemi icin deney sonuclari
- 2B gorseller
- Yorumlar
- Sunum slaytlari

## Guvenli Baslangic Stratejisi
Baslangic icin en dusuk riskli yaklasim su olur:

1. Iki acik kaynak causal language model sec.
2. Ortak bir prompt havuzu kur.
3. Her modelden en az 1000 yorunge topla.
4. Her yorunge icin en az 50 token uret.
5. Geometrik trajectory ozellikleri cikar.
6. Ayrica 2 boyutlu indirgeme tabanli ikinci bir ozellik kumesi cikar.
7. Ilk tahmin problemi olarak `hangi model?` sorusunu sec.
8. Ikinci tahmin problemi olarak `hangi prompt kategorisi?` veya `tekrar var mi?` sorusunu sec.
9. Sonuclari basit ama acik metriklerle yorumla.

## Sunumda Beklenen Ana Hikaye
Sunumda yalnizca teknik boru hatti degil, asagidaki hikaye de kurulmeli:

- Neden bu modeller secildi?
- Neden bu prompt'lar secildi?
- Neden bu trajectory ozellikleri anlamli?
- Hangi tahmin problemi neden secildi?
- Sonuclar ne diyor?
- Yorungelerden model davranisi hakkinda ne ogrendik?
- Insan cozumune karsi YZ cozumu nasil fark yaratti?

## Kisa Sonuc
Bu odev:

- kodlama gerektirir,
- deney tasarimi gerektirir,
- veri toplama ve analiz gerektirir,
- yorum agirlikli sunum gerektirir.

Yani bu bir mini arastirma projesidir.
