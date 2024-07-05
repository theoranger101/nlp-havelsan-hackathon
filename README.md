# 1. Çok-sınıflı Metin Sınıflandırma
İlk olarak elimizdeki verileri ön işlemeden geçirdik. Klasör düzenindeki dosyaları
kategori kategori okuyacak şekilde sistemi yazdık. Ardından bu klasörlerin içindeki
metinlerdeki büyük harfleri küçüğe çevirip tokenlerine (kelime gruplarına) ayırdık.
Ayrıca veri sayısı az olan kategoriler için rastgele bir veri arttırma metodu uyguladık.
TF-IDF vectorization yöntemini kullanarak kelimelerin farklı kategorilere göre
ağırlıklarını yani kullanılma frekanslarını hesapladık.
Classification algoritması için birkaç farklı model denedik (örneğin Random Forest)
ve aralarında en iyi değerleri Logistic Regression algoritması ile yakaladığımız için
onu tercih ettik. TF-IDF yönteminden elde ettiğimiz verileri classification için
kullandık. Ayrıca birkaç parametre arasından en iyi değerleri vereni seçebilmek için
Logistic Regression’ı Grid Search ile çalıştırdık. Ardından en iyi tahminleri veren
Logistic Regression değerlerini evaluate ettik.
Sonuç olarak en iyi değerleri
```
LogisticRegression(C=100, solver='lbfgs', max_iter=1000)
```
bu parametrelerle elde ettik.

# 2. Anlamsal Arama
Bu problem için de aynı şekilde veri okumada kolaylık sağlaması için büyük harfleri
küçüğe çevirdiğimiz basit bir ön işleme ile başladık. Bir önceki problemde
kullandığımız TF-IDF Anlamsal arama üzerine ChatGPT ve internetten yardım
alarak bir metod oluşturduk. Kosinüs benzerliği ve TF-IDF ile vektörleştirdiğimiz
değerleri kullanarak en büyük benzerliği gösteren 5 kelime grubunu seçtiğimiz bir
algoritma izledik. Ardından pdf’te verilen örnek sorularla programımızı test ettik.
Değerler verilen örneğe tam olarak uymadığı için farklı gömme modelleri kullanmayı
denedik fakat daha iyi değerlere ulaşamadık.
