# Machine-Learning-Random-Forest
# Random Forest
# Random Forest ile Iris Veri Seti Sınıflandırma
Bu proje, Random Forest algoritması kullanarak Iris veri setinin sınıflandırılmasını içermektedir. Proje Python dilinde yazılmıştır ve pandas, numpy, matplotlib.pyplot ve sklearn kütüphaneleri kullanılmıştır.

# Veri Seti
Proje, Iris veri setini kullanmaktadır. Bu veri seti, bitki türlerinin özelliklerini içermektedir. Toplamda 5 sütundan oluşmaktadır ve bağımlı değişken olarak "species" sütunu bulunmaktadır. Bağımsız değişkenler ise bitki türlerinin özelliklerini ifade etmektedir.

# Random Forest Algoritması
Random Forest, birden fazla karar ağacının bir araya getirilmesiyle oluşturulur. Her bir karar ağacı, bir alt küme (bootstrap sample) veri seti kullanılarak eğitilir. Ayrıca, her bir karar ağacı için rastgele bir alt küme değişken belirlenir. Bu sayede, overfitting probleminin önüne geçilir.

Proje, sklearn kütüphanesinde bulunan RandomForestClassifier sınıfı kullanılarak Random Forest algoritması uygulanmıştır. Eğitim verileri ve test verileri için ayrı ayrı X_train, X_test, Y_train ve Y_test değişkenleri oluşturulmuştur. RandomForestClassifier sınıfı n_estimators parametresi ile kullanılacak karar ağacı sayısı belirlenmiştir.

# Performans Ölçümü
Sınıflandırma performansı, confusion matrix ve accuracy metrikleri kullanılarak ölçülmüştür. Confusion matrix, gerçek ve tahmin edilen sınıfların karşılaştırılması sonucunda oluşan bir tablodur. Accuracy, doğru sınıflandırılmış örneklerin toplam sayısının tüm örneklerin toplam sayısına oranıdır.

# Çıktı
Proje sonucunda, 50 test verisi içinden 48 tanesinin doğru bir şekilde sınıflandırıldığı görülmüştür. Confusion matrix sonucu ise aşağıdaki gibidir:

[[16  0  0]
 [ 0 18  1]
 [ 0  2 13]]
Accuracy değeri ise 0.96 olarak hesaplanmıştır.
