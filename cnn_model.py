#Convolutional Neural Network(CNN)
#Keras kütüphanelerini ve paketlerini içe aktarma
#Keras, Python'da yazılmış açık kaynaklı bir sinir ağı kütüphanesidir.
#Keras TensorFlow, Microsoft Cognitive Toolkit, R, Theano veya PlaidML ile beraber çalışabilir. Derin sinir ağları ile hızlı deney yapabilmek için tasarlanan bu cihaz kullanıcı dostu, modüler ve genişletilebilir olmaya odaklanıyor.
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras import optimizers

#https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
#https://keras.io/preprocessing/image/


# CNN modeli Başlatma oluşturma
classifier = Sequential() #Ardışık sıralı kerasda model oluşturmanın en kolay yoludur

#Modelimize katman eklemek için .add kullanılır
#Convolution2d katmanı 2 boyutlu matrisler olarak görülen girdi görüntülerimizle ilgilenecek evrişim katmanlarıdır
#32 her kattaki düğümlerin(nöron) sayısıdır 64 ve 32 olabilir iyi çalışıyor 3 çekirdek boyutu 3x3 filtre matrisine sahip olacağı anlamına gelir
#aktivasyon fonksiyonu relu veya rektifiye doğrusal aktivasyondur sinir ağlarında iyi çalışır
classifier.add(Convolution2D(32, 3,  3, input_shape = (64, 64, 3), activation = 'relu'))
#Pooling
classifier.add(MaxPooling2D(pool_size =(2,2)))

# İkinci convolution layer ekleme
classifier.add(Convolution2D(32, 3,  3, activation = 'relu'))

#verilen pool_size boyutunda kümeler alıp bu kümeler içerisindeki en büyük değerleri kullanarak yeni bir matris oluşturur. 2x2 matris
classifier.add(MaxPooling2D(pool_size =(2,2)))

# Üçüncü convolution(kıvrım,evrişimsel) layer ekleme
#https://medium.com/@ayyucekizrak/derin-%C3%B6%C4%9Frenme-i%C3%A7in-aktivasyon-fonksiyonlar%C4%B1n%C4%B1n-kar%C5%9F%C4%B1la%C5%9Ft%C4%B1r%C4%B1lmas%C4%B1-cee17fd1d9cd
classifier.add(Convolution2D(64, 3,  3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))

#Conv2D katmanları ile yoğun katman arasında bir 'Flatten' katmanı vardır. Flatten, convolution  ve dense katmanlar arasında bir bağlantı görevi görür.
#Flattening
#yapay sinir ağı matris boyutunu kabul etmiyor düzleşmesi lazım
#'Flatten' Genellikle Convolutional bölümün sonuna konan Flatten metodu çok boyutlu olan verimizi tek boyutlu hale getirerek standart yapay sinir ağı için hazır hale getirir.
classifier.add(Flatten())

# Fully Connected Layer
# 'Dense' Bir standart yapay sinir ağı katmanı oluşturur, ilk parametrede verilen sayı kadar nöron barındırır.
# modele bir katman ekliyoruz(gizli katman) katmanımızda her olası sonuç için bir tane olmak üzere 256 nöron olacak
classifier.add(Dense(256, activation = 'relu'))

# Ağ içindeki bazı bağlantıların kaldırılmasıyla  eğitim performansı artacağı varsayılıyor.
# aşırı öğrenmeyi(overfitting) engellemek için bazı nöronları unutmak için kullanılanılır diyebiliriz.
classifier.add(Dropout(0.5))

#çıkış katmanı olarak 10 nöron oluyor çıkış katmanı olarak softmax öneriliyor 0-1 arasında değer gelir 1 e en yakın değeri softmax döndürür
# tahmin ettiğimiz değere daha yakın sonuc olur
# en son katmanda kesinlikle softmax kullanılmalıdır
classifier.add(Dense(10, activation = 'softmax'))

#CNN modelimizi derlemek için kullanılır
#optimizer eğitim boyunca öğrenme hızını ayarlar.
#Kayıp fonksiyonumuz için 'categorical_crossentropy' kullanacağız. Bu, sınıflandırma için en yaygın seçimdir. Düşük puan, modelin daha iyi performans gösterdiğini gösterir.
#İşlerin yorumlanmasını daha da kolaylaştırmak için, modeli eğitirken doğrulama setindeki doğruluk puanını görmek için 'accuracy' metriğini kullanacağız.
classifier.compile(
              optimizer = optimizers.SGD(lr = 0.01), # learning rate öğrenme hızı
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

# Fittting the CNN to the image
from keras.preprocessing.image import ImageDataGenerator
#verileri getirmek için keras kütüphanesinden ImageDataGenerator kullanıyoruz
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
#eğitim setini aktarıyoruz yolunu belirtip size 64x64 olucak batch_size dan 32 er 32 er veriler gelmeye başlıyor
#https://keras.io/preprocessing/image/
training_set = train_datagen.flow_from_directory(
        'mydata/training_set',
        target_size = (64, 64),
        batch_size = 32,
        class_mode = 'categorical')
# test set doğrulama testi, modeliniz tarafından üretilen çıktıyı doğrulamak için kullanılır eğitimli modeli değerlendirir
test_set = test_datagen.flow_from_directory(
        'mydata/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

#Şimdi modelimizi eğiteceğiz. Eğitmek için, modelimizdeki 'fit ()' işlevini şu parametrelerle kullanacağız
#Eğitim tur (Epoch)
#Model eğitilirken verilerin tamamı aynı anda eğitime katılmaz. Belli sayıda parçalar halinde eğitimde yer alırlar. İlk parça eğitilir, modelin başarımı test edilir
#Her dönemin sonunda kaybın ve herhangi bir model metriğin değerlendirileceği veriler. Model bu veriler üzerinde eğitilmeyecektir
#https://keras.io/models/sequential/
model = classifier.fit_generator(
        training_set,
        steps_per_epoch=800,
        epochs=27,
        validation_data = test_set,
        validation_steps = 5000
      )

#Saving the model
classifier.save('Trained_model.h5')

#eğitim almış bir model için, bu aşağıdaki listeyi oluşturabilir:
#['accuracy', 'loss', 'val_accuracy', 'val_loss'] çok önemli değil
print(model.history.keys())








