#opencv görüntü işleme
import cv2
#bilimsel hesaplamaları hızlı bir şekilde yapmamızı sağlayan bir matematik kütüphanesidir. Numpy'ın temelini numpy dizileri oluşturur.
#Numerical Python
import numpy as np
# dosya ile ilgili işlemleri vs. yaptığımız kütüphane
import os

#Hiçbir şey yapmadan geç fonksiyonu hataların önüne geçmek için kullanılır
def nothing(x):
    pass

 #64x64 resim boyutu
image_x, image_y = 64, 64

#dosya oluşturma
def create_folder(folder_name):
    #eğer ./mydata/training_set yolundan çıkmadıysak girilen ismi göre dosya oluşturur
    if not os.path.exists('./mydata/training_set/' + folder_name):
        os.mkdir('./mydata/training_set/' + folder_name)
    # eğer ./mydata/test_set yolundan çıkmadıysak girilen ismi göre dosya oluşturur
    if not os.path.exists('./mydata/test_set/' + folder_name):
        os.mkdir('./mydata/test_set/' + folder_name)
    
        


#resimleri yakalama
def capture_images(ges_name):
    #dosyanın içine yakalanan resimleri koy isim sırasına göre
    create_folder(str(ges_name))
    #kamerayı yakala aç
    cam = cv2.VideoCapture(0)
    #pencerenin ismi
    cv2.namedWindow("test")
    #resim sayısı 0 dan başla
    img_counter = 0
    # resimleri 1 den başlayarak say
    t_counter = 1
    # eğitilen resimleri 1 den başlat
    training_set_image_name = 1
    #test verilerini 1 den başlat
    test_set_image_name = 1
    listImage = [1,2,3,4,5]
    #döngünün pek bi görevi yok hata almamak için yapıldı
    for loop in listImage:

        #pencerenin genişliğini al =>640 pixel
        frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        #pencerenin yüksekliğini al => 480 pixel
        frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        x, y = 0.0, 0.4
        x0 = int(frame_width * x)
        y0 = int(frame_height * y)
        width = 230
        height = 230
        #sonsuz döngüye gir
        while True:
            #kamerayı oku
            ret, frame = cam.read()
            #kameranın simetriyini al
            frame = cv2.flip(frame, 1)


            # frame(çerçeve) içinde dikdörtgen oluştur
            #pencerenin x=0 y=480x0.4 = > 192,genişliği ve yüksekliği 230x230 rengi ve kalınlığı
            img = cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0), 2)

            #region of image dikdörtgen içinde alan oluşturuyoruz
            roi = img[y0:y0 + height, x0:x0 + width]
            #1 lerden oluşan matris oluşturuyoruz
            kernel = np.ones((2, 2), np.uint8)
            #hsv Hue, Saturation ve Value terimleri ile rengi tanımlar.
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # HSV'de cilt rengi aralığını tanımlar

            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)

            # ten rengi görüntü ayıklamak

            mask = cv2.inRange(hsv, lower_skin, upper_skin)

            # içindeki karanlık noktaları doldurmak için eli tahmin et
            mask = cv2.dilate(mask, kernel, iterations=4)


            # resimlerin sayısını gösteren text
            cv2.putText(frame, str(img_counter), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
            #frame i göster
            cv2.imshow("test", frame)
            #mask ı göster
            cv2.imshow("mask", mask)
            # görüntülenen frame sayısı
            k = cv2.waitKey(1)
            # s harfine basınca kaydet
            if k == ord('s'):
                # eğer 350 ve 350 den küçükse training_set e kaydet
                if t_counter <= 350:
                    img_name = "./mydata/training_set/" + str(ges_name) + "/{}.png".format(training_set_image_name)
                    # resmi kaydet boyutlara göre
                    save_img = cv2.resize(mask, (image_x, image_y))
                    # opencv komutu ile resmi yaz
                    cv2.imwrite(img_name, save_img)
                    #yazılan resmi console ekranında göster
                    print("{} written!".format(img_name))
                    # training_set-image 1 artır
                    training_set_image_name += 1

                #eğer 350 den büyük ve 400 e eşit ve küçükse test_set dosyasına yaz
                if t_counter > 350 and t_counter <= 400:
                    img_name = "./mydata/test_set/" + str(ges_name) + "/{}.png".format(test_set_image_name)
                    #resmi kaydet
                    save_img = cv2.resize(mask, (image_x, image_y))
                    #resmi yaz
                    cv2.imwrite(img_name, save_img)
                    # resim ismini console da göster
                    print("{} written!".format(img_name))
                    # ismini 1 artır
                    test_set_image_name += 1
                    # eğer resim sayısı 500 den büyükse çık
                    if test_set_image_name > 500:
                        break

                # resim sayısı artır
                t_counter += 1
                # resim sayısı 401 se 1 yap
                if t_counter == 401:
                    t_counter = 1
                # sonra arttır
                img_counter += 1

            # dikdörtgen alanı aynı konumuna getir
            elif k == ord('q'):
                break
            # dikdörtgen alanı aşağı 20 pixel ilerler
            elif k == ord("2"):
                y0 = min(y0 + 20, frame_height - height)
            # dikdörtgen alanı yukarı 20 pixel ilerler
            elif k == ord("8"):
                y0 = max(y0 - 20, 0)
            # dikdörtgen alanı sola 20 pixel ilerler
            elif k == ord("4"):
                x0 = max(x0 - 20, 0)
            # dikdörtgen alanı sağa 20 pixel ilerler
            elif k == ord("6"):
                x0 = min(x0 + 20, frame_width - width)
        # test resimleri 500 den büyükse çık
        if test_set_image_name > 500:
            break

    # kamerayı serbest bırak
    cam.release()
    # tüm pencereleri kapat
    cv2.destroyAllWindows()

# console ekranından kaydedilcek dosya ismi alma
ges_name = input("Enter gesture name: ")
# gesture_name i capture_image fonksiyonunda çağırıyoruz
capture_images(ges_name)