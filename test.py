#Mail için gereken kütüphaneler
#Simple Mail Transfer Protocol
#e-posta gönderirken sunucu ile işlemci arasındaki haberleşme protokülüdür.
import smtplib

#mesaj içeriğimizi oluşturan kütüphane
from email.mime.text import MIMEText
#mesaj gövdemizi oluşturmaktadır
from email.mime.multipart import MIMEMultipart
#farklı uzantılardaki dosyaları tanımlamak için kullanılır
from email.mime.base import MIMEBase
#kodlayıcı
from email import encoders

#arayüz tasarımı için gereken kütüphaneler
from tkinter import *
#mesaj
import tkinter.messagebox
#dosya açma
from tkinter import filedialog

#opencv görüntü işleme
import cv2

#Python Imaging Library grafik işleme kütüphanesidir.
from PIL import Image,ImageTk

#rastgele sayı üretme
import random

#Numerical Python
#bilimsel hesaplamaları hızlı bir şekilde yapmamızı sağlayan bir matematik kütüphanesidir. Numpy'ın temelini numpy dizileri oluşturur.
import numpy as np

# keras kütüphanesinden image ve modelimizi yüklemek için kullanılan paketler
from keras.preprocessing import image
from keras.models import load_model

# dosya ile ilgili işlemleri vs. yaptığımız kütüphane
import os

class FingerDetection:
    #self => class içindeki methodların birbiri ile haberleşmesi için kullanılan özel bir değişken yapı
    # self yapısını anlamak için deneme yapmak kod yazmak şart
    # __init__ özel bir yapıdır class kullanıldıgı zaman method cagırmasak bile tetikleniyor
    # diğer kullanılan methodlar classdan sonra tetiklendiğinde çalışıyor
    def __init__(self,master):
        #template başlığı
        master.title('Finger Detection')
        #template boyutu
        master.geometry('500x670')
        #RIDGE(çıkıntı) görünümünde çerceve kalınlıgı 2 olan
        frame = Frame(master, relief=RIDGE, borderwidth=2)
        frame.pack()
        #template arkaplan açık mavi
        frame.config(background='light blue')
        #oluşturulan etiket text editör
        label = Label(frame, text="Finger Detection", fg="navy blue", bg='light blue', font=('Times 35 bold'))
        #yukarı hizala
        label.pack(side=TOP)
        #img değişkenine atılan dosya yolundan seçilen resim
        self.img = ImageTk.PhotoImage(Image.open('tensorflow.png'))
        #label içine resmi koyma
        label = Label(frame,image = self.img)
        #label ı yukarı hizala
        label.pack(side=TOP)

        # MENULER
        # menu nesnesi oluşturma başlatma
        menu = Menu(master)
        # menuyu template üstüne yapılandırıyoruz
        master.config(menu=menu)

        #subm1 adından menu nesnesi oluşturma
        subm1 = Menu(menu)
        #menu baslik
        menu.add_cascade(label="Tools", menu=subm1)
        #menu isimleri command ile hangi fonksiyon çalıştırılacaksa o olucak
        subm1.add_command(label="Open CV Docs",command=self.openHelp)
        subm1.add_command(label="Learn Detect",command = self.learnDetectImage)



        subm2 = Menu(menu)
        menu.add_cascade(label="About", menu=subm2)
        subm2.add_command(label="Finger Detection",command = self.anotherWin)
        subm2.add_command(label="Contributors",command = self.Contributors)

        #İCONLAR İÇİN FOTO
        # Görüntüyü kullanmak için bir photoimage nesnesi oluşturma
        self.photo = PhotoImage(file=r"icon\icons8-camera-100.png")
        # Görüntüyü butona sığacak şekilde yeniden boyutlandırma
        self.photoimage = self.photo.subsample(2, 2)


        self.photo2 = PhotoImage(file=r"icon\icons8-photo-gallery-100.png")

        self.photoimage2 = self.photo2.subsample(2, 2)


        self.photo3 = PhotoImage(file=r"icon\icons8-send-96.png")

        self.photoimage3 = self.photo3.subsample(2, 2)


        self.photo4 = PhotoImage(file=r"icon\logout.png")

        self.photoimage4 = self.photo4.subsample(2, 2)

        #BUTONLAR
        #içeriden padding değeri 5 veriyoruz genişliği 470 arka plan rengi beyaz yazı rengi siyah button groove görünümünde(çizgi)
        # image= dosya yolunda dahil edilen resim
        #compound left sola hizala text=> yazı command => fonksyionun kendisi
        # 1.buton için => place butonun konumu x ekseninde 5 y ekseninde 75
        but1 = Button(frame, padx=5, pady=5, width=470, bg='white', fg='black', relief=GROOVE, image=self.photoimage,
                       compound=LEFT, text='Open Cam & Detect',command = self.camDetect, font=('helvetica 15 bold'))
        but1.place(x=5, y=75)

        but2 = Button(frame, padx=5, pady=5, width=470, bg='white', fg='black', image=self.photoimage2,
                      compound=LEFT, relief=GROOVE, text='Open Image', command = self.openImage, font=('helvetica 15 bold'))
        but2.place(x=5, y=150)

        but4 = Button(frame, padx=5, pady=5, width=470, bg='white', fg='black', relief=GROOVE, image=self.photoimage3,
                      compound=LEFT, text='Send E-Mail', command = self.sendEmail,font=('helvetica 15 bold'))
        but4.place(x=5, y=525)

        but5 = Button(frame, padx=5, pady=5, width=470, bg='white', fg='black', relief=GROOVE, image=self.photoimage4,
                      compound=LEFT, text='EXIT', command = exit,font=('helvetica 15 bold'))
        but5.place(x=5, y=600)


        #locad_model nesnesi ile modelimizi yüklüyoruz
        self.classifier = load_model('Trained_model.h5')
        #resmi kaydediceğimiz yer
        self.SAVE_PATH = "img/"
        #yazı tipi fontu
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        #64x64 resim boyutu
        self.image_x, self.image_y = 64, 64
        self.cam = cv2.VideoCapture(0)

    #belirleyici fonksiyon tahmin edici
    def predictor(self):
        # 64x64 lük 1.png lik resmi load_img ile yüklüyoruz
        test_image = image.load_img('1.png', target_size=(64, 64))
        # resmi numpy dizisine dönüştürüyor
        test_image = image.img_to_array(test_image)
        #yeni bir eksen ekleyerek diziyi genişletir image.array[1,2] image.shape => 2 genişletilmiş dizisi [1,2]
        test_image = np.expand_dims(test_image, axis=0)

        # sınıflandırıcıya yani modelimize tahmin etmesi için gönderiyor
        result = self.classifier.predict(test_image)
        #eğer indexi 0 olan ilk klasör varsa doğruysa 1=true değer 0 = false değer
        #0 değerini döndür
        if result[0][0] == 1:
            return '0'
        elif result[0][1] == 1:
            return '1'
        elif result[0][2] == 1:
            return '2'
        elif result[0][3] == 1:
            return '3'
        elif result[0][4] == 1:
            return '4'
        elif result[0][5] == 1:
            return '5'
        elif result[0][6] == 1:
            return '6'
        elif result[0][7] == 1:
            return '7'
        elif result[0][8] == 1:
            return '8'
        elif result[0][9] == 1:
            return '9'


    #kamera tespiti
    def camDetect(self):
        #pencerenin genişlik değerini alma
        # ön kamera yakalama

        frame_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        #pecerenin yükseklik değerini alma
        frame_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        x, y = 0.0, 0.4
        x0 = int(frame_width * x)
        y0 = int(frame_height * y)
        width = 230
        height = 230

        # ilk başta img_text boş başlar
        img_text = ''
        while True:
            # kamerayı okuma
            ret, frame = self.cam.read()
            # kamera görüntüsünün simetriyi alma
            frame = cv2.flip(frame, 1)
            # frame üzerine dikdörtgen
            img = cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0), 2)
            # region of image alan oluşturma
            roi = img[y0:y0 + height, x0:x0 + width]
            # 1 ler matrisinde oluşan çekirdek
            kernel = np.ones((2, 2), np.uint8)
            # roi içine hsv
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # HSV'de cilt rengi aralığını tanımlar
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)

            # ten rengi görüntü ayıklamak

            mask = cv2.inRange(hsv, lower_skin, upper_skin)

            # içindeki karanlık noktaları doldurmak
            mask = cv2.dilate(mask, kernel, iterations=4)

            # görüntüyü bulanıklaştır
            mask = cv2.GaussianBlur(mask, (5, 5), 100)

            # frame üzerine yazı imag_text den gelen değer
            cv2.putText(frame, img_text, (22, 34), self.font, 1, (200, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "s: Save;  q: Quit", (22, 470), self.font, 1, (200, 255, 255), 2, cv2.LINE_AA)
            # alanı göster
            cv2.imshow("roi", roi)
            # frame göster
            cv2.imshow("test", frame)
            # maskı göster
            cv2.imshow("mask", mask)

            # 1.png olarak 64x64 resim kaydet
            img_name = "1.png"
            save_img = cv2.resize(mask, (self.image_x, self.image_y))
            #resmi yaz
            cv2.imwrite(img_name, save_img)
            # resmi 1.png olarak console da göster
            print("{} written!".format(img_name))
            # predictor fonksiyonunu çalıştırıp img_text içine aktarma
            img_text = self.predictor()
            # frame per second oynatıcalacak frame sayısı
            k = cv2.waitKey(1)
            # q harfiyle çık
            if k == ord('q'):
                break
            # random bir şekilde tanımladığımız sayilardan resmi ismini oluşturduk
            elif k == ord('s'):
                # 1 ile 10000000 arası tamsayı üretir
                a = random.randint(1, 10000000)
                b = random.randint(1, 10000000)
                c = random.randint(1, 10000000)
                d = random.randint(1, 10000000)
                e = random.randint(1, 10000000)
                f = random.randint(1, 10000000)
                g = a + b + c + d + e + f
                #toplanan sayilardan resmin ismi oluştu
                img_name = "data{}.png".format(g)#resmin boyutunu ayarladık

                save_img = cv2.resize(frame, (470, 300))
                # resmi belirtilen yola yazdık
                cv2.imwrite(os.path.join(self.SAVE_PATH, img_name), save_img)
            # dikdörtgen alan 20 pixel aşağı ilerler
            elif k == ord("2"):
                y0 = min(y0 + 20, frame_height - height)
            # dikdörtgen alan 20 pixel yukarı ilerler
            elif k == ord("8"):
                y0 = max(y0 - 20, 0)
            # dikdörtgen alan 20 pixel sola ilerler
            elif k == ord("4"):
                x0 = max(x0 - 20, 0)
            # dikdörtgen alan 20 pixel sağa ilerler
            elif k == ord("6"):
                x0 = min(x0 + 20, frame_width - width)

        #kamerayı serbest bırak
        self.cam.release()
        #bütün penceleri kapa
        cv2.destroyAllWindows()

    #Hiçbir şey yapmadan geç fonksiyonu hataların önüne geçmek için kullanılır
    def nothing(x):
        pass

    # opencv help version and file
    def openHelp(self):
        help(cv2)

    # resim açma
    def learnDetectImage(self):
        # Görüntü dosyasını açmak için kullanılır
        im = Image.open("language.jpg")

        # Herhangi bir resim görüntüleyicide görüntüyü gösterir.
        im.show()

    # Projede kullanılan kütüphaneler
    def anotherWin(self):
        tkinter.messagebox.showinfo("About",
                                    'Finger Detection\n Made Using\n-OpenCV\n-Numpy\n-Tkinter\n In Python 3')
    #Projenin yazarları
    def Contributors(self):
        tkinter.messagebox.showinfo("Contributors", "\n 1. Bülent Akyüz\n 2. M.ASTAM \n 3. Güray Emiroğulları \n 4.Oğuzhan Güllü \n")

    #Çıkış
    def exitt(self):
        exit()

    #Resimlerin kaydedildiği yeri açar
    def openImage(self):
        #global değişkenlere programın her yerinden ulaşılır
        global my_image
        #resmi seçiceğimiz dosyaların bulundugu yeri açar
        root.filename = filedialog.askopenfilename(initialdir="/img", title="Select A File",
                                                   filetypes=(("all files", "*.*"), ("jpg files", "*.jpg")))
        #dosya yolundaki seçilen resmi değişken içine atıyor
        my_image = ImageTk.PhotoImage(Image.open(root.filename))
        #seçilen resmi label üzerine yapıştırır
        my_image_label = Label(image=my_image, padx=5, pady=5).place(x=12.5, y=220).pack()

    #mail göndericeğimiz fonksiyon
    def sendEmail(self):
        #kullanıcı email adresi
        email_user = 'hakan.sandal8@gmail.com'
        # kullanıcı şifresi
        email_password = '2180656906'
        # alıcı email adresi
        email_send = 'hakan.sandal8@gmail.com'
        #mail konusu
        subject = 'Python İmage!'
        # mail gövde nesnesi oluştu
        msg = MIMEMultipart()
        #mail kimden
        msg['From'] = email_user
        # kime yani alıcı
        msg['To'] = email_send
        # mail konusu
        msg['Subject'] = subject
        # mail gövde yazısı içerik yazısı
        body = 'Hi there, sending this email from Python!'
        # bodyi sade bir şekilde ekliyoruz
        msg.attach(MIMEText(body, 'plain'))
        #resmi yoldan alıyoruz
        attachment = open(root.filename, 'rb')
        #dosya uzantısı bin farklı uzantılardaki dosyaları tanımlamak için kullanılır
        part = MIMEBase('application', 'octet-stream')
        # attachment set ediyoruz okuyoruz
        part.set_payload((attachment).read())
        #kodlama işlemi
        encoders.encode_base64(part)
        #içerik eğilimi
        part.add_header('Content-Disposition', "attachment; filename= " + root.filename)
        # msg ile iliştirme yapıyoruz
        msg.attach(part)
        # msg iliştirilen değişkeni string değere yani text olarak dönüştürür
        text = msg.as_string()
        #sunucu adresini ve portunu giriyoruz
        server = smtplib.SMTP('smtp.gmail.com', 587)
        #güvenli olmayan bir bağlantıyı güvenli olana dönüştürmek ister
        server.starttls()
        # sunucuya giriş yapıyoruz
        server.login(email_user, email_password)
        # sunucuya mail gönderme işlemi
        server.sendmail(email_user, email_send, text)
        # gönderilme başarılı ise mesaj göster
        tkinter.messagebox.showinfo("Email About", "E-mail Successful")
        #sunucudan çık
        server.quit()

# pencere için kök nesnesi oluştu
root = Tk()
# class değerine kök değişkenini veriyoruz
s = FingerDetection(root)
# pencereden çarpıya basınca çık methodu durdur
root.mainloop()


