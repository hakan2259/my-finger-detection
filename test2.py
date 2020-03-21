#Mail için gereken kütüphaneler
import smtplib

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

#arayüz tasarımı için gereken kütüphaneler
from tkinter import *
import tkinter.messagebox
from tkinter import filedialog

#opencv
import cv2

#resim
from PIL import Image,ImageTk

#rastgele
import random
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import os

class FingerDetection:
    def __init__(self,master):
        master.title('Finger Detection')
        master.geometry('500x670')
        frame = Frame(master, relief=RIDGE, borderwidth=2)
        frame.pack(fill=BOTH, expand=1)
        frame.config(background='light blue')
        label = Label(frame, text="Finger Detection", fg="navy blue", bg='light blue', font=('Times 35 bold'))
        label.pack(side=TOP)
        self.img = ImageTk.PhotoImage(Image.open('tensorflow.png'))
        label = Label(frame,image = self.img)
        label.pack(side=TOP)

        # MENULER
        menu = Menu(master)
        root.config(menu=menu)

        subm1 = Menu(menu)
        menu.add_cascade(label="Tools", menu=subm1)
        subm1.add_command(label="Open CV Docs",command=self.openHelp)
        subm1.add_command(label="Learn Detect",command = self.learnDetectImage)

        subm2 = Menu(menu)
        menu.add_cascade(label="About", menu=subm2)
        subm2.add_command(label="Finger Detection",command = self.anotherWin)
        subm2.add_command(label="Contributors",command = self.Contributors)

        #İCONLAR İÇİN FOTO
        # Creating a photoimage object to use image
        self.photo = PhotoImage(file=r"icon\icons8-camera-100.png")
        # Resizing image to fit on button
        self.photoimage = self.photo.subsample(2, 2)

        # Creating a photoimage object to use image
        self.photo2 = PhotoImage(file=r"icon\icons8-photo-gallery-100.png")
        # Resizing image to fit on button
        self.photoimage2 = self.photo2.subsample(2, 2)

        # Creating a photoimage object to use image
        self.photo3 = PhotoImage(file=r"icon\icons8-send-96.png")
        # Resizing image to fit on button
        self.photoimage3 = self.photo3.subsample(2, 2)

        # Creating a photoimage object to use image
        self.photo4 = PhotoImage(file=r"icon\logout.png")
        # Resizing image to fit on button
        self.photoimage4 = self.photo4.subsample(2, 2)

        #BUTONLAR

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


        self.cam = cv2.VideoCapture(0)
        self.classifier = load_model('Trained_model.h5')
        self.SAVE_PATH = "img/"
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.image_x, self.image_y = 64, 64





    def predictor(self):

        test_image = image.load_img('1.png', target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = self.classifier.predict(test_image)
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



    def camDetect(self):

        frame_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        x, y = 0.0, 0.4
        x0 = int(frame_width * x)
        y0 = int(frame_height * y)
        width = 230
        height = 230

        img_text = ''
        while True:
            ret, frame = self.cam.read()
            frame = cv2.flip(frame, 1)

            img = cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0), 2)

            roi = img[y0:y0 + height, x0:x0 + width]

            kernel = np.ones((2, 2), np.uint8)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # define range of skin color in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)

            # extract skin colur imagw

            mask = cv2.inRange(hsv, lower_skin, upper_skin)

            # extrapolate the hand to fill dark spots within
            mask = cv2.dilate(mask, kernel, iterations=4)

            # blur the image
            mask = cv2.GaussianBlur(mask, (5, 5), 100)

            cv2.putText(frame, img_text, (22, 34), self.font, 1, (200, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "s: Save;  q: Quit", (22, 470), self.font, 1, (200, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("roi", roi)
            cv2.imshow("test", frame)
            cv2.imshow("mask", mask)


            img_name = "1.png"
            save_img = cv2.resize(mask, (self.image_x, self.image_y))
            cv2.imwrite(img_name, save_img)
            print("{} written!".format(img_name))
            img_text = self.predictor()

            k = cv2.waitKey(1)
            if k == ord('q'):
                break
            elif k == ord('s'):
                a = random.randint(1, 10000000)
                b = random.randint(1, 10000000)
                c = random.randint(1, 10000000)
                d = random.randint(1, 10000000)
                e = random.randint(1, 10000000)
                f = random.randint(1, 10000000)
                g = a + b + c + d + e + f
                img_name = "data{}.png".format(g)
                save_img = cv2.resize(frame, (470, 300))
                cv2.imwrite(os.path.join(self.SAVE_PATH, img_name), save_img)
            elif k == ord("2"):
                y0 = min(y0 + 20, frame_height - height)
            elif k == ord("8"):
                y0 = max(y0 - 20, 0)
            elif k == ord("4"):
                x0 = max(x0 - 20, 0)
            elif k == ord("6"):
                x0 = min(x0 + 20, frame_width - width)

        self.cam.release()
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
        tkinter.messagebox.showinfo("Contributors", "\n1. Bülent Akyüz\n2. M.ASTAM \n3. Güray Emiroğulları \n")

    #Çıkış
    def exitt(self):
        exit()

    #Resimlerin kaydedildiği yeri açar
    def openImage(self):
        global my_image
        root.filename = filedialog.askopenfilename(initialdir="/img", title="Select A File",
                                                   filetypes=(("all files", "*.*"), ("jpg files", "*.jpg")))

        my_image = ImageTk.PhotoImage(Image.open(root.filename))

        my_image_label = Label(image=my_image, padx=5, pady=5).place(x=12.5, y=220).pack()

    def sendEmail(self):
        email_user = 'hakan.sandal8@gmail.com'
        email_password = '2180656906'
        email_send = 'hakan.sandal8@gmail.com'

        subject = 'Python İmage!'
        msg = MIMEMultipart()
        msg['From'] = email_user
        msg['To'] = email_send
        msg['Subject'] = subject

        body = 'Hi there, sending this email from Python!'
        msg.attach(MIMEText(body, 'plain'))
        attachment = open(root.filename, 'rb')

        part = MIMEBase('application', 'octet-stream')
        part.set_payload((attachment).read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', "attachment; filename= " + root.filename)

        msg.attach(part)
        text = msg.as_string()
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(email_user, email_password)

        server.sendmail(email_user, email_send, text)
        tkinter.messagebox.showinfo("Email About", "E-mail Successful")

        server.quit()

root = Tk()
s = FingerDetection(root)
root.mainloop()


