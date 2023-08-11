import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import tkinter  as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
import tensorflow as tf
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt 
import pathlib 
import random
from pathlib import Path
import imghdr
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import layers
import warnings
from zipfile import ZipFile
from tkinter import*
from tkinter import filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
from mpl_toolkits.axisartist.axislines import Subplot
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
def knnm():
    def browseFiles():
        ftypes = [('CSV',"*.csv"),('All',"*.*")]
        file = filedialog.askopenfilename(filetypes=ftypes)
        d=pd.read_csv(file)
        x=d.iloc[:,2:32]
        y=d.iloc[:,1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)
        knn= KNeighborsClassifier(n_neighbors = 21)
        knn.fit(x_train, y_train)
        pred=knn.predict(x_test)
        acc=metrics.accuracy_score(y_test,pred)*100
        cm=metrics.confusion_matrix(y_test,pred)
        l1 = tk.Label(window,text='Accuracy',width=30,anchor=CENTER,font=("Times","15"))
        l1.place(relx = 0.4,rely = 0.3)
        text = Text(window, width=16, height=1 )
        text.place(relx = 0.5,rely = 0.4)
        text.insert(INSERT, acc)
        l2 = tk.Label(window,text='Confusion Matrix',width=30,anchor=CENTER,font=("Times","15"))
        l2.place(relx = 0.4,rely = 0.5)
        text = Text(window, width=16, height=2 )
        text.place(relx = 0.5,rely = 0.6)
        text.insert(INSERT, cm)
        l3 = tk.Label(window,text='Random predicted values',width=30,anchor=CENTER,font=("Times","15"))
        l3.place(relx = 0.4,rely = 0.7)
        text = Text(window, width=42, height=1)
        text.place(relx=0.4, rely=0.8)
        text.insert(INSERT, pred[:10])

    window = Toplevel(w)
    window.title('KNN Model')
    window.geometry("1000x1000")
    l3 = tk.Label(window,text='BREAST CANCER DETECTION USING KNN ALGORITHM',width=70,font=("Times","20","bold"),anchor=CENTER)
    l3.pack()
    lab1 = Label(window,text='Upload your dataset as csv file',width=70,font=("Times","20"),anchor=CENTER)
    lab1.place(relx = 0.0,rely = 0.1)
    button_explore = Button(window,text = "Browse Files",command = browseFiles,anchor=CENTER,font=("Times","12"))
    button_explore.place(relx = 0.5,rely = 0.2)
def cnnm():
    def browseFiles():
        def graph():
            newWindow1 = Toplevel(window)
            newWindow1.title("Accuracy")
            newWindow1.geometry("1000x1000")
            l1=Label(newWindow1,text='Accuracy Vs Loss',width=70,font=("Times","20"),anchor=CENTER)
            l1.pack()
            fig=plt.figure(figsize=(8,8))
            ax = Subplot(fig, 121)
            fig.add_subplot(ax)
            ax.plot(epochs_range,acc,label='Accuracy')
            ax.plot(epochs_range,val_acc,label="Validation Accuracy")
            ax.set_title("Accuracy")
            ax.legend()
            bx = Subplot(fig, 122)
            fig.add_subplot(bx)
            bx.plot(epochs_range,loss,label='Loss')
            bx.plot(epochs_range,val_loss,label="Validation Loss")
            bx.set_title("Loss")
            bx.legend()
            canvas = FigureCanvasTkAgg(fig,master =  newWindow1)
            canvas.draw()
            canvas.get_tk_widget().pack()
        def image():
            newWindow2 = Toplevel(window)
            newWindow2.title("Prediction")
            newWindow2.geometry("1000x1000")
            l1=Label(newWindow2,text='Random predicted images',width=70,font=("Times","20"),anchor=CENTER)
            l1.pack()
            figim=plt.figure(figsize=(10, 10))
            class_names = val_data.class_names
            result = ' | False'
            for images, labels in val_data.take(1):
                for i in range(9):
                    cx = Subplot(figim, 3,3, i + 1)
                    figim.add_subplot(cx)
                    img = images[i].numpy().astype("uint8")
                    img = tf.expand_dims(img, axis=0)
                    predictions = model.predict(img)
                    print( predictions)
                    predicted_class = np.argmax(predictions)
                    if class_names[predicted_class] == class_names[labels[i]]:
                        result = ' | TRUE'    
                    cx.imshow(images[i].numpy().astype("uint8"))
                    cx.set_title(class_names[predicted_class]+result  )
                    cx.axis("off")
            canvas = FigureCanvasTkAgg(figim,master =  newWindow2)
            canvas.draw()
            canvas.get_tk_widget().pack()
        def check():
            def imgget():
                img_path = filedialog.askopenfilename(initialdir = "/",title = "Select a File",filetypes = (("jpg files","*.jpg*"),("png files","*.png*"),("all files","*.*"))) 
                img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
                img_array = tf.keras.utils.img_to_array(img)
                img_batch = np.expand_dims(img_array, axis=0)
                img_preprocessed = preprocess_input(img_batch)
                img = tf.expand_dims(img, axis=0)
                pred = model.predict(img)
                pred_class = np.argmax(pred)
                res=class_names[pred_class]
                l=Label(newWindow3,text=res,width=70,font=("Times","20","bold"),anchor=CENTER)
                l.place(relx = 0.0,rely = 0.3)
            newWindow3 = Toplevel(window)
            newWindow3.title("Prediction")
            newWindow3.geometry("1000x1000")
            head=Label(newWindow3,text="Upload your testing input",width=70,font=("Times","20","bold"),anchor=CENTER)
            head.place(relx = 0.0,rely = 0.1)
            but = Button(newWindow3,text = "Browse Files",command = imgget,anchor=CENTER,font=("Times","12"))
            but.place(relx = 0.5,rely = 0.2)
    
        filename = filedialog.askopenfilename(initialdir = "/",title = "Select a File",filetypes = (("Zip files","*.zip*"),("all files","*.*")))
        loc=filename
        ploc= pathlib.Path(loc)
        with ZipFile(loc) as zObject:
            zObject.extractall(path=r"C:\Users\sakth\Desktop")
        warnings.filterwarnings('ignore')
        path = r"C:\Users\sakth\Desktop"
        data_dir = pathlib.Path(path)
        pname=ploc.stem
        data_dir =os.path.join(data_dir,pname)
        class_names = np.array(('benign','malignant','normal'))
        image_extensions = [".png", ".jpg"]  # add there all your images file extensions
        img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
        for filepath in Path(data_dir).rglob("*"):
            if filepath.suffix.lower() in image_extensions:
                img_type = imghdr.what(filepath)
                if img_type is None:
                    print(f"{filepath} is not an image")
                elif img_type not in img_type_accepted_by_tf:
                    print(f"{filepath} is a {img_type}, not accepted by TensorFlow")
        batch_size = 32
        img_height = 224
        img_width = 224
        train_data = image_dataset_from_directory(data_dir,validation_split=0.3,subset="training",seed=123,image_size=(img_height, img_width),batch_size=batch_size)
        val_data = image_dataset_from_directory(data_dir,validation_split=0.3,subset="validation",seed=123,image_size=(img_height,img_width),batch_size=batch_size)
        model=tf.keras.Sequential([layers.Conv2D(16, kernel_size=(3, 3),activation='relu',padding = 'Same',input_shape=(img_height, img_width, 3)),
                                 layers.Conv2D(16,kernel_size=(3, 3), activation='relu',padding = 'Same',),
                                 layers.MaxPooling2D(),
                                 layers.Dropout(0.20),
                                 layers.Conv2D(32,kernel_size=(3, 3), activation='relu',padding = 'Same',),
                                 layers.Conv2D(32,kernel_size=(3, 3), activation='relu',padding = 'Same',),
                                 layers.MaxPooling2D(),
                                 layers.Dropout(0.40),
                                 layers.Flatten(),
                                 layers.Dense(128, activation='relu'),
                                 layers.Dense(9,activation="softmax")])
        model.compile(optimizer="Adam",loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=["accuracy"])
        epochs = 2
        history = model.fit(train_data,epochs=epochs,validation_data=val_data,batch_size=batch_size)
        history.history.keys()
        acc = history.history['accuracy']
        val_acc =  history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(epochs)
        model.evaluate(val_data)
        model.summary()
        l=Label(window,text='Model Training done!',width=70,font=("Times","20","bold"),anchor=CENTER)
        l.place(relx = 0.0,rely = 0.3)
        b1=Button(window,text = "Accuracy",command = graph,font=("Times","12"))
        b1.place(relx = 0.3,rely = 0.4)
        b2=Button(window,text = "Prediction",command = image,font=("Times","12"))
        b2.place(relx = 0.5,rely = 0.4)
        b3=Button(window,text = "Test",command = check,font=("Times","12"))
        b3.place(relx = 0.7,rely = 0.4)
    window = Toplevel(w)
    window.title('CNN Model')
    window.geometry("1000x1000")
    lab = Label(window,text='BREAST CANCER DETECTION USING CNN ALGORITHM',width=70,font=("Times","20","bold"))
    lab.place(relx = 0.0,rely = 0.0)
    lab1 = Label(window,text='Upload your dataset as zip file',width=70,font=("Times","20"))
    lab1.place(relx = 0.0,rely = 0.1)
    button_explore = Button(window,text = "Browse Files",command = browseFiles,anchor=CENTER,font=("Times","12"))
    button_explore.place(relx = 0.5,rely = 0.2)  
    
w=Tk()
w.title("Breast Cancer Detection")
w.geometry("1000x1000")
top=Label(w,text='Choose the model you need',width=70,font=("Times","20","bold"))
top.place(relx = 0.0,rely = 0.0)
cnn=Label(w,text='CNN Model',width=70,font=("Times","20"))
cnn.place(relx = 0.0,rely = 0.1)
cnnb=Button(w,text = "Select",command = cnnm,anchor=CENTER,font=("Times","12"))
cnnb.place(relx = 0.5,rely = 0.2)
knn=Label(w,text='KNN Model',width=70,font=("Times","20"))
knn.place(relx = 0.0,rely = 0.3)
knnb=Button(w,text = "Select",command = knnm,anchor=CENTER,font=("Times","12"))
knnb.place(relx = 0.5,rely = 0.4)
w.mainloop()
