from flask import Flask, render_template, request, flash, redirect, url_for
import sqlite3
import pickle
import numpy as np

from twilio.rest import Client
account_sid = "AC13b040427ec706a07b2dd9cb2e36a4a1"
auth_token = "83f6ccd3b2eb4d633fa073d9fb9347db"
client = Client(account_sid, auth_token)


import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing.image import img_to_array
import pickle
from flask import Flask, render_template, url_for, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array, array_to_img
from keras.preprocessing import image
import sqlite3
import shutil

app = Flask(__name__)
import pickle
rfc=pickle.load(open("new.pkl","rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return redirect(url_for('predictPage'))

@app.route("/predictPage")
def predictPage():
    from serial_test import Read
    turb, con, ph, temp = Read()
    print(f"pH : {ph} \ntemperature : {temp} \n turbidity: {turb} \n conductivity : {con}")
    return render_template('userlog.html',ph=ph,tempe=temp,turb=turb,con=con)

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':

        connection = sqlite3.connect('database.db')
        cursor = connection.cursor()

        phone = request.form['phone']
        password = request.form['password']

        query = "SELECT * FROM user WHERE mobile = '"+phone+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchone()

        if result:
            return redirect(url_for('predictPage'))
        else:
            return render_template('signin.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')

    return render_template('signin.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':

        connection = sqlite3.connect('database.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('signin.html', msg='Successfully Registered')
    
    return render_template('signup.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    from serial_test import Read
    turb, con, ph, tempe = Read()

    print(f"pH : {ph} \ntemperature : {tempe} \n turbidity: {turb} \n conductivity : {con}")
    if request.method == 'POST':
        Conductivity = request.form['Conductivity']
        Turbidity = request.form['Turbidity']
        Temperature = request.form['Temperature']
        Ph = request.form['Ph']
        
        data = np.array([[Turbidity,Conductivity, Temperature, Ph]])
        my_prediction = rfc.predict(data)
        result = my_prediction[0]
        aa=[]
        print(result)

        tur, cond, temp, p = " ", " ", " ", " "
        if result == 0 :
            res='in Good Condition '
 
        elif result == 1: 
            res='in Bad Condition'
            # client.api.account.messages.create(
            #     to="+91-8431991946",
            #     from_="+12566009618",
            #     body="Water is in Bad Condition")

            if float(Temperature) <= 26:
                temp = "Low Temperature. Ideal temperature range is between 26 to 32 degrees Celsius."
            elif float(Temperature) > 32 and float(Temperature)  < 100:
                temp = "High Temperature. Ideal temperature range is between 26 to 32  degrees Celsius."
            else:
                temp="Ideal Temperature"
            if float(Ph) < 6:
                p = "Low pH. Ideal pH range is between 6 to 9."
            elif float(Ph) > 9 and float(Ph) <= 14:
                p = "High pH. Ideal pH range is between 6 to 9."
            else:
                p="ph Ideal"
            if  float(Turbidity) >= 0 and float(Turbidity) < 1025:
                tur="Water is murky"
            if float(Conductivity) >= 0 and float(Conductivity) < 1025:
                cond="Current is Passing"
            print("Turbidity {} \n  Conductivity {} \n Temperature {} \n Ph {} \n".format(tur,cond,temp,p))
        
        print(res)
        return render_template('userlog.html', status=res,tur=tur,cond=cond,temp=temp,p=p,ph=ph,tempe=tempe,turb=turb,con=con)
    return render_template('userlog.html')

@app.route('/analyse', methods=['GET', 'POST'])
def analyse():
    if request.method == 'POST':

        dirPath = "static/testimage"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['File']
        dst = "static/testimage"
        
        shutil.copy("test/"+fileName, dst)
        image = cv2.imread("test/"+fileName)
        
        #color conversion
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('static/gray.jpg', gray_image)
        #apply the Canny edge detection
        edges = cv2.Canny(image, 250, 254)
        cv2.imwrite('static/edges.jpg', edges)
        #apply thresholding to segment the image
        retval2,threshold2 = cv2.threshold(gray_image,128,255,cv2.THRESH_BINARY)
        cv2.imwrite('static/threshold.jpg', threshold2)
        
        model=load_model('FISHH.h5')
        path='static/testimage/'+fileName

        # Load the class names
        with open('class_names.pkl', 'rb') as f:
            class_names = pickle.load(f)
        dec=""
        dec1=""
        # Function to preprocess the input image
        def preprocess_input_image(path):
            img = load_img(path, target_size=(150,150))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize the image
            return img_array

        # Function to make predictions on a single image
        def predict_single_image(path):
            input_image = preprocess_input_image(path)
            prediction = model.predict(input_image)
            print(prediction)
            predicted_class_index = np.argmax(prediction)
            predicted_class = class_names[predicted_class_index]
            confidence = prediction[0][predicted_class_index]

            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.2%}")
                
            return predicted_class, confidence 

        predicted_class, confidence = predict_single_image(path)
        #predicted_class, confidence = predict_single_image(path, model, class_names)
        print(predicted_class, confidence)
        
        str_label = predicted_class

        accuracy = f"predicted with a confidence of {confidence:.2%}"    


        return render_template('fish.html', status=str_label,accuracy=accuracy, ImageDisplay="http://127.0.0.1:5000/static/testimage/"+fileName,ImageDisplay1="http://127.0.0.1:5000/static/gray.jpg",ImageDisplay2="http://127.0.0.1:5000/static/edges.jpg",ImageDisplay3="http://127.0.0.1:5000/static/threshold.jpg")
    
    return render_template('fish.html')

@app.route('/developer')
def developer():
    return render_template('developer.html')

@app.route('/graph', methods=['GET', 'POST'])
def graph():
    
    images = ['http://127.0.0.1:5000/static/accuracy_plot.png',
              'http://127.0.0.1:5000/static/confusion_matrix.png']
    content=['Accuracy Graph',
             'Confusion Matrix']
        
    return render_template('graph.html',images=images,content=content)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
