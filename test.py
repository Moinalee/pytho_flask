from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_cors import cross_origin,CORS
import ssl
from io import StringIO,BytesIO
import base64
import cv2
import imutils
from PIL import Image
import numpy as np
from threading import Thread
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
#CORS(app, supports_credentials=True)
run_with_ngrok(app)
socketio = SocketIO(app)

@app.route('/',methods=['POST', 'GET'])
@cross_origin(supports_credentials=True)
def index():
    return render_template('index.html')

@socketio.on('catch-frame')
def catch_frame(data):

    ## getting the data frames

    ## do some processing 

    ## send it back to client
    emit('response_back', data)  ## ??

@socketio.on('image')
def image(data_image):
    #print("starting",data_image,"ending_data_image")


    # decode and convert into image
    #print(type(data_image))
    encoded_data = data_image.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #print(type(frame))
    #frame1 = np.zeros(frame,np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('Original image',frame1)
    #cv2.waitKey(5)
    #cv2.imshow('Gray image', gray)
    #print("just",frame,"now",gray)
    #cv2.imwrite('color_img.jpg', gray)
    #showing()

    # Process the image frame
    #frame = imutils.resize(frame, width=200)
    gray = cv2.flip(gray, 1)
    imgencode = cv2.imencode('.jpg', gray)[1]

    # base64 encode
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpg;base64,'
    stringData = b64_src + stringData

    # emit the frame back
    emit('response_back', stringData)
    #print(stringData)
def showing():
        just = cv2.imread('/color_img.jpg')
        cv2.imshow('Gray',just)
        cv2.waitKey(10)
        print("here")
if __name__ == '__main__':
    #context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    #context.load_cert_chain('server.crt', 'server.key')
    #socketio.run(app,host = '0.0.0.0',debug=True)#,ssl_context=context)
    app.run()
    #Thread(target=showing).start()
