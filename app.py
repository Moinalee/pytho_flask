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
#from threading import Thread
from flask_ngrok import run_with_ngrok
from fastai import *
from fastai.vision import *
import eventlet


path = Path(__file__).parent

export_file_name = 'export.pkl'
code=np.array(["0","1","2","3","4","5","6","7","leg","plane"])
#def load_model():
name2id = {v:k for k,v in enumerate(code)}
void_code = name2id['0']

def acc_camvid(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()




def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img



app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
#CORS(app, supports_credentials=True)
run_with_ngrok(app)
global emit_num
socketio = SocketIO(app,cors_allowed_origins=['http://0.0.0.0:3000','https://34.105.3.146:3000'])

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
    image1 = data_image[0]
    count = data_image[1]
    print(count)
    frame = data_uri_to_cv2_img(image1)
    t = pil2tensor(frame,dtype=np.uint8)
    t = t.permute(2,0,1)
    t = t.float()/255. #Convert to float
    im = Image(t)
    #count = 0
    #count += 1
    if(count%30==0):
    #frame1 = np.zeros(frame,np.uint8)
        image2 = learn.predict(im)[1].squeeze()
    ## do some processing 
    #cv2.imwrite('hello.jpg',image2)

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
    #socketio.run(app,host = 'localhost',debug=True,ssl_context=context)
    metrics = acc_camvid

    learn = load_learner(path,export_file_name)
    app.run(debug=True,host='0.0.0.0',port=3000)
    #Thread(target=showing).start()
    #global count
    #count = 0
    #eventlet.wsgi.server(
    #    eventlet.wrap_ssl(eventlet.listen(("0.0.0.0", 3000)),
    #                      certfile='cert.pem',
    #                      keyfile='key.pem',
    #                      server_side=True), app)   

