from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_cors import cross_origin,CORS
import ssl
from io import StringIO,BytesIO
import base64
import cv2
import imutils
from PIL import Image as PIL_Image #renaming to prevent clashes
import numpy as np
from threading import Thread
from flask_ngrok import run_with_ngrok
from fastai import *
from fastai.vision import *
import eventlet
from torchvision.utils import save_image

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
    with open('image.jpeg', 'wb') as image_file:
      image_file.write(base64.b64decode(encoded_data))
      return nparr,image_file


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'


global emit_num
socketio = SocketIO(app,cors_allowed_origins=['https://0.0.0.0:3000','https://35.197.59.60:3000'])

@app.route('/',methods=['POST', 'GET'])
@cross_origin(supports_credentials=True)
def index():
    return render_template('index.html')

@socketio.on('catch-frame')
def catch_frame(data):

    emit('response_back', data)  ## ??

@socketio.on('image')
def image(data_image):
    #print("starting",data_image,"ending_data_image")

    # decode and convert into image
    image1 = data_image[0]   # base64 string
    count = data_image[1]   # int count
    print(count)


    encoded_data = image1.split(',')[1]
    img_data = base64.b64decode(str(encoded_data))
    image_pil = PIL_Image.open(io.BytesIO(img_data))
    image_cv2 = cv2.cvtColor(np.array(image_pil),cv2.COLOR_BGR2RGB)
    cv2.imwrite("cv2.jpg",image_cv2)
    t = pil2tensor(image_cv2, dtype=np.float32).div_(255)

    if(count%30==0):
      t = Image(t)
      t.save("t.jpg")
      pred_tensor = learn.predict(t)[1].squeeze() #torch tensor
      plt.imshow(pred_tensor)
      plt.show()

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

    #learn = load_learner(path, export_file_name)
    learn = load_learner("./")
    #app.run(host='0.0.0.0',port=3000)
    #Thread(target=showing).start()
    #global count
    #count = 0
    eventlet.wsgi.server(
        eventlet.wrap_ssl(eventlet.listen(("0.0.0.0", 3000)),
                          certfile='cert.pem',
                          keyfile='key.pem',
                          server_side=True), app)

