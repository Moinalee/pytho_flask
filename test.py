from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_cors import cross_origin,CORS

app = Flask(__name__)
#CORS(app, supports_credentials=True)
socketio = SocketIO(app)

@app.route('/',methods=['POST', 'GET'])
@cross_origin(supports_credentials=True)
def index():
    return render_template('index3.html')

@socketio.on('catch-frame')
def catch_frame(data):

    ## getting the data frames

    ## do some processing 

    ## send it back to client
    emit('response_back', data)  ## ??

@socketio.on('image')
def image(data_image):
    sbuf = StringIO()
    sbuf.write(data_image)

    # decode and convert into image
    b = io.BytesIO(base64.b64decode(data_image))
    pimg = Image.open(b)

    # Process the image frame
    frame = imutils.resize(frame, width=700)
    frame = cv2.flip(frame, 1)
    imgencode = cv2.imencode('.jpg', frame)[1]

    # base64 encode
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpg;base64,'
    stringData = b64_src + stringData

    # emit the frame back
    emit('response_back', stringData)
    print(stringData)

if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1',debug=True)
