import io
import numpy as np
import tensorflow as tf

from flask import Flask, redirect, request
from flask_cors import CORS
from PIL import Image

FRONTEND_ORIGIN='http://localhost:5001/'

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origin": [FRONTEND_ORIGIN]}})

model = tf.keras.models.load_model('model/convnet1.h5')

@app.get('/')
def root():
    return redirect('/static/index.html')

@app.post('/api/predict')
def generate_message():
    # image loading
    data = request.files['file'].read()
    stream = io.BytesIO(data)
    image = Image.open(stream).resize((28, 28))

    # transform
    # NOTE: image[:,:,3] is the alpha channel
    x = tf.constant(np.array(image)[:,:,3], dtype=tf.float32, shape=(1, 28, 28, 1))
    y = np.argmax(model.predict(x, verbose=0))
    return {'prediction': int(y)}

if __name__ == '__main__':
    app.run(host='localhost', port=5000)