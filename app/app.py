import io
import torch
import torch.nn.functional as F
import numpy as np

from flask import Flask, redirect, request
from flask_cors import CORS
from model import Shallow300
from PIL import Image

FRONTEND_ORIGIN='http://localhost:5001/'

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origin": [FRONTEND_ORIGIN]}})

model = Shallow300()
model.load_state_dict(torch.load('weight.pth'))
model.eval()

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
    x = torch.tensor(np.array(image), dtype=torch.float32)
    x = x[:, :, -1] / 255
    x = x.reshape((1, -1))
    
    # inference
    with torch.no_grad():
        y = F.softmax(model(x), dim=1)
        prediction = y.argmax(dim=1)[0].item()

    return {'prediction': prediction}

if __name__ == '__main__':
    app.run(host='localhost', port=5000)