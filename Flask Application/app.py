import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import DenseNet121

# ✅ Use DenseNet121
base_model = DenseNet121(include_top=False, input_shape=(224, 224, 3), weights=None)

x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# ✅ Load your trained DenseNet weights
model.load_weights(r'C:\Users\sudee\OneDrive\Desktop\Pneumonia Detection\pneumonia\densenet_model_01.h5')

app = Flask(__name__)
print('Model loaded. Check http://127.0.0.1:5000/')

def get_className(classNo):
    return "Normal" if classNo == 0 else "Pneumonia"

def getResult(img_path):
    image = cv2.imread(img_path)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((224, 224))
    image = np.array(image) / 255.0  # Normalize
    input_img = np.expand_dims(image, axis=0)

    preds = model.predict(input_img)
    class_idx = np.argmax(preds, axis=1)[0]
    severity = float(preds[0][class_idx]) * 100  # Severity confidence %
    return class_idx, severity

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        class_idx, severity = getResult(file_path)
        result = get_className(class_idx)
        return f"{result} - Severity: {severity:.2f}%"

if __name__ == '__main__':
    app.run(debug=True)
