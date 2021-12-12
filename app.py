from flask import Flask, request, render_template
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from app_utils import predict_model
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

MODEL_PATH = 'model-1.h5'
model = load_model(MODEL_PATH)


@app.route('/', methods=['GET'])
def index():
    # Main Page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method=='POST':
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename)
        )
        f.save(file_path)

        return predict_model(file_path, model)
    return None


if __name__ == '__main__':
    app.run(debug=True)