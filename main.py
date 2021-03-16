from flask import Flask, render_template, url_for, redirect, request, flash, send_from_directory
import tensorflow as tf
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
plt.switch_backend('Qt4Agg')
from tensorflow.keras.models import load_model
import numpy as np
import os
from keras_preprocessing import image

UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


app = Flask(__name__)
app.config['SECRET_KEY'] = '2008de4bbf105d61f26a763f8'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
mod = load_model('beans.model')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# scale
def scale(image):
    image = tf.cast(image, tf.float16)
    image /=2555.0
    return tf.image.resize(image, [224, 224])


# image preprocessing and prediction.
def predict(image):
    img = plt.imread(image)
    image_scaled = scale(img)
    image_scaled = np.expand_dims(image_scaled, axis=0)
    predict = mod.predict(image_scaled)
    names = ['Angular-Leaf-Spot', 'Bean Rust', 'Healthy']
    prediction = names[np.argmax(predict)]
    return prediction


@app.route('/')
def upload_form():
    return render_template('main.html')


@app.route('/', methods=['POST', 'GET'])
def main():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Image successfully uploaded!', 'success')
        image = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        prediction = predict(image)
        return render_template('main.html', filename=filename, prediction=prediction)
    else:
        flash('Allowed image types are: png, jpg, jpeg', 'danger')
        return redirect(request.url)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == '__main__':
    app.run(debug=True)
