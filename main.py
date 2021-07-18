import numpy as np
from flask import Flask, jsonify, request
import os
import tensorflow as tf
from werkzeug.utils import secure_filename
from keras.preprocessing import image

app = Flask(__name__)
UPLOAD_FOLDER = './Uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def tell(classes):
    if classes[0][2] == 1:
        return 'Its a Scissors'
        # print('Its a Scissors')
    if classes[0][1] == 1:
        return 'Its a Rock'
        # print('Its a Rock')
    if classes[0][0] == 1:
        return 'Its a Paper'
        # print('Its a Paper')


@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        # f = request.files['file']
        # f.save(secure_filename(f.filename))

        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        if model != None:
            temp_image = image.load_img(os.path.join(app.config['UPLOAD_FOLDER'], filename),
                                        target_size=(150, 150))
            x = image.img_to_array(temp_image)
            x = np.expand_dims(x, axis=0)

            # x=preprocess_input(x)   # preprocessing with vgg16
            images = np.vstack([x])
            classes = model.predict(images, batch_size=10)
            return jsonify({'response': tell(classes)})

    return jsonify({'response': 'Failed'})


if __name__ == '__main__':

    global model
    model = tf.keras.models.load_model('model.h5')
    # app.run(host='127.0.0.1', port=105)
    app.run(debug=False)
