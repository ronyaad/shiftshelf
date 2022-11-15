from flask import Flask
import matplotlib.pyplot as plt
from detecto import  utils
from detecto.core import Model
import os
import threading
import helpers

import os,shutil
from flask import Flask, flash, render_template, send_from_directory,url_for
from flask_uploads import IMAGES, UploadSet, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
from pyngrok import ngrok
ngrok.set_auth_token("2FlltFrUyENOhEpY1jFToeGlFHV_4Uu5VudhuTWcDtExqY5Zx")
os.environ["FLASK_ENV"] = "development"

app = Flask(__name__)
port = 5000
public_url = ngrok.connect(port).public_url
print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, port))
# Update any base URLs to use the public ngrok URL
app.config["BASE_URL"] = public_url
app.config["SECRET_KEY"] = os.urandom(24)
app.config["UPLOADED_PHOTOS_DEST"] = "uploads"

photos = UploadSet("photos", IMAGES)
configure_uploads(app, photos)

class UploadForm(FlaskForm):
    folder = app.config["UPLOADED_PHOTOS_DEST"]
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    photo = FileField(validators=[FileAllowed(photos, 'Only images are allowed'),FileRequired('File field should not be empty')])
    submit = SubmitField('Upload')

class ObjectDetectionModel():
    def __init__(self):
        self.labels = ['Nescafe-Glass-Red',
            'Nescafe-Bag-2in1',
            'Nescafe-Bag-Cappuccino',
            'Nescafe-Glass-GoldDecaf',
            'Nescafe-Glass-Yellow',
            'Tchibo-Glass-Gold',
            'Nestle-Plastic-CoffeeMate',
            'Nestle-Bag-CoffeeMate']
    def getModel(self):
        return Model.load('detecto.pth', self.labels)

detector_model = ObjectDetectionModel().getModel()

def get_plan(filename):
    planogram = [['Nescafe-Bag-2in1', 'Nescafe-Bag-2in1'],
                ['Nescafe-Bag-Cappuccino', 'Nescafe-Glass-Yellow'],
                ['Nescafe-Glass-GoldDecaf',
                'Nestle-Bag-CoffeeMate',
                'Nestle-Plastic-CoffeeMate']]
    obj = ObjectDetectionModel()
    return helpers.getDifferenceGrid(detector_model, utils.read_image(app.config["UPLOADED_PHOTOS_DEST"] + "/" +filename), planogram)
   
    
@app.route('/uploads/<filename>')
def get_file(filename):
    if os.path.exists("result.png"):
        os.remove("result.png")
    images= []
    image = utils.read_image(app.config["UPLOADED_PHOTOS_DEST"] + "/" +filename)
    images.append(image)
    obj = ObjectDetectionModel()
    helpers.plot_prediction_grid_with_ImageSave(detector_model,images, figsize=(16, 12))
    return send_from_directory('','result.png')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = url_for('get_file', filename= filename)
        plan = get_plan(filename)
    else:
        file_url = None
        plan = None
    return render_template('upload.html', form=form, file_url=file_url, plan=plan)

# Start the Flask server in a new thread
threading.Thread(target=app.run, kwargs={"use_reloader": False}).start()