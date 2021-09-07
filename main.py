from flask import Flask, render_template, redirect, url_for, request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from werkzeug.utils import secure_filename
from wtforms import SubmitField
import matplotlib.pyplot as plt
import cv2
from collections import Counter
from sklearn.cluster import KMeans
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(18)
app.config["IMAGE_UPLOADS"] = "C:/Users/nonye/PycharmProjects/imagecolorpalletegen_capstonr"
Bootstrap(app)


class UploadForm(FlaskForm):
    image = FileField(validators=[FileRequired()])
    submit = SubmitField('Submit')


def rgb_to_hex(rgb_color):
    # Converting the colors to hex colors thus returning the hex value instead of the color name
    hex_color = "#"
    for i in rgb_color:
        i = int(i)
        hex_color += ("{:02x}".format(i))
    return hex_color


# Function tp prep the image
def prep_image(raw_img):
    modified_img = cv2.resize(raw_img, (900, 600), interpolation=cv2.INTER_AREA)
    modified_img = modified_img.reshape(modified_img.shape[0]*modified_img.shape[1], 3)
    return modified_img


def color_analysis(img):
    # Using k-means to cluster the top 10 colors
    clf = KMeans(n_clusters=10)
    color_labels = clf.fit_predict(img)
    center_colors = clf.cluster_centers_
    # Creating a dictionary to store the colors and their volume
    counts = Counter(color_labels)
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys()]
    # assigning hex values to colors
    plt.figure(figsize=(12, 8))
    plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)
    plt.savefig("color_analysis_report.png")
    return hex_colors


@app.route('/', methods=['GET', 'POST'])
def home():
    form = UploadForm()
    if form.validate_on_submit():
        image = form.image.data
        filename = secure_filename(image.filename)
        image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        prepped_image = prep_image(image)
        result = color_analysis(prepped_image)
        return redirect(url_for('home', hex_values=result))
    return render_template("index.html", form=form)


if __name__ == "__main__":
    app.run(debug=True)
