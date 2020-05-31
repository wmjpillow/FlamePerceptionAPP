#----------------------------------------------------------------------------#
# Imports
#----------------------------------------------------------------------------#

from flask import Flask, render_template, request, redirect, url_for, send_file, Response
# from flask.ext.images import resized_img_src
# from flask.ext.sqlalchemy import SQLAlchemy
import logging
from logging import Formatter, FileHandler
from forms import *
import os
import FlameBoundingbox
from werkzeug.utils import secure_filename
import FireVideoProcessing
import FlameShape
from pathlib import Path

#----------------------------------------------------------------------------#
# App Config.
#----------------------------------------------------------------------------#

app = Flask(__name__)
app.config.from_object('config')
#db = SQLAlchemy(app)

# Automatically tear down SQLAlchemy.
'''
@app.teardown_request
def shutdown_session(exception=None):
    db_session.remove()
'''

# Login required decorator.
'''
def login_required(test):
    @wraps(test)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return test(*args, **kwargs)
        else:
            flash('You need to login first.')
            return redirect(url_for('login'))
    return wrap
'''
#----------------------------------------------------------------------------#
# Controllers.
#----------------------------------------------------------------------------#

# Flask Save Uploads on the server REFEERENCE: 
# https://riptutorial.com/flask/example/19418/save-uploads-on-the-server
# https://stackoverflow.com/questions/19898283/folder-and-files-upload-with-flask

uploads_dir = "./static/Videos"
os.makedirs(uploads_dir, exist_ok=True)
# Share variables in different functions
# https://stackoverflow.com/questions/41636867/how-to-share-variable-between-functions-in-python
parseFile = uploads_dir

@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        # save the single "profile" file
        profile = request.files['profile[]']
        profile.save(os.path.join(uploads_dir, secure_filename(profile.filename)))
        global parseFile
        parseFile = os.path.join(uploads_dir, secure_filename(profile.filename))
        print(parseFile)
        # return redirect(url_for('upload'))
        # return parseFile
    return render_template('pages/upload.html')

@app.route('/Video', methods=['GET','POST'])
def Video():
    return render_template('pages/video.html', passFile= os.path.basename(Path(parseFile)).split('.')[0])

# https://stackoverflow.com/questions/60509538/how-do-i-stream-python-opencv-output-to-html-canvas
@app.route('/BoundingBox', methods=['GET','POST'])
def BoundingBox():
    # print(parseFile)
    return Response(FlameBoundingbox.BoundingBox(parseFile),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/FlameShape', methods=['GET','POST'])
def CallFlameShape():
    return Response(FlameShape.shape(parseFile),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# https://stackoverflow.com/questions/42601478/flask-calling-python-function-on-button-onclick-event
@app.route('/Bounding')
def Bounding():
    # FlameBoundingbox.BoundingBox()
    FireVideoProcessing.FireVideoProcess(parseFile)
    return ("nothing")

@app.route('/')
def home():
    return render_template('pages/placeholder.home.html')

@app.route('/Visualization')
def Visualization():
    return render_template('pages/visualization.html', passFile= os.path.basename(Path(parseFile)).split('.')[0])

# https://stackoverflow.com/questions/24577349/flask-download-a-file
@app.route('/download')
def download():
    name = os.path.basename(Path(parseFile)).split('.')[0]
    filename = name.__add__(".csv")
    downloadfile = os.path.join("./static/data/", filename)
    return send_file(downloadfile , as_attachment=True)

@app.route('/playground')
def playground():
    return render_template('pages/playground.html')

@app.route('/login')
def login():
    form = LoginForm(request.form)
    return render_template('forms/login.html', form=form)


@app.route('/register')
def register():
    form = RegisterForm(request.form)
    return render_template('forms/register.html', form=form)


@app.route('/forgot')
def forgot():
    form = ForgotForm(request.form)
    return render_template('forms/forgot.html', form=form)

# Error handlers.


@app.errorhandler(500)
def internal_error(error):
    #db_session.rollback()
    return render_template('errors/500.html'), 500


@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404

if not app.debug:
    file_handler = FileHandler('error.log')
    file_handler.setFormatter(
        Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
    )
    app.logger.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.info('errors')

#----------------------------------------------------------------------------#
# Launch.
#----------------------------------------------------------------------------#

# Default port:
# if __name__ == '__main__':
#     # FlameBoundingbox.BoundingBox()
#     app.run()


# Or specify port manually:
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=70)

