#!/usr/bin/env python

from flask import Flask
from flask import request
from flask import render_template

import label_image as label
import json
import os
import logging
import datetime
import werkzeug

# See https://github.com/BVLC/caffe/blob/master/examples/web_demo/app.py for more info
UPLOAD_FOLDER = '/tmp/inception_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['jpg', 'jpeg'])

app = Flask(__name__)

# @app.route('/')
# def root():
# 	return render_template('index.html')

@app.route('/api/classify_file')
def classify_file():
	path = request.args.get('path')
	result = label.classify(path)
	return json.dumps( result )

@app.route('/api/classify_upload', methods=['POST'])
def classify_upload():
	# try:
		# We will save the file to disk for possible data collection.
	imagefile = request.files['file']
	filename_ = str(datetime.datetime.now()).replace(' ', '_') + werkzeug.secure_filename(imagefile.filename)
	filename = os.path.join(UPLOAD_FOLDER, filename_)
	imagefile.save(filename)
	logging.info('Saving to %s.', filename)
	result = label.classify(filename)
	return json.dumps( result )
	# except Exception as err:
	# 	print err
	# 	return "Upload classify failed", 404

@app.route('/<path:path>')
def static_file(path):
    return app.send_static_file(path)

def allowed_file(filename):
	return (
		'.' in filename and
		filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
	)

if __name__ == "__main__":
	# Ensure upload folder exists
	if not os.path.exists(UPLOAD_FOLDER):
		os.makedirs(UPLOAD_FOLDER)

	app.run(debug=True, host='0.0.0.0')