#!/usr/bin/env python

from flask import Flask
from flask import request
from flask import render_template

import label_image as label
import resnet50
import json
import os
import logging
import datetime
import werkzeug
import sys

# See https://github.com/BVLC/caffe/blob/master/examples/web_demo/app.py for more info
UPLOAD_FOLDER = '/tmp/inception_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['jpg', 'jpeg'])

app = Flask(__name__)

# @app.route('/')
# def root():
# 	return render_template('index.html')

@app.route('/api/ping')
def ping():
	return 'pong'

@app.route('/api/classify_file')
def classify_file():
	path = request.args.get('path')
	result = label.classify(path)
	return json.dumps( result )

@app.route('/api/classify_upload', methods=['POST'])
def classify_upload():
	try:
		print 'begin classify'
		print request.files
		imagefile = request.files['file']
		print imagefile
		filename_ = str(datetime.datetime.now()).replace(' ', '_') + werkzeug.secure_filename(imagefile.filename)
		filename = os.path.join(UPLOAD_FOLDER, filename_)
		imagefile.save(filename)
		print 'Saving to ' + filename
		result = label.classify(imagefile.filename, filename)
		return json.dumps( result )
	except:
	    print "Unexpected error:", sys.exc_info()[0]
	    raise

@app.route('/api/classify_upload/resnet50', methods=['POST'])
def classify_upload_resnet50():
	try:
		print 'begin classify'
		print request.files
		imagefile = request.files['file']
		print imagefile
		filename_ = str(datetime.datetime.now()).replace(' ', '_') + werkzeug.secure_filename(imagefile.filename)
		filename = os.path.join(UPLOAD_FOLDER, filename_)
		imagefile.save(filename)
		print 'Saving to ' + filename
		result = resnet50.classify(imagefile.filename, filename)
		return json.dumps( result )
	except:
	    print "Unexpected error:", sys.exc_info()[0]
	    raise

@app.route('/')
def root():
    return app.send_static_file("index.html")

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
