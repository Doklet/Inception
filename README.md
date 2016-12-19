# Classify Service

Service to classify images by using a transfer based learning. This service use google inception model as a base
then only retrain the last layer of the classification model.

## Installation

Make sure to create a virtualenv by executing 'virtualenv .env'
The activate the virtualenv by executing 'source .env/bin/activate'

### Tensorflow
Start by installing tensorflow by executing 'make install_tf'

### pip requirements
Then install the rest of the requirements by executing 'pip install -r requirements.txt'

## Stage the UI
Stage the classify ui by executing 'make stage'