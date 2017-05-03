import os
import caffe
import numpy as np
import pandas as pd
import exifutil

def classify(name, img_path):
  image = exifutil.open_oriented_im(img_path)
  labels = load_labels(name)
  classifier = load_classifier(name)
  scores = classifier.predict([image], oversample=True).flatten()
  print img_path
  return zip(labels, scores)

def load_classifier(name):
  model_def_file = "inference/" + name + "/deploy.prototxt"
  pretrained_model_file = "inference/" + name + "/weights.caffemodel"
  mean = load_mean(name)
  image_dim = 256
  raw_scale = 255
  return caffe.Classifier(
            model_def_file, pretrained_model_file,
            image_dims=(image_dim, image_dim), raw_scale=raw_scale,
            mean=mean.mean(1).mean(1), channel_swap=(2, 1, 0)
        )


def load_mean(name):
  blob = caffe.proto.caffe_pb2.BlobProto()
  data = open( 'inference/' + name + '/mean.binaryproto' , 'rb' ).read()
  blob.ParseFromString(data)
  return np.array( caffe.io.blobproto_to_array(blob) )[0]

def load_labels(name):
    # class_labels_file = 'inference/' + name + '/labels.txt'
    # return pd.read_csv(class_labels_file, header = None)
    return ['good', 'bad']
    # with open(class_labels_file) as f:
    #   labels_df = pd.DataFrame([
    #       {
    #           'synset_id': l.strip().split(' ')[0],
    #           'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
    #       }
    #       for l in f.readlines()
    #   ])
    #   return labels_df.sort('synset_id')['name'].values

if __name__ == "__main__":
  img_path1 = './data/apple/bad/thumb_IMG_0573_1024.jpg'
  img_path2 = './data/apple/good/thumb_IMG_0610_1024.jpg'
  name = 'apple'
  result = classify(name, img_path2)
  print result