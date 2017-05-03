from models.keras.resnet50 import ResNet50
from keras.preprocessing import image
from models.keras.imagenet_utils import preprocess_input, decode_predictions
import numpy as np

## This one has problems with mixin with inceptionv3 seems to be some threading issue
## https://github.com/fchollet/keras/issues/2397
def classify(name, img_path):
    model = ResNet50(weights='imagenet')
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    predictions = decode_predictions(preds)
    classify_result = []
    for p in predictions[0]:
      result = {
            'label':p[1],
            'score':np.asscalar(p[2])
            }
      classify_result.append(result)

    return {
      'model':name,
      'labels':classify_result
      }


if __name__ == "__main__":
  img_path = './data/apple/bad/thumb_IMG_0557_1024.jpg'
  name = 'bad_apple'
  result = classify(name, img_path)
  print result