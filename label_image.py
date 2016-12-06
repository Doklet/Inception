import sys
import tensorflow as tf
import numpy as np

# change this as you see fit
# arg_image_path = sys.argv[1]

# if arg_image_path:
#     classify(arg_image_path)

def classify(name, image_path):
    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line 
                       in tf.gfile.GFile("data/retrained_labels.txt")]

    # Unpersists graph from file
    with tf.gfile.FastGFile("data/retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    # Disable GPU
    config = tf.ConfigProto(
            device_count = {'GPU': 0}
    )
    with tf.Session(config=config) as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
        
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        
        classify_result = []
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            result = {
                'name':name,
                'label':human_string,
                'score':np.asscalar(score)
            }
            classify_result.append(result)
            print result
        
        return classify_result

if __name__ == '__main__':
    # 
    if sys.argv[1]:
        classify(sys.argv[1])
