import cv2
import os
import gc
import json
import numpy as np

import argparse
import logging
import sys

# INFO messages are not printed.
# This must be run before loading other modules.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf
import tensorflow_similarity as tfsim

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# tfsim.utils.tf_cap_memory()  # Avoid GPU memory blow up

# Clear out any old model state.
gc.collect()
tf.keras.backend.clear_session()

print("TensorFlow:", tf.__version__)
print("TensorFlow Similarity", tfsim.__version__)

#Use this format (%Y-%m-%dT%H:%M:%SZ) to record timestamp of the metrics
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    level=logging.DEBUG)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', 
                        type=str, 
                        default='../models/pokemon_similarity',
                        help='Name of the model folder.')
    parser.add_argument('--data_dir',
                        type=str,
                        default='../test_images/xy12_test/Clefairy',
                        help='Name of the inference data folder.')
    parser.add_argument('--width',
                        type=int,
                        default=120,
                        help='The width of the inference data.')
    parser.add_argument('--height',
                        type=int,
                        default=185,
                        help='The height of the inference data.')
    parser.add_argument('--export_folder',
                        type=str,
                        default='./predictions',
                        help='Folder to save prediction.')

    args, _ = parser.parse_known_args(args=argv[1:])

    return args

def import_inf_data(path, dim):
    x = []
    for filepath in os.listdir(path):
        # import images
        stream = open(u'{0}/{1}'.format(path, filepath), "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        bgrImage = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

        # resize
        bgrImage = cv2.resize(bgrImage, dim, interpolation = cv2.INTER_AREA)

        # convert bgr to rgb
        rgbImage = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2RGB)

        # convert to float32
        rgbImage = np.asarray(rgbImage).astype('float32')

        # Check height and width
        if rgbImage.shape[0] == dim[1] and rgbImage.shape[1] == dim[0]:
            x.append(rgbImage / 255)
        else:
            print('Dim Not Matching!')

    x = np.asarray(x)

    return x

# Load the model and index
def load_model(model_path):
    # reload the model
    reloaded_model = tf.keras.models.load_model(
        model_path,
        custom_objects={"SimilarityModel": tfsim.models.SimilarityModel},
    )
    # reload the index
    reloaded_model.load_index(model_path)

    return reloaded_model

def load_class_info():
    # Import class_names
    with open('class_names.npy', 'rb') as f:
        class_names = np.load(f)

    # Import classes
    with open('classes.npy', 'rb') as f:
        classes = np.load(f)

    # Create class labels
    labels = np.append(classes, "Unknown")

    return labels, class_names

def inference(argv=None):
    args = parse_arguments(sys.argv if argv is None else argv)

    reloaded_model = load_model(args.model_dir)
    labels, class_names = load_class_info()
    x_inf = import_inf_data(args.data_dir, (args.width, args.height))

    # Get the prediction
    pred = reloaded_model.match(x_inf, cutpoint="optimal", no_match_label=len(labels)-1)

    # Get the predicted class name
    outputs = []
    for val in pred:
        if val > len(class_names)-1:
            outputs.append('Unknown')
        else:
            outputs.append(class_names[val])
    
    print(outputs)
    # Save outputs
    with open(f'{args.export_folder}/output.txt', 'w') as filehandle:
        json.dump(outputs, filehandle)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    inference()
