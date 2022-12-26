import os
import cv2
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split

import gc
from typing import Tuple

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

class StdOutCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logging.info(
            "Epoch {:4d}: accuracy={:.4f} - loss={:.4f}".format(
                epoch+1, logs["accuracy"], logs["loss"]
            )
        )

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', 
                        type=str, 
                        default='/logs',
                        help='Name of the model folder.')
    parser.add_argument('--data_dir',
                        type=str,
                        default='../pokemon_data/xy12_images/',
                        help='Name of the train data folder.')
    parser.add_argument('--width',
                        type=int,
                        default=120,
                        help='The width of the train data.')
    parser.add_argument('--height',
                        type=int,
                        default=185,
                        help='The height of the train data.')
    parser.add_argument('--examples_per_class',
                        type=int,
                        default=3,
                        help='The min number of examples per class.')
    parser.add_argument('--steps_per_epoch',
                        type=int,
                        default=100,
                        help='The number of train steps per epoch.')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.0001,
                        help='Learning rate for training.')
    parser.add_argument('--embedding_size',
                        type=int,
                        default=128,
                        help='Size of output embedding.')
    parser.add_argument('--epochs',
                        type=int,
                        default=4,
                        help='The number of train epochs.')
    parser.add_argument('--export_folder',
                        type=str,
                        default='../models/pokemon_similarity',
                        help='Folder to save model.')

    args, _ = parser.parse_known_args(args=argv[1:])

    return args

def import_train_data(path, dim):
    x = []
    y = []
    class_names=[]
    i = 0
    for folder in os.listdir(path):
        class_names.append(folder)
        for filepath in os.listdir(path+folder):
            # import images
            stream = open(u'{0}{1}/{2}'.format(path, folder, filepath), "rb")
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

            y.append(i)
        i += 1

    x = np.asarray(x)
    y = np.asarray(y)

    # Save the classes
    np.save('classes.npy', np.unique(y))

    # Save the class_names
    np.save('class_names.npy', class_names)

    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, stratify=y, random_state=42)
    print('x_train shape:', x_train.shape, 'y_train shape:', y_train.shape)
    print('x_test shape:', x_test.shape, 'y_test shape:', y_test.shape)

    return x_train, x_test, y_train, y_test, class_names

def build_sampler(x_train, y_train, argv=None):
    args = parse_arguments(sys.argv if argv is None else argv)

    # Define data augmentation layers
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.1, fill_mode='constant'),
            tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=(-0.05, 0.2), fill_mode='constant'),
        ]
    )

    def augmenter(
        x: tfsim.types.FloatTensor, y: tfsim.types.IntTensor, examples_per_class: int, is_warmup: bool, stddev=0.08
    ) -> Tuple[tfsim.types.FloatTensor, tfsim.types.IntTensor]:
        """Image augmentation function.

        Args:
            X: FloatTensor representing the example features.
            y: IntTensor representing the class id. In this case
            the example index will be used as the class id.
            examples_per_class: The number of examples per class.
            Not used here.
            is_warmup: If True, the training is still in a warm
            up state. Not used here.
            stddev: Sets the amount of gaussian noise added to
            the image.
        """
        _ = examples_per_class
        _ = is_warmup

        aug = tf.squeeze(data_augmentation(x))
        aug = aug + tf.random.normal(tf.shape(aug), stddev=stddev)
        x = tf.concat((x, aug), axis=0)
        y = tf.concat((y, y), axis=0)
        idxs = tf.range(start=0, limit=tf.shape(x)[0])
        idxs = tf.random.shuffle(idxs)
        x = tf.gather(x, idxs)
        y = tf.gather(y, idxs)
        return x, y

    # Create the sampler
    sampler = tfsim.samplers.MultiShotMemorySampler(
        x_train,
        y_train,
        augmenter=augmenter,
        classes_per_batch = len(np.unique(y_train)),
        examples_per_class_per_batch = args.examples_per_class,
        class_list=list(np.unique(y_train)),
        steps_per_epoch=args.steps_per_epoch,
    )

    return sampler

def build_model(dim, embedding_size=128, distance="cosine", learning_rate=0.0001):
    model = tfsim.architectures.ResNet50Sim(
        (dim[1], dim[0], 3), # NHWC is default
        embedding_size=embedding_size, # embedding_size
        trainable='full',
        pooling="gem",    # Can change to use `gem` -> GeneralizedMeanPooling2D
        gem_p=3.0,        # Increase the contrast between activations in the feature map.
    )

    # Define the loss function
    # distance=["cosine", "L2", "L1"]
    loss = tfsim.losses.MultiSimilarityLoss(distance=distance)

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=loss)

    return model

def train(argv=None):
    args = parse_arguments(sys.argv if argv is None else argv)

    dim = (args.width, args.height) # Input dimensions

    # Import training data
    x_train, x_test, y_train, y_test, _ = import_train_data(args.data_dir, dim)

    # Build the sampler
    sampler = build_sampler(x_train, y_train, argv=None)
    print(f"The sampler contains {len(sampler)} steps per epoch.")
    print(f"The sampler is using {sampler.num_examples} examples out of the original {len(x_train)}.")
    print(f"Each examples has the following shape: {sampler.example_shape}.")

    # Build the model
    model = build_model(dim, embedding_size=args.embedding_size, distance="cosine", learning_rate=args.learning_rate)

    # Define callbacks
    logdir = args.log_dir + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback= tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                                        update_freq='epoch')
    std_out = StdOutCallback()                                                    

    # Train the model
    _ = model.fit(sampler,
                  epochs=args.epochs,
                  validation_data=(x_test, y_test),
                  callbacks=[tensorboard_callback, std_out])

    # Index the model
    x_index, y_index = tfsim.samplers.select_examples(x_train, y_train, np.unique(y_train), 20)
    model.reset_index()
    model.index(x_index, y_index, data=x_index)

    # Calibrate the model (find optimal threshold using all train samples)
    num_calibration_samples = x_train.shape[0]  # @param {type:"integer"}
    _ = model.calibrate(
        x_train[:num_calibration_samples],
        y_train[:num_calibration_samples],
        extra_metrics=["precision", "recall", "binary_accuracy"],
        verbose=1,
    )

    # save the model and the index
    if args.export_folder:
        model.save(args.export_folder, save_index=True)
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    train()
