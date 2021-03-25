import tensorflow as tf
import numpy as np
import os
import utils

if __name__ == '__main__':

    if os.path.exists("./checkpoints/savage-RNN-1616611589/rnnckpt" + '.index'):
        model = utils.create_model()
        model.load_weights("./checkpoints/savage-RNN-1616611589/rnnckpt")
        print("*******************model loaded***************************")
        model.save("RNNModel")
        print("Model saved")
