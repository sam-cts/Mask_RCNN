import tensorflow as tf
import os
##############################################################################
# Load model
##############################################################################

ROOT_DIR = os.getcwd()

# Model Directory
MODEL_DIR = "/datasets/models/mrcnn/fullstack_fullnetwork/chips20200326T0339/"
DEFAULT_WEIGHTS = "/datasets/models/mrcnn/fullstack_fullnetwork/chips20200326T0339/mask_rcnn_chips_0030.h5"

if __name__ == "__main__":
    model_dir = os.path.join(ROOT_DIR, "Model")
    export_path = os.path.join(model_dir, "mask_rcnn_chips_builder_no_sign")

    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ["chips"], export_path)

        graph = tf.get_default_graph()
        # print(graph.get_operations())

        sess.run()