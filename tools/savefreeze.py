import os
import sys
import warnings
from tensorflow.keras import backend as K
# import keras.backend as K
import tensorflow as tf
import tensorflow_hub as hub
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
warnings.filterwarnings('ignore', category=FutureWarning)
# suppress warning and error message tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Root directory of the project
ROOT_DIR = os.getcwd()
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import model as modellib
from mrcnn import utils
from samples.chips import chips_fullstack
K.clear_session()
K.set_learning_phase(0)

##############################################################################
# Load model
##############################################################################


# Model Directory
MODEL_DIR = "/datasets/models/mrcnn/fullstack_fullnetwork/chips20200513T0630/"
DEFAULT_WEIGHTS = "/datasets/models/mrcnn/fullstack_fullnetwork/chips20200513T0630/mask_rcnn_chips_0030.h5"

##############################################################################
# Load configuration
##############################################################################



class InferenceConfig(chips_fullstack.ChipsConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        

##############################################################################
# Save entire model function
##############################################################################

def h5_to_pb(h5_model, output_dir, model_name, out_prefix="output_"):
    out_nodes = []
    for i in range(len(h5_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(h5_model.output[i], out_prefix + str(i + 1))
    sess = K.get_session()
    init_graph = sess.graph.as_graph_def()
    # main_graph = tf._api.v1.graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)
    main_graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)
    with tf.gfile.GFile(os.path.join(output_dir, model_name), "wb") as filemodel:
        filemodel.write(main_graph.SerializeToString())
    print("pb model: ", {os.path.join(output_dir, model_name)})


if __name__ == "__main__":
    config = InferenceConfig()
    config.display()
    # Create model in inference mode
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Set path to model weights
    weights_path = DEFAULT_WEIGHTS#model.find_last()
    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)
    model.keras_model.summary()

    # make folder for full model
    model_dir = os.path.join(ROOT_DIR, "Model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # # save h5 full model
    # name_model = os.path.join(model_dir, "mask_rcnn_chips.h5")
    # # tf.keras.experimental.export_saved_model(model.keras_model, name_model)
    # model.keras_model.save(name_model)
    # print("save model: ", name_model)    

    # tf_model_path = os.path.join(model_dir, "tfmodel")


    # saved_model = tf.keras.models.load_model(name_model)
    # saved_model.save(tf_model_path, save_format='tf')

    # # reloaded_model = tf.keras.experimental.load_from_saved_model(name_model, custom_objects={'KerasLayer':hub.KerasLayer})
    # # print(reloaded_model.get_config())
    # # reloaded_model.build((None, 224, 224, 3))
    # # reloaded_model.summary()    


    # export pb model
    # pb_name_model = "mask_rcnn_chips_builder.pb"
    # h5_to_pb(model.keras_model, output_dir=model_dir, model_name=pb_name_model)

    model.keras_model.inputs

    # export to savedmodel
    saved_model_name = os.path.join(model_dir, "idNerd_chips")
    # builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(saved_model_name)                                                                                                                  
    print(model.keras_model.inputs)
    print(model.keras_model.outputs)
    # signature = tf.saved_model.signature_def_utils.predict_signature_def(                                                                        
    #     inputs={'image': model.keras_model.input[0]}, 
    #     outputs={'class': model.keras_model.output[1], 
    #              'bbox': model.keras_model.output[2], 
    #              'mask': model.keras_model.output[3]})                                                                         
                                                                                                                                                
    builder = tf.saved_model.builder.SavedModelBuilder(saved_model_name)                                                                    
    builder.add_meta_graph_and_variables(                                                                                                        
        sess=K.get_session(),                                                                                                                    
        tags=["serve"],                                                                                             
        # signature_def_map={                                                                                                                      
        #     tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:                                                                
        #         signature                                                                                                                        
        # }
        )                                                                                                                                       
    builder.save()

    K.clear_session()
    sys.exit()