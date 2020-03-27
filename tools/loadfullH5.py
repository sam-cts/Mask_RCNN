import tensorflow as tf 


reloaded_model = tf.keras.experimental.load_from_saved_model(name_model, custom_objects={'KerasLayer':tf.hub.KerasLayer})
print(reloaded_model.get_config())
reloaded_model.build((None, 224, 224, 3))
reloaded_model.summary()    
