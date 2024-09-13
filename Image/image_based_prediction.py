import tensorflow as tf
# import keras
import pandas as pd
import numpy as np
model = tf.keras.models.load_model("image_based.h5")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
image_gen = ImageDataGenerator(preprocessing_function= tf.keras.applications.mobilenet_v2.preprocess_input)
ser1 = pd.Series(["test.jpg", "Normal"])

testset = pd.DataFrame([ser1])

test = image_gen.flow_from_dataframe(dataframe=testset,x_col=0, y_col=1,
                                    target_size=(244,244),
                                    color_mode= 'rgb',
                                    class_mode="categorical",
                                    batch_size=8,
                                    shuffle=False
                                   )

a = model.predict(test)
a = np.argmax(a, axis=1)
first_test = (True if a[0] < 0.5 else False)