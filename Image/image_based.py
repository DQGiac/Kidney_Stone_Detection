import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings(action="ignore")
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,  Conv2D, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split

Normal_dirs = ["CT-KIDNEY-DATASET-Normal-Stone/Normal"]
Stone_dirs = ["CT-KIDNEY-DATASET-Normal-Stone/Stone"]


filepaths = []
labels = []
dict_lists = [Normal_dirs, Stone_dirs]
class_labels = ['Normal', 'Stone']

print("fdj")

for i, dir_list in enumerate(dict_lists):
    for j in dir_list:
        flist = os.listdir(j)
        for f in flist:
            fpath = os.path.join(j, f)
            filepaths.append(fpath)
            labels.append(class_labels[i])

Fseries = pd.Series(filepaths, name="filepaths")
Lseries = pd.Series(labels, name="labels")
KIDNEY_data = pd.concat([Fseries, Lseries], axis=1)
KIDNEY_df = pd.DataFrame(KIDNEY_data)

train_set, val_set = train_test_split(KIDNEY_df, test_size=0.2, random_state=42)
val_x = val_set["filepaths"]
val_y = val_set["labels"]

image_gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=keras.applications.mobilenet_v2.preprocess_input)
train = image_gen.flow_from_dataframe(dataframe=train_set,x_col="filepaths",y_col="labels",
                                      target_size=(244,244),
                                      color_mode='rgb',
                                      class_mode="categorical", 
                                      batch_size=8,
                                      shuffle=False            
                                     )
val = image_gen.flow_from_dataframe(dataframe=val_set,x_col="filepaths", y_col="labels",
                                    target_size=(244,244),
                                    color_mode= 'rgb',
                                    class_mode="categorical",
                                    batch_size=8,
                                    shuffle=False
                                   )

model = Sequential([
    Conv2D(filters=128, kernel_size=(8, 8), strides=(3, 3), activation='relu', input_shape=(224, 224, 3)),
    BatchNormalization(),
    
    Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(3, 3)),
    
    Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    BatchNormalization(),
    Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(train, epochs=1, validation_data=val, verbose=1)
model.save("image_based.h5")
# pred = model.predict(test)
# pred = np.argmax(pred, axis=1)

# labels = (train.class_indices)
# labels = dict((v,k) for k,v in labels.items())
# pred2 = [labels[k] for k in pred]
# history = model.fit(train, epochs=1, validation_data=val, verbose=1)
# model.save("KIDNEY-Diseases.h5")