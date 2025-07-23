"""         FIRST TEST          """
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
first_test = (False if a[0] < 0.5 else True)


"""         SECOND TEST             """


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
train = pd.read_csv("data_based/train.csv")
test = pd.read_csv("data_based/test.csv")
val = pd.read_csv("data_based/val.csv")

cols = ["gravity", "ph", "osmo", "cond", "urea", "calc"]
X_train = train[cols]
y_train = train["target"]
X_val = val[cols]
y_val = val["target"]

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
xgb = XGBRegressor(n_estimators=500, learning_rate=0.005)
xgb.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_val, y_val)])

grav = float(input("Tỷ trọng tương đối của nước tiểu: "))
ph = float(input("Độ pH của nước tiểu: "))
osmo = float(input("Độ thẩm thấu (mOsm) của nước tiểu: "))
cond = float(input("Độ dẫn điện (milliMho) của nước tiểu: "))
urea = float(input("Nồng độ Urea (milimoles/lít) của nước tiểu: "))
calc = float(input("Nồng độ Calcium (milimoles/lít) của nước tiểu: "))

a = pd.DataFrame([pd.Series([grav, ph, osmo, cond, urea, calc])])
a.columns = cols
res = xgb.predict(a)
sec_test = (False if res[0] < 0.5 else True)


if first_test == sec_test:
    print("\nBệnh nhân có thể", "bị" if sec_test == True else "không bị", "sỏi thận.")
else:
    print("Chúng tôi không rõ liệu bệnh nhân có sỏi thận hay không. Xin hãy đến cơ quan y tế gần nhất để khám với bác sĩ có chuyển môn.")
print("\n\nXin hãy chú ý rằng vì đây là một chuẩn đoán của một hệ thống AI chưa được kiểm chứng nên những chuẩn đoán có khả năng nhầm lẫn. Bệnh nhân nên khám với bác sĩ có chuyên môn.")