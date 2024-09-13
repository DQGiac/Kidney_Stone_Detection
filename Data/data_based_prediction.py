import pandas as pd
import joblib
xgb = joblib.load("data_based.pkl")

val = pd.read_csv("data_based/val.csv")
print(val)
for i in range(len(val)):
    grav = val["grav"][i]
    ph = val["ph"][i]
    osmo = val["osmo"][i]
    cond = val["cond"][i]
    urea = val["urea"][i]
    calc = val["calc"][i]
    
    a = pd.DataFrame([pd.Series([grav, ph, osmo, cond, urea, calc])])
    print(a)
    a.columns = ['grav', 'ph', 'osmo', 'cond', 'urea', 'calc']
    res = xgb.predict(a)
    print(res)
    sec_test = (True if res[0] < 0.5 else False)
    # print(sec_test)

