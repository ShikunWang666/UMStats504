import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import numpy as np
import statsmodels.api as sm
from deed_tax import deed_tax
import pandas as pd
from statsmodels.sandbox.predict_functional import predict_functional
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

df = deed_tax.copy()

df["year"] = np.floor(1960 + df.SALE_DATE / 365.25)

df = df.loc[df.SALE_AMOUNT >= 50000, :]
df = df.loc[df.SALE_AMOUNT <= 1000000, :]

df["log_SALE_AMOUNT"] = np.log2(df.SALE_AMOUNT)

df = df.loc[df.SALE_DATE >= 365.25*20]
df["age"] = df.year - df.YEAR_BUILT

sale_amt = np.array(df.log_SALE_AMOUNT.values, dtype=np.float32)
year = np.array(df.year.values, dtype=np.float32)
age = np.array(df.age.values, dtype=np.float32)
land_sqft = np.array(df.LAND_SQUARE_FOOTAGE.values, dtype=np.float32)
living_sqft = np.array(df.LIVING_SQUARE_FEET.values, dtype=np.float32)

def normalize(x, z):
    return (x - z.min()) / (z.max() - z.min())

def input_fn():

    sale_amt1 = normalize(sale_amt, sale_amt)
    year1 = normalize(year, year)
    age1 = normalize(age, age)
    land_sqft1 = normalize(land_sqft, land_sqft)
    living_sqft1 = normalize(living_sqft, living_sqft)
    
    feature_cols = {'year': tf.constant(year1), 'age':
                    tf.constant(age1), 'land_sqft':
                    tf.constant(land_sqft1), 'living_sqft':
                    tf.constant(living_sqft1)}
    response = tf.constant(sale_amt1)

    return feature_cols, response

def gen_predict_fn(z):
    def predict_fn():

        # DV can be set to anything
        sale_amt1 = np.zeros(2016-1980, dtype=np.float32)

        # IVs
        year1 = np.arange(1980, 2016).astype(np.float32)
        land_sqft1 = np.ones(2016-1980, dtype=np.float32) * np.median(df.LAND_SQUARE_FOOTAGE)
        living_sqft1 = np.ones(2016-1980, dtype=np.float32) * np.median(df.LIVING_SQUARE_FEET)
        age1 = z*np.ones(2016-1980, dtype=np.float32)

        # Normalize compatibly with training data
        year1 = normalize(year1, year)
        age1 = normalize(age1, age)
        land_sqft1 = normalize(land_sqft1, land_sqft)
        living_sqft1 = normalize(living_sqft1, living_sqft)

        feature_cols = {"year": tf.constant(year1), "age": tf.constant(age1),
                        "land_sqft": tf.constant(land_sqft1), "living_sqft": tf.constant(living_sqft1)}

        response = tf.constant(sale_amt1)

        return feature_cols, response
        
    return predict_fn


features = [tf.contrib.layers.real_valued_column("year", dimension=1),
            tf.contrib.layers.real_valued_column("age", dimension=1),
            tf.contrib.layers.real_valued_column("land_sqft", dimension=1),
            tf.contrib.layers.real_valued_column("living_sqft", dimension=1)]


pdf = PdfPages("salesprice_tf.pdf")

#for units in ([1,], [2,], [4,], [8,], [16,], [32,], [64,], [128,], [256,], [512,], 
#              [512, 256]): #, [512, 256, 128], [512, 256, 128, 64], [512, 256, 128, 64, 32]):
#    for steps in 50,100,500:
#        for jo in 0,1:
#            for lr in 0.01,0.1:

for units in ([128,], [256,], [512], [512, 256]):
    for steps in 500,:
        for jo in 0,1:
            for lr in 0.01,0.1,:

                if jo == 0:
                    optname = "GD"
                    opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
                elif jo == 1:
                    optname = "Adam"
                    opt = tf.train.AdamOptimizer(learning_rate=lr)

                estimator = tf.contrib.learn.DNNRegressor(feature_columns=features, hidden_units=units,
                                                          optimizer=opt)
                estimator.fit(input_fn=input_fn, steps=steps)

                # Get final training loss
                ev = estimator.evaluate(input_fn=input_fn, steps=1)
                loss = ev['loss']

                plt.clf()
                plt.axes([0.12, 0.1, 0.76, 0.8])

                for p in 0,10,20,40:
                    z = []
                    for x in estimator.predict(input_fn=gen_predict_fn(p)):
                        z.append(x)
                    z = np.asarray(z)
                    z = z * (sale_amt.max() - sale_amt.min()) + sale_amt.min()
                    plt.plot(np.arange(1980, 2016), z, '-', label=str(p))

                ha,lb = plt.gca().get_legend_handles_labels()
                leg = plt.figlegend(ha, lb, "center right")
                leg.draw_frame(False)
                plt.grid(True)
                plt.xlabel("Year", size=15)
                plt.ylabel("Sale amount (log2)", size=15)
                plt.title("units=%s, steps=%s, rate=%.2f, %s optimizer, loss=%.4f" % 
                          (str(units), str(steps), lr, optname, 100*loss))
                pdf.savefig()

pdf.close()
