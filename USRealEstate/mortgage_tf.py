import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import numpy as np
import statsmodels.api as sm
from deed_data import deed
import pandas as pd
from statsmodels.sandbox.predict_functional import predict_functional
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

deed.columns = [x.replace(" ", "_") for x in deed.columns]

mort = deed.copy()
mort = mort.loc[pd.notnull(mort["MORTGAGE_AMOUNT"]), :]

mort["year"] = np.floor(1960 + mort.SALE_DATE / 365.25)

mort = mort.loc[mort.SALE_AMOUNT >= 50000, :]
mort = mort.loc[mort.SALE_AMOUNT <= 1000000, :]

mort = mort.loc[mort.MORTGAGE_AMOUNT >= 0.5*mort.SALE_AMOUNT, :]
mort = mort.loc[mort.MORTGAGE_AMOUNT <= mort.SALE_AMOUNT, :]

mort["log_MORTGAGE_AMOUNT"] = np.log2(mort.MORTGAGE_AMOUNT)
mort["log_SALE_AMOUNT"] = np.log2(mort.SALE_AMOUNT)

mort = mort.loc[mort.SALE_DATE >= 365.25*20]
mort = mort[["log_MORTGAGE_AMOUNT", "log_SALE_AMOUNT", "SALE_DATE", "FIPS", "year"]].dropna()

mort_amt = np.array(mort.log_MORTGAGE_AMOUNT.values, dtype=np.float32)
sale_amt = np.array(mort.log_SALE_AMOUNT.values, dtype=np.float32)
year = np.array(mort.year.values, dtype=np.float32)

def normalize(x, z):
    return (x - z.min()) / (z.max() - z.min())

def input_fn():

    mort_amt1 = normalize(mort_amt, mort_amt)
    sale_amt1 = normalize(sale_amt, sale_amt)
    year1 = normalize(year, year)
    feature_cols = {'sale_amt': tf.constant(sale_amt1), 'year': tf.constant(year1)}
    response = tf.constant(mort_amt1)

    return feature_cols, response

def gen_predict_fn(z):
    def predict_fn():

        mort_amt1 = np.zeros(shape=2016-1980, dtype=np.float32)
        sale_amt1 = np.array(z*np.ones(shape=2016-1980), dtype=np.float32)
        year1 = np.arange(1980, 2016).astype(np.float32)

        mort_amt1 = normalize(mort_amt1, mort_amt)
        sale_amt1 = normalize(sale_amt1, sale_amt)
        year1 = normalize(year1, year)
        feature_cols = {"sale_amt": tf.constant(sale_amt1), "year": tf.constant(year1)}
        response = tf.constant(mort_amt1)

        return feature_cols, response
        
    return predict_fn


features = [tf.contrib.layers.real_valued_column("sale_amt", dimension=1),
            tf.contrib.layers.real_valued_column("year", dimension=1)]

estimator = tf.contrib.learn.DNNRegressor(feature_columns=features, hidden_units=[512, 256])
estimator.fit(input_fn=input_fn, steps=100)

pdf = PdfPages("mortgage_tf.pdf")

plt.clf()
plt.axes([0.12, 0.1, 0.76, 0.8])

for p in 16,17,18,19:
    z = []
    for x in estimator.predict(input_fn=gen_predict_fn(p)):
        z.append(x)
    z = np.asarray(z)
    z = z * (mort_amt.max() - mort_amt.min()) + mort_amt.min()
    plt.plot(np.arange(1980, 2016), z, '-', label=str(p))

ha,lb = plt.gca().get_legend_handles_labels()
leg = plt.figlegend(ha, lb, "center right")
leg.draw_frame(False)
plt.grid(True)
plt.xlabel("Year", size=15)
plt.ylabel("Mortgage (log2)", size=15)
pdf.savefig()

pdf.close()
