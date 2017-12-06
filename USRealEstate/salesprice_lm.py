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

fml = "log_SALE_AMOUNT ~ bs(year, 6) * bs(age, 6) + bs(year, 6) * (bs(LAND_SQUARE_FOOTAGE, 6) + bs(LIVING_SQUARE_FEET, 6) + bs(age, 6))"
model = sm.OLS.from_formula(fml, df)
result = model.fit()


pdf = PdfPages("salesprice_lm.pdf")

plt.clf()
for age in 0,10,20,40:
    pred, cb, fvals = predict_functional(result, "year", values={"age": age}, summaries={"LAND_SQUARE_FOOTAGE": np.median,
                                                                                         "LIVING_SQUARE_FEET": np.median})
    plt.plot(fvals, pred, '-', label=str(age))

plt.grid(True)
ha, lb = plt.gca().get_legend_handles_labels()
leg = plt.figlegend(ha, lb, "center right")
leg.draw_frame(False)
plt.ylabel("Sales price (log2)", size=15)
plt.xlabel("Year of sale", size=15)

pdf.savefig()

pdf.close()
