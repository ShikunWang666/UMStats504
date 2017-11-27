import matplotlib
matplotlib.use('Agg')
import numpy as np
import statsmodels.api as sm
from deed_data import deed
import pandas as pd
from statsmodels.sandbox.predict_functional import predict_functional
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

pdf = PdfPages("mortgage_plots.pdf")

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

model1 = sm.OLS.from_formula("log_MORTGAGE_AMOUNT ~ bs(log_SALE_AMOUNT, 8) * bs(SALE_DATE, 8)", data=mort)
result1 = model1.fit()

plt.clf()
ax = plt.axes([0.1, 0.12, 0.75, 0.8])
for k in range(7):
    pred, cb, fvals = predict_functional(result1, "log_SALE_AMOUNT", values={"SALE_DATE": 365.25*(25+5*k)})
    plt.plot(fvals, pred, '-', label="%4d" % (1960 + 25 + 5*k))
ha, lb = ax.get_legend_handles_labels()
leg = plt.figlegend(ha, lb, "center right")
leg.draw_frame(False)
plt.xlabel("log2 Sale Amount", size=15)
plt.ylabel("log2 Mortgage Amount", size=15)
plt.grid(True)
pdf.savefig()

plt.clf()
ax = plt.axes([0.12, 0.12, 0.75, 0.8])
for k in [16, 17, 18, 19]:
    pred, cb, fvals = predict_functional(result1, "SALE_DATE", values={"log_SALE_AMOUNT": k})
    plt.plot(1960 + fvals/365.25, pred, '-', label="%4d" % k)
ha, lb = ax.get_legend_handles_labels()
leg = plt.figlegend(ha, lb, "center right")
leg.draw_frame(False)
plt.xlabel("Sale date", size=15)
plt.ylabel("log2 Mortgage Amount", size=15)
plt.grid(True)
pdf.savefig()

mort["resid"] = result1.resid
var_vals = []
for y,dx in mort.groupby("year"):
    var_vals.append([y, dx.resid.std()])
var_vals = np.asarray(var_vals)

plt.clf()
plt.plot(var_vals[:, 0], var_vals[:, 1], '-')
plt.grid(True)
plt.xlabel("Year", size=15)
plt.ylabel("SD unexplained by year and sales price", size=15)
pdf.savefig()

icc_vals = []
for y in range(1980, 2010, 5):
    dx = mort.loc[(mort.year >= y) & (mort.year < y+5), :]
    icc = dx.groupby("FIPS")["resid"].agg(np.mean).var() / dx.resid.var()
    icc_vals.append([y, icc])
icc_vals = np.asarray(icc_vals)

plt.clf()
plt.plot(icc_vals[:, 0], icc_vals[:, 1], '-')
plt.grid(True)
plt.xlabel("Year", size=15)
plt.ylabel("ICC", size=15)
pdf.savefig()

cr = []
for f,dx in mort.groupby("FIPS"):
    cr.append([dx.shape[0], np.corrcoef(dx.SALE_DATE, dx.resid)[0, 1]])
cr = np.asarray(cr)

for n in 50,100,200,400,800:
    ii = np.flatnonzero(cr[:, 0] > n)
    cr0 = cr[ii, :]
    print(n, len(ii), np.mean(np.abs(cr0[:, 1]) > 2/np.sqrt(n)),
          np.mean(cr0[:, 1] > 2/np.sqrt(n)),
          np.mean(cr0[:, 1] < -2/np.sqrt(n)))

plt.clf()

pdf.savefig()

pdf.close()
