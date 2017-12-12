import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from data import dfps, dfpc, dfps0, dfpc0
import statsmodels.api as sm
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

pdf = PdfPages("afghan.pdf")

## Turnout analysis

for df in dfps0, dfpc0:

    if df is dfps0:
        title = "Polling stations"
    else:
        title = "Polling centers"

    plt.clf()
    plt.axes([0.15, 0.1, 0.8, 0.8])
    plt.plot(np.arange(df.shape[0]), np.sort(df.Total_pre), 'o', rasterized=True)
    plt.xlabel("Rank", size=15)
    plt.ylabel("Sorted round 1 turnout", size=15)
    plt.grid(True)
    plt.title(title)
    pdf.savefig()

    plt.clf()
    plt.axes([0.15, 0.1, 0.8, 0.8])
    plt.plot(np.arange(df.shape[0]), np.sort(df.Total), 'o', rasterized=True)
    plt.xlabel("Rank", size=15)
    plt.ylabel("Sorted round 2 turnout", size=15)
    plt.grid(True)
    plt.title(title)
    pdf.savefig()

    plt.clf()
    plt.axes([0.15, 0.1, 0.8, 0.8])
    plt.plot(df.Total_pre, df.Total, 'o', rasterized=True,
             alpha=0.3)
    plt.grid(True)
    plt.xlabel("Round 1 turnout", size=15)
    plt.ylabel("Round 2 turnout", size=15)
    plt.title(title)
    pdf.savefig()


f = dfps0.Total.sum() / dfps0.Total_pre.sum()
r = dfps0.Total - f*dfps0.Total_pre
sr = np.asarray(r / np.sqrt(dfps0.Total + f*dfps0.Total_pre))
sr = sr[np.isfinite(sr)]

for df in dfps, dfpc:
    model1 = sm.OLS.from_formula("logTotal ~ logTotal_pre", data=df)
    result1 = model1.fit()
    print(result1.summary())

# Look at overdispersion in vote totals.
print("Overdispersion in vote totals:")
for j,df in enumerate([dfpc, dfps, dfpc0, dfps0]):
    print(["PC (all)", "PS (all)", "PC (nonzero)", "PS (nonzero)"][j])
    model = sm.OLS.from_formula("Total ~ Total_pre", data=df)
    result = model.fit()
    print("OLS    %10.2f" % result.aic)
    model = sm.OLS.from_formula("Total ~ logTotal_pre", data=df)
    result = model.fit()
    print("OLS    %10.2f" % result.aic)
    model = sm.GLM.from_formula("Total ~ logTotal_pre", data=df, family=sm.families.Poisson())
    result = model.fit()
    print("Pois   %10.2f" % result.aic)
    for alpha in 0.001,0.1,0.5,0.75,1,1.25,1.5,2:
        model = sm.GLM.from_formula("Total ~ logTotal_pre", family=sm.families.NegativeBinomial(alpha=alpha), 
                                    data=df, missing='drop')
        result = model.fit()
        print("%5.3f  %10.2f" % (alpha, result.aic))
    print("\n")

model = sm.OLS.from_formula("Total ~ Total_pre", data=dfpc)
result = model.fit()
plt.clf()
plt.plot(result.fittedvalues, result.resid, 'o', rasterized=True, alpha=0.5)
plt.grid(True)
plt.title("OLS")
plt.xlabel("Fitted value", size=15)
plt.ylabel("Residual", size=15)
pdf.savefig()

model = sm.OLS.from_formula("logTotal ~ logTotal_pre", data=dfpc)
result = model.fit()
plt.clf()
plt.plot(result.fittedvalues, result.resid, 'o', rasterized=True, alpha=0.5)
plt.grid(True)
plt.title("log/log OLS")
plt.xlabel("Fitted value", size=15)
plt.ylabel("Residual", size=15)
pdf.savefig()

model = sm.OLS.from_formula("I(np.sqrt(Total)) ~ I(np.sqrt(Total_pre))", data=dfpc)
result = model.fit()
plt.clf()
plt.plot(result.fittedvalues, result.resid, 'o', rasterized=True, alpha=0.5)
plt.grid(True)
plt.title("sqrt/sqrt OLS")
plt.xlabel("Fitted value", size=15)
plt.ylabel("Residual", size=15)
pdf.savefig()


for df in dfps, dfpc:

    if df is dfps:
        title = "Polling stations"
    else:
        title = "Polling centers"

    prop = np.asarray((1 + df.Ghani) / (1 + df.Total))
    plt.clf()
    plt.hist(prop, bins=50)
    plt.xlabel("Ghani proportion")
    plt.title(title)
    pdf.savefig()

    plt.clf()
    plt.plot(df.Total, prop, 'o', alpha=0.3, rasterized=True)
    plt.xlabel("Total votes", size=15)
    plt.ylabel("Ghani proportion", size=15)
    plt.title(title)
    pdf.savefig()

# Poisson fit looking at scaling of votes for one candidate relative to the total.
for df in dfps, dfpc:
    model1 = sm.GLM.from_formula("Ghani ~ logTotal", family=sm.families.Poisson(), data=df)
    result1 = model1.fit()

# Look at overdispersion.
print("Overdispersion in Ghani votes:")
for alpha in 0.1,0.5,0.75,1,1.25,1.5,2:
    model = sm.GLM.from_formula("Ghani ~ logTotal", family=sm.families.NegativeBinomial(alpha=alpha), 
                                 data=dfpc, missing='drop')
    result = model.fit()
    print("%5.2f %10.2f" % (alpha, result.aic))
    
model2 = sm.GLM.from_formula("Ghani ~ logTotal", family=sm.families.NegativeBinomial(alpha=1), 
                            data=df, missing='drop')
result2 = model2.fit()
pa = result2.params

# Look 
df["logVoteDiff"] = df.logTotal - df.logTotal_pre
model3 = sm.GLM.from_formula("Ghani ~ logTotal + logVoteDiff", 
                            family=sm.families.NegativeBinomial(alpha=1), 
                            data=df, missing="drop")
result3 = model3.fit()

pre = ['Eng_Qutbuddin_Hilal_pre', 'Dr_Abdullah_Abdullah_pre',
       'Zalmai_Rassoul_pre', 'Abdul_Rahim_Wardak_pre', 'Quayum_Karzai_pre',
       'Prof_Abdo_Rabe_Rasool_Sayyaf_pre',
       'Dr_Mohammad_Ashraf_Ghani_Ahmadzai_pre',
       'Mohammad_Daoud_Sultanzoy_pre', 'Mohd_Shafiq_Gul_Agha_Sherzai_pre',
       'Mohammad_Nadir_Naeem_pre', 'Hedayat_Amin_Arsala_pre']

# Model in terms of relative increases
for x in pre:
    df[x + "_prop"] = np.log2(1 + 100 * df[x] / df[pre].sum(1))
pre_prop = [x + "_prop" for x in pre]
fml = "Ghani ~ logTotal + " + " + ".join(pre_prop)
model4 = sm.GLM.from_formula(fml, family=sm.families.Poisson(), data=df)
result4 = model4.fit()

# Model in terms of absolute increases
for x in pre:
    df[x + "_prop"] = 100 * df[x] / df[pre].sum(1)
pre_prop = [x + "_prop" for x in pre]
fml = "Ghani ~ 0 + logTotal + " + " + ".join(pre_prop)
model5 = sm.GLM.from_formula(fml, family=sm.families.Poisson(), data=df)
result5 = model5.fit()
    
xm = np.asarray(df[pre])
xm = np.log(1+xm)
cm = np.cov(xm.T)
u, s, vt = np.linalg.svd(cm)
cr = np.corrcoef(xm.T)
ur, sr, vrt = np.linalg.svd(cr)

pdf.close()
