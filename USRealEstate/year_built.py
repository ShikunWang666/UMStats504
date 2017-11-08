import pandas as pd
import numpy as np
from tax_data import tax
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# County population data
# http://www.nber.org/data/census-intercensal-county-population.html

cp = pd.read_csv("county_population.csv", encoding="latin1")
cp = pd.melt(cp, id_vars="fips", var_name="year", value_name="pop")
ii = cp.year.str.startswith("pop")
cp = cp.loc[ii, :]
cp["year"] = cp.year.apply(lambda x: x[3:])
cp = cp.loc[cp.fips != 0, :]
cp = cp.dropna()
cp["year"] = cp.year.astype(np.int)
cp["pop"] = cp["pop"].astype(np.int)
cp["logpop"] = np.log(cp["pop"])

nb = tax.groupby(["FIPS CODE", "YEAR BUILT"]).size()
nb = nb.reset_index()
nb = nb.loc[nb["YEAR BUILT"] >= 1960, :]

nb.columns = ["fips", "year", "numbuilt"]
nb["year"] = nb["year"].astype(np.int)
nb["lognumbuilt"] = np.log(nb.numbuilt)

nb = pd.merge(cp, nb, left_on=["fips", "year"], right_on=["fips", "year"])

# Fit a Poisson model
model = sm.GLM.from_formula("numbuilt ~ bs(year, 10) * logpop", data=nb, family=sm.families.Poisson())
result = model.fit()

# Check the mean/variance relationship for the Poisson model
qt = pd.qcut(result.fittedvalues, 50)
dy = pd.DataFrame({"fit": result.fittedvalues, "obs": model.endog})
qa = dy.groupby(qt).agg({"fit": np.mean, "obs": np.var})

for alpha in 0.1, 0.25, 0.5, 0.75, 1:
    model = sm.GLM.from_formula("numbuilt ~ bs(year, 10) * logpop", data=nb, family=sm.families.NegativeBinomial(alpha=alpha))
    result = model.fit()
    print(alpha, result.aic)

model1 = sm.GEE.from_formula("numbuilt ~ bs(year, 10) * logpop", data=nb, family=sm.families.NegativeBinomial(alpha=0.5),
                            cov_struct=sm.cov_struct.Exchangeable(), groups="fips")
result1 = model1.fit(maxiter=50)

model2 = sm.GEE.from_formula("numbuilt ~ bs(year, 10) * logpop", data=nb, family=sm.families.NegativeBinomial(alpha=0.5),
                             cov_struct=sm.cov_struct.Stationary(max_lag=20), time=nb.year-1960, groups="fips")
result2 = model2.fit(maxiter=50)

model3 = sm.GEE.from_formula("numbuilt ~ bs(year, 10) * logpop", data=nb, family=sm.families.NegativeBinomial(alpha=0.5),
                             cov_struct=sm.cov_struct.Exchangeable(), time=nb.year-1960, groups="fips")
result3 = model3.fit()


from statsmodels.sandbox import predict_functional
pred1, cb1, fvals1 = predict_functional.predict_functional(result3, "year", ci_method="simultaneous", values={"logpop": 10})
pred2, cb2, fvals2 = predict_functional.predict_functional(result3, "year", ci_method="simultaneous", values={"logpop": 11})
pred3, cb3, fvals3 = predict_functional.predict_functional(result3, "year", ci_method="simultaneous", values={"logpop": 12})

pdf = PdfPages("year_built.pdf")

plt.clf()
plt.plot(np.log(qa.fit), np.log(qa.obs), 'o')
x = np.linspace(-0.5, 4, 10)
plt.plot(x, np.log(np.exp(x) + 0.5*np.exp(2*x)), '-')
plt.xlabel("log mean", size=15)
plt.ylabel("log variance", size=15)
plt.grid(True)
pdf.savefig()

plt.clf()
plt.plot(np.log(qa.fit), np.log(qa.obs - qa.fit), 'o')
x = np.linspace(-0.5, 4, 10)
plt.plot(x, np.log(0.5) + 2*x, '-')
plt.xlabel("log mean", size=15)
plt.ylabel("log (variance - mean)", size=15)
plt.grid(True)
pdf.savefig()

plt.clf()
plt.plot(result2.cov_struct.dep_params)
plt.grid(True)
plt.gca().set_xticks(range(18))
plt.xlim(0, 18)
plt.xlabel("Lag (years)", size=15)
plt.ylabel("Autocorrelation", size=15)
pdf.savefig()

for k in range(2):
    plt.clf()
    plt.axes([0.15, 0.1, 0.72, 0.86])
    if k == 0:
        plt.plot(fvals1, pred1, '-', label="10")
        plt.plot(fvals2, pred2, '-', label="11")
        plt.plot(fvals3, pred3, '-', label="12")
        plt.fill_between(fvals1, cb1[:, 0], cb1[:, 1], color='grey')
        plt.fill_between(fvals2, cb2[:, 0], cb2[:, 1], color='grey')
        plt.fill_between(fvals3, cb3[:, 0], cb3[:, 1], color='grey')
        plt.ylabel("Housing starts (log scale)", size=15)
    else:
        plt.plot(fvals1, 100*np.exp(pred1), '-', label="10")
        plt.plot(fvals2, 100*np.exp(pred2), '-', label="11")
        plt.plot(fvals3, 100*np.exp(pred3), '-', label="12")
        plt.fill_between(fvals1, 100*np.exp(cb1[:, 0]), 100*np.exp(cb1[:, 1]), color='grey')
        plt.fill_between(fvals2, 100*np.exp(cb2[:, 0]), 100*np.exp(cb2[:, 1]), color='grey')
        plt.fill_between(fvals3, 100*np.exp(cb3[:, 0]), 100*np.exp(cb3[:, 1]), color='grey')
        plt.ylabel("Housing starts (US projection)", size=15)
    ha, lb = plt.gca().get_legend_handles_labels()
    leg = plt.figlegend(ha, lb, "center right")
    leg.set_title("log pop")
    leg.draw_frame(False)
    plt.xlabel("Year", size=15)
    pdf.savefig()

nb["resid"] = result1.resid_pearson
xt = nb.pivot("fips", "year", "resid")

xm = np.asarray(pd.isnull(xt)).astype(np.int)
x0 = np.asarray(xt.fillna(value=xt.mean().mean()))

d = 3
for k in range(5):
    u,s,vt = np.linalg.svd(x0, 0)
    u = u[:, 0:d]
    s = s[0:d]
    vt = vt[0:d, :]
    xp = np.dot(u * s, vt)
    x0 = xp*xm + x0*(1-xm)

u,s,vt = np.linalg.svd(x0, 0)    

plt.clf()
plt.title("Singular values of completed residuals")
plt.plot(1960 + np.arange(len(s)), s)
plt.grid(True)
plt.xlabel("Component", size=15)
plt.ylabel("Singular value", size=15)
pdf.savefig()

plt.clf()
plt.title("Singular vectors of completed residuals")
plt.plot(vt[0,:], label="1")
plt.plot(vt[1,:], label="2")
plt.grid(True)
plt.xlabel("Year", size=15)
plt.ylabel("Singular vector loading", size=15)
ha,lb = plt.gca().get_legend_handles_labels()
leg = plt.figlegend(ha, lb, "center right")
leg.draw_frame(False)
pdf.savefig()

plt.clf()
plt.hist(u[:,0] * s[0])
plt.title("Component 1")
plt.xlabel("FIPS region coefficient", size=15)
plt.ylabel("Frequency", size=15)
pdf.savefig()

plt.clf()
plt.hist(u[:,1] * s[1])
plt.title("Component 2")
plt.xlabel("FIPS region coefficient", size=15)
plt.ylabel("Frequency", size=15)
pdf.savefig()

pdf.close()
