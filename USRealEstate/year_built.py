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
nb.numbuilt *= 100
nb["year"] = nb["year"].astype(np.int)
nb["lognumbuilt"] = np.log(nb.numbuilt)


def absr(x):
    x = np.asarray(x).ravel()
    x = x[np.isfinite(x)]
    return np.median(np.abs(x))

xt = nb.pivot("fips", "year", "lognumbuilt")
mp_years, mp_fips = 0, 0
for k in range(3):
    print(absr(xt))
    mp_fips_0 = xt.median(axis=1)
    xt = xt.subtract(np.outer(mp_fips_0, np.ones(xt.shape[1]))) 
    mp_fips += mp_fips_0
    mp_years_0 = xt.median(axis=0)
    print(absr(xt))
    xt = xt.subtract(np.outer(np.ones(xt.shape[0]), mp_years_0))
    mp_years += mp_years_0

print(absr(xt))

1/0


nb = pd.merge(cp, nb, left_on=["fips", "year"], right_on=["fips", "year"])

model = sm.GLM.from_formula("numbuilt ~ bs(year, 10) + logpop", data=nb, family=sm.families.Poisson())
result = model.fit()

for alpha in 0.1, 0.25, 0.5, 0.75, 1:
    model = sm.GLM.from_formula("numbuilt ~ bs(year, 10) + logpop", data=nb, family=sm.families.NegativeBinomial(alpha=alpha))
    result = model.fit()
    print(result.aic)
