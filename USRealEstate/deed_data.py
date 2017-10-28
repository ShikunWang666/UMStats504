import pandas as pd
import numpy as np

dpath = "/nfs/turbo/arcts-dads-corelogic/Data/deed/0003.gz"

dtp = {"SALE DATE (YYYYMMDD)": str}

rdr = pd.read_csv(dpath, delimiter="|", chunksize=200000, low_memory=False, 
                  dtype=dtp)

# Loop over sub-chunks
dat = []
while True:

    try:
        df = rdr.get_chunk()
    except StopIteration:
        break

    df["SALE DATE (YYYYMMDD)"] =\
           pd.to_datetime(df["SALE DATE (YYYYMMDD)"], format="%Y%m%d", errors='coerce')

    dx = df[["APN (Parcel Number) (unformatted)", "SALE DATE (YYYYMMDD)", "SALE AMOUNT", "MORTGAGE AMOUNT",
             "RESALE/NEW CONSTRUCTION", "RESIDENTIAL MODEL INDICATOR", "CASH/MORTGAGE PURCHASE",
             "FORECLOSURE", "FIPS", "TRANSACTION TYPE"]]
    ii = pd.notnull(dx[["APN (Parcel Number) (unformatted)", "SALE DATE (YYYYMMDD)", "CASH/MORTGAGE PURCHASE",
                        "TRANSACTION TYPE"]]).all(1)
    dx = dx.loc[ii, :]

    # Only retain records that are for "arms length sales"
    dx = dx.loc[dx["TRANSACTION TYPE"] == 1, :]

    # Convert to a number (days since 1960-01-01)
    dx["SALE DATE (YYYYMMDD)"] -= pd.to_datetime("1960-01-01")
    dx["SALE DATE (YYYYMMDD)"] = dx["SALE DATE (YYYYMMDD)"].dt.days
    dx["SALE DATE (YYYYMMDD)"] = dx["SALE DATE (YYYYMMDD)"].astype(np.float64)

    # Drop non-residential properties
    dx = dx.loc[dx["RESIDENTIAL MODEL INDICATOR"] == "Y"]

    # Drop properties with only one record
    gb = dx.groupby("APN (Parcel Number) (unformatted)")
    nr = pd.DataFrame(gb.size())
    nr.columns = ["numrecs"]
    dx = pd.merge(dx, nr, left_on="APN (Parcel Number) (unformatted)", right_index=True)
    dx = dx.loc[dx.numrecs > 1]
    
    dx = dx.loc[:, ["APN (Parcel Number) (unformatted)", "SALE DATE (YYYYMMDD)", "SALE AMOUNT", "FIPS", "CASH/MORTGAGE PURCHASE"]]

    dat.append(dx)

deed = pd.concat(dat, axis=0)
