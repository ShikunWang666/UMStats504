import pandas as pd
import numpy as np

# Round 2
df1 = pd.read_csv("2014_afghanistan_preliminary_runoff_election_results.csv")
df1.columns = [x.replace(" ", "_") for x in df1.columns]
df1.columns = [x.replace(".", "") for x in df1.columns]
df1.columns = [x.replace("-", "_") for x in df1.columns]

# Round 1
df2 = pd.read_csv("2014_afghanistan_election_results.csv")
df2.columns = [x.replace(" ", "_") for x in df2.columns]
df2.columns = [x.replace(".", "") for x in df2.columns]
df2.columns = [x.replace("-", "_") for x in df2.columns]
df2.columns = [x+"_pre" for x in df2.columns]

df1["ID"] = df1.PC_number + df1.PS_number / 100
df2["ID"] = df2.PC_number_pre + df2.PS_number_pre / 100

# Polling station-level data
dfps = pd.merge(df1, df2, left_on="ID", right_on="ID", how="outer")

cp = ['Eng_Qutbuddin_Hilal_pre', 'Dr_Abdullah_Abdullah_pre',
      'Zalmai_Rassoul_pre', 'Abdul_Rahim_Wardak_pre',
      'Quayum_Karzai_pre', 'Prof_Abdo_Rabe_Rasool_Sayyaf_pre',
      'Dr_Mohammad_Ashraf_Ghani_Ahmadzai_pre',
      'Mohammad_Daoud_Sultanzoy_pre',
      'Mohd_Shafiq_Gul_Agha_Sherzai_pre', 'Mohammad_Nadir_Naeem_pre',
      'Hedayat_Amin_Arsala_pre', 'Total_pre']

dp = ['Abdullah', 'Ghani', 'Total'] 
c =  dp + cp

df1 = df1.groupby("PC_number").agg(np.sum)
df1 = df1.loc[:, dp]

df2 = df2.groupby("PC_number_pre").agg(np.sum)
df2 = df2.loc[:, cp]

# Polling center-level data
dfpc = pd.merge(df1, df2, left_index=True, right_index=True, how="outer")

dfpc0 = dfpc.copy()
dfpc0[c] = dfpc0[c].fillna(0)

dfps0 = dfps.copy()
dfps0[c] = dfps0[c].fillna(0)

dfps = dfps.dropna()
dfpc = dfpc.dropna()

