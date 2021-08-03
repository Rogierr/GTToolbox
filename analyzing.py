import pandas as pd
import numpy as np

sh_mb = pd.read_csv('./data/sh_mb.csv')
sh_sb = pd.read_csv('./data/sh_sb.csv')
pd_mb = pd.read_csv('./data/pd_mb.csv')
pd_sb = pd.read_csv('./data/pd_sb.csv')

pd_sb_pareto = pd.read_csv('./data/pd_sb_pareto.csv')
pd_mb_pareto = pd.read_csv('./data/pd_mb_pareto.csv')
sh_sb_pareto = pd.read_csv('./data/sh_sb_pareto.csv')
sh_mb_pareto = pd.read_csv('./data/sh_mb_pareto.csv')

for i in range(0, 11):
    print("Prisoners Dilemma with singular benefits", pd_sb.iloc[i])
    # print("Stag Hunt with mutual benefits", sh_sb.iloc[i])
# print("Stag Hunt with singular benefits", sh_sb.iloc[0])
# print("Prisoners Dilemma with mutual benefits", pd_mb.iloc[0])
# print("Prisoners Dilemma with singular benefits", pd_sb.iloc[0])

#
# print(np.min(pd_sb_pareto.iloc[:,2]))
# print(pd_mb_pareto.iloc[:,0:3])