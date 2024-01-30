import pandas as pd
import os

ph = pd.read_csv("phenotypes.csv")

wd = pd.read_csv("wd_cc.csv", index_col=0 )
rd = pd.read_csv("rd_cc.csv", index_col=0 )

male = 0
fem = 0

for row in wd.index:
    x = os.path.splitext(rd['0'][row])
    z = x[0].split('_')
    p = z.pop()
    for row2 in ph.index:

        if str(ph['SUB_ID'][row2]) == p:

            if ph['SEX'][row2] == 1:
                male = male + 1
            if ph['SEX'][row2] == 2:
                fem = fem + 1
            
