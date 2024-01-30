import numpy as np
import pandas as pd

directory = r'C:\rois_cc400_CPAC'

pds = pd.read_csv('phenotype stuff.csv') 
x = 0
am = 0
af = 0
ageam = 0
ageaf = 0
cm = 0
cf = 0
agecm = 0
agecf = 0

for i in pds.index:

        if pds['DX_GROUP'][x] == 1:
            if pd['SEX'][x] == 1:
                am = am + 1
                ageam = ageam + pds['AGE_AT_SCAN'][x]
                
        if pds['DX_GROUP'][x] == 2:
            if pd['SEX'][x] == 1:
                cm = cm + 1
                agecm = agecm + pds['AGE_AT_SCAN'][x]
            
        if pds['DX_GROUP'][x] == 1:
            if pd['SEX'][x] == 2:
                af = af + 1
                agecm = agecm + pds['AGE_AT_SCAN'][x]
                    
        if pds['DX_GROUP'][x] == 2:
            if pd['SEX'][x] == 2:
                cf = cf + 1
                agecf = agecf + pds['AGE_AT_SCAN'][x]
                

        x = x + 1
    
print('ASD male:', am)
print('ASD fem:', af)
if am+af != 0:
    print('ASD age:', (ageam + ageaf)/(am+af))
else :
    print('ASD age: n/a')

print('C male:', cm)
print('C female:', cf)
if cm + cf != 0:
    print('C age:', (agecf + agecm)/(cm+cf))
else :
    print('C age: n/a')
