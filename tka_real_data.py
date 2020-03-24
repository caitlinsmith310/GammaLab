# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 17:34:40 2020

@author: caitl
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:54:57 2020

@author: caitl
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from uncertainties import ufloat
from uncertainties import unumpy
import scipy.linalg as linalg
from scipy.optimize import curve_fit
from matplotlib.patches import Polygon
from scipy.integrate import quad
import scipy.integrate as integrate
import scipy.special as special
from scipy.integrate import quad


directory =r'C:\Users\caitl\Documents\390\Gamma_ray'
os.chdir(directory)

plt.close("all")

source="Cs"  #Co or Cs
material="Al"  #Al or Pb


thickness=0
    
name=source+"_"+material+"_"+str(thickness)


chan=np.arange(0,512,1)
chan=chan[15:-1]

with open(source+"_"+material+"_"+str(thickness)+".TKA") as f:
    counts = [float(s) for s in f.readlines()]
    
counts=np.array(counts)    
counts=counts[15:-1]
counts=unumpy.uarray(counts, np.sqrt(counts))

with open('BG_'+source+'.TKA') as f:
    BG = [float(s) for s in f.readlines()]
    
BG=np.array(BG)
BG=BG[15:-1]
BG=unumpy.uarray(BG, np.sqrt(BG))    

true=counts-(BG/6)


thickness=0
def Gauss_peak(x, A, d, x0, b, c):
    y = A*np.exp(-(x-x0)**2/d**2)+b*x+c   #Defining a function for one Gaussian peak
    return y

lowerlim=257#236 238 : 279
upperlim=295#341
print("Bounds:",lowerlim,":",upperlim)
midchan = chan[lowerlim:upperlim]     #Selecting only the middle points of data
midcounts_all=unumpy.uarray(unumpy.nominal_values(true)[lowerlim:upperlim], unumpy.std_devs(true)[lowerlim:upperlim])
midcounts=unumpy.nominal_values(midcounts_all)
sy = unumpy.std_devs(midcounts_all)


pguess = [300, 250, 200, -1, 300]             #Our guess is made by varying parameters and observing the effect on the theoretical curve 
p,cov = curve_fit(Gauss_peak,midchan,midcounts,sigma=sy,p0=pguess)
yestimate = Gauss_peak(midchan , *p)
  
plt.figure("G_fit")
plt.plot(midchan, midcounts, midchan, yestimate)
plt.ylabel("Counts")
plt.xlabel("Channel Number")
#plt.savefig(name+"fig_g_fit")

chi_square = np.sum((yestimate-midcounts)**2/(sy)**2)
normalised_chi_square=chi_square/(len(midcounts) - 5)    #where did 5 come from
print("Chi Squared:", chi_square, "Normalised:", normalised_chi_square)

print("A:", p[0],"Error in A:", np.sqrt(cov[0][0]))
print("d:", p[1],"Error in d:", np.sqrt(cov[1][1]))
print("x0:", p[2],"Error in x0:", np.sqrt(cov[2][2]))

A=ufloat(p[0],np.sqrt(cov[0][0]))
d=ufloat(p[1],np.sqrt(cov[1][1]))
x0=ufloat(p[2],np.sqrt(cov[2][2]))

print("uncertainty ufloats for A, d, x0=",A,d,x0)

y=A.n*np.exp(-(chan-x0.n)**2/d.n**2)+p[3]*chan+p[4]

y_plain=A.n*np.exp(-(chan-x0.n)**2/d.n**2)
plt.figure("G_clean")
plt.plot(chan,y)
plt.fill_between(chan,p[3]*chan+p[4],y,alpha=0.2 )
#plt.plot(chan, p[3]*chan+p[4])
plt.plot(chan, unumpy.nominal_values(counts))
#plt.xlim(lowerlim-60, upperlim+60)
#plt.ylim(-10,80)
plt.ylabel("Counts")
plt.xlabel("Channel Number")
#plt.savefig(name+"fig_g_clean")


plt.figure("G_shade")
bottom=np.zeros(512)
counts_shift=counts-p[3]*chan-p[4]
plt.plot(chan,  unumpy.nominal_values(counts)-p[3]*chan-p[4])
plt.plot(chan, y_plain,color="orange")
plt.plot(chan, bottom, color="orange")
plt.xlim(lowerlim-30, upperlim+30)
#plt.ylim(-40,250)
plt.fill_between(chan, bottom,y_plain, alpha=0.2, color="orange" )
plt.savefig(name+"fig_g_shade")


area=(np.sqrt(np.pi)*A)/((d**-2)**0.5)
print("Area under gaussian, 5min=",area)

print("___________________________________________________")


    

