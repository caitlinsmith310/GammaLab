# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:21:17 2020

@author: caitl
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 12:06:36 2020

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
import matplotlib
matplotlib.axes.Axes.errorbar
matplotlib.pyplot.errorbar

def wregress(x,y,sy):
    m1=[[np.sum(x**2/sy**2), np.sum(x/sy**2)],[np.sum(x/sy**2),np.sum(1/sy**2)]]
    m2=[[np.sum(x*y/sy**2)],[np.sum(y/sy**2)]]
    [a,b]=linalg.solve(m1,m2)
    return [a,b]

def cerror(x,y,sy):
    d=np.sum(1/sy**2)*np.sum(x**2/sy**2)-(np.sum(x/sy**2))**2
    sa=sa=np.sqrt(1/d*np.sum(1/sy**2))
    sb=np.sqrt(1/d*np.sum(x**2/sy**2))
    return sa, sb


thickness=np.array([0,1,2,3,4,5,6])

Al_width_single=ufloat(10e-3,0.01e-3)
Pb_widthlarge=ufloat(10.03e-3,0.02e-3)
Pb_widthsmall=ufloat(4.98e-3,0.03e-3)

density_Al=ufloat(2.558,0.001)
Al_width=Al_width_single*thickness
Al_thickness=Al_width*100*density_Al
Al_n=unumpy.nominal_values(Al_thickness)
Al_e=unumpy.std_devs(Al_thickness)

Al_1_n=np.array([0.0, 5.116,7.6739999999999995,10.232,12.79, 15.347999999999999])
Al_1_e=np.array([0, 0.005493037047025989, 0.008239555570538983 , 0.010986074094051978, 0.013732592617564975 , 0.016479111141077966])

print(Al_e[1])
Al_5_n=np.array([0.0, 2.558, 5.116,7.6739999999999995,10.232, 15.347999999999999])
Al_5_e=np.array([0,0.0027465185235129945, 0.005493037047025989, 0.008239555570538983 , 0.010986074094051978 , 0.016479111141077966])


Pb_t_nom=np.array([0,4.98e-3,0.01003,0.01501,0.02006,0.02504,0.03009])
Pb_t_std=np.array([0,3e-5,0.00004,0.000030,0.00004,0.00005,0.00006])
Pb_width=unumpy.uarray(Pb_t_nom,Pb_t_std)
density_Pb=ufloat(11.25,0.01)
Pb_thickness=Pb_width*100*density_Pb
Pb_n=unumpy.nominal_values(Pb_thickness)
Pb_e=unumpy.std_devs(Pb_thickness)

Cs_Pb_area=np.array([1.664e4,9.55e3,5.28e3,3.00e3,1.61e3,8.4e2,4.4e2])
Cs_Pb_error=np.array([0.026e4,0.16e3,0.18e3,0.09e3,0.09e3,0.6e2,0.8e2])
Cs_Pb=unumpy.uarray(Cs_Pb_area, Cs_Pb_error)

Cs_Al_area=np.array([1.674e4,1.156e4,9.46e3,7.89e3,6.43e3,5.35e3])   #
Cs_Al_error=np.array([0.027e4,0.024e4,0.22e3,0.18e3,0.14e3,0.14e3])  #
Cs_Al=unumpy.uarray(Cs_Al_area, Cs_Al_error)

Co_Al_area=np.array([2.67e3,2.26e3,1.82e3,1.59e3,1.25e3,9.5e2])
Co_Al_error=np.array([0.14e3,0.12e3,0.09e3,0.10e3,0.08e3,0.8e2])
Co_Al=unumpy.uarray(Co_Al_area, Co_Al_error)

Co_Pb_area=np.array([2.67e3,1.84e3, 1.26e3,9.2e2,6.7e2,4.9e2,2.8e2])
Co_Pb_error=np.array([0.14e3,0.09e3,0.08e3,0.7e2,0.7e2,0.6e2,0.5e2])
Co_Pb=unumpy.uarray(Co_Pb_area, Co_Pb_error)


'''
'''


source="Cs"
absor="Pb"

nominal=Cs_Pb_area
print(nominal)
error=Cs_Pb_error
overall=Cs_Pb
x=Pb_n
x_e=Pb_e
density=density_Pb
print(density)


plt.figure(absor+" Absorber with "+ source+" rod")
plt.plot(x,nominal, color="white")#,label="Co Source")
plt.errorbar(x ,nominal, yerr=error,capthick=1, label="Data")
plt.xlabel("Absorber thickness, $gcm^{-1}$")
plt.ylabel("Counts, 300s")
plt.legend()
plt.title(absor+" Absorber with "+ source+" rod")
plt.grid()

ln_error=error/nominal
ln_nominal=np.log(nominal)
print(ln_nominal)

c,cov=np.polyfit(x,ln_nominal,1, w=ln_error, cov=True)
a=c[0]
b=c[1]
sa=np.sqrt(cov[0][0])
sb=np.sqrt(cov[1][1])


grad=ufloat(a,sa)
inter=ufloat(b,sb)
I0=ufloat(np.exp(b),sb/b)
print("I0=",I0)
print("mass attenuation=",-grad)

yhat=grad.n*x+inter.n

plt.figure(absor+" Absorber with"+ source+" rod-fitetd")
plt.errorbar(x, ln_nominal, yerr=ln_error, label="Data")
plt.plot(x, yhat, label="Fitted data", color="red")
plt.fill_between(x,(grad.n+grad.s)*x+(inter.n+inter.s),(grad.n-grad.s)*x+(inter.n-inter.s), color="red", alpha=0.2)
plt.grid()
plt.xlabel("Absorber thickness, x ($gcm^{-1}$)")
plt.ylabel("ln(Counts, 300s)")
plt.legend()
plt.title(absor+" Absorber with "+ source+" rod")

chi_square = np.sum((yhat-ln_nominal)**2/(ln_error)**2)
normalised_chi_square=chi_square/(len(ln_nominal) - 2)    #where did 5 come from
print("Chi Squared:", chi_square)

print("lineratten:",-grad*density)





'''
'''
Na=6.002*10**23
M_Al=(0.99*26.982)+(0.005*28.085)+(0.005*24.305)
M_Pb=207.2
MoR_Cs_Al=ufloat(0.09058433464068517,0.003438801119407964)
MoR_Cs_Pb=ufloat(0.10548996113816424,0.004087725500167894)
MoR_Co_Al=ufloat(0.0830770008765312,0.006297749727999732)
MoR_Co_Pb=ufloat(0.07150660785758561,0.004291316231323125)

Sigma=(M_Pb*MoR_Co_Pb)/Na
print(Sigma)

E=0.662*10**6
e=1.6*10**(-19)
m=9.11*10**-31
c=3*10**8
#r=5.29*10**-9
r=2.13*10**-13
a=E*e/(m*c**2)
print(a)

hi=2*np.pi*r**2*((1+a/a**2)*((2*(1+a))/(1+2*a))-(np.log(1+2*a)/a)+(np.log(1+2*a)/2*a)-(1+3*a/(1+2*a)**3))
print(hi)

print(hi*82)
print(hi*13)

