# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 15:51:12 2024

@author: Luke

Performs Fourier Analysis on a single x,y set of data of intensity (a.u) against position
Outputs amplitude against wavelength
"""

import numpy as np
import numpy.random as npr
import scipy as sp
import pylab as pl
import scipy.fftpack as spf
import matplotlib.pyplot as plt
import read_data_results3 as rd
from scipy import signal

# Defining Gaussian
def Gau(x, mu, std, A):
    return A*np.exp((x-mu)**2/(2*std**2))


file = 'blueLED.txt'
results = rd.read_data3(file)
x = np.array(results[5])
y = np.array(results[1])

if x[len(x)-1] < 0:
    x = -x


# Now set up the experiment that you want to do
sampling_freq = 50 #Hz
motor_speed = 30000 # musteps per second
start_position = x[0] # musteps
end_position = x[len(x)-1]  # musteps


metres_per_microstep = 1.94e-11
metres_per_microstep = 1.79e-11


# set up the x-grid as seen on the interferogram
dsamp= motor_speed/sampling_freq # distance in musteps between samples 
#nsamp= int((end_position-start_position)/dsamp) # number of samples that you will take (set in the software)



nsamp = len(x)

#Centres around 0 
#x = x - (0.5*(end_position-start_position))  #Put this because just for this one we started at different position
#x = x -1500000




x = x * 2 * metres_per_microstep




#Plots amplitude against distance
plt.figure(0)
pl.plot(x,y,'bo-')
pl.xlabel("Distance from starting position(m)")
pl.ylabel("Amplitude")
#pl.xlim(-1e-6,1e-6)
pl.savefig("figures/interferogram.png")

pl.show()

print('Plotted Figure 1')

#Applies a windowing function
from scipy.signal.windows import blackman
w = blackman(nsamp,False)

#from scipy.signal.windows import kaiser
#w = kaiser(nsamp,10,False)
from scipy.signal.windows import hamming
w = blackman(nsamp)

w=1
#Take fourier transform of interferogram
yf=spf.fft(y*w)
xf=spf.fftfreq(nsamp) 

#now some shifts to make plotting easier
xf=spf.fftshift(xf)
yf=spf.fftshift(yf)

# Now try to reconstruct the original wavelength spectrum
# only take the positive part of the FT
# need to go from oscillations per step to steps per oscillation
# times by the step size

specMin = int(len(xf)/2+1)
specMax = len(xf)


#specMin = 0 # set this as we only have positives here

xx=xf[specMin:specMax]

distance = x[1:]-x[:-1]
repx = distance.mean()/xx

print(len(repx))

yaxis = abs(yf[specMin:specMax])

repx = repx[:20000]
yaxis = yaxis[:20000]

print(len(yaxis))

minLim = 300e-9
maxLim = 700e-9



#e = 0
#for i in range(len(repx)):
#    if repx[i] >= minLim and repx[i] <= maxLim:
#        newx.append(repx[i])
#        newy.append(yaxis[i])





pl.figure(1)
#pl.plot(repx,yaxis)
pl.plot(repx,yaxis)
pl.xlim(minLim,maxLim)
#pl.xlim(3e-8, 1e-7)
#pl.ylim(0,40000)



#pl.ylim(0,0.03e11)

print('Transformed')
#'''
#'''
pl.xlabel("Wavelength (m)")
pl.ylabel("Amplitude")


#pl.xlim(minLim,maxLim)

#pl.xlim(0,8e-7)


pl.savefig("figures/sim_spectrum.png")
pl.show()


pl.figure(2)

newx = []
newy = []

print('Completed Fourier Anal')

e = 0

#step = len(newx) // 1000
#step = 1
#print(step)

step = 1

flag = False

for i in range(0,len(repx),step):
    
    if repx[i] >= minLim and repx[i] <= maxLim:
        newx.append(repx[i])
        newy.append(yaxis[i])
        flag = True
    else:
        if flag == True:
            break
        
    #if repx[i] > maxLim:
    #    break
        

newx2 = []
newy2 = []    
        
print(len(newx))

step = len(newx) // 100

for i in range(0,len(newx),step):
    newx2.append(newx[i])
    newy2.append(newy[i])
    
newx = newx2
newy = newy2
        

# Fitting Gaussian to the Fourier transform of the intensity vs. position measurements
pl.plot(newx,newy,'x')
pl.xlim(minLim,maxLim)



ini_guess = [4.75e-7, 0.2e-7, 35000] # mu, std, A


fit, fit_cov = sp.optimize.curve_fit(Gau,newx,newy,ini_guess)

fit_cal = Gau(newx, *fit)
#Plots the optimised curve line.






pl.plot(newx,fit_cal)

pl.show()