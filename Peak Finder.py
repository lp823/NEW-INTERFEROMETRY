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

#def envelope(x, x0, k, A, s, o):
#    f1 = A*np.exp(-((x-x0)**2)/2*s)
#    f2 = (np.cos(k*(x-x0)))**2
#    return f1*f2 + o

def envelope(x, x0, k, A, s, o):
    f1 = A*np.exp(-(np.abs(x-x0)/2*s))
    f2 = (np.cos(k/2*(x-x0)))**2
    return f1*f2 + o

def parabola(x,a,x0,b):
    return -a*(x-x0)**2+b

file = 'green_laser/laser2.txt'
results = rd.read_data3(file)
x = np.array(results[5])
y = np.array(results[1])





# Now set up the experiment that you want to do
sampling_freq = 50 #Hz
motor_speed = 30000 # musteps per second
start_position = x[0] # musteps
end_position = x[len(x)-1]  # musteps


metres_per_microstep = 1.94e-11
#metres_per_microstep = 1.79e-11

#metres_per_microstep = 1.79e-11


# set up the x-grid as seen on the interferogram
dsamp= motor_speed/sampling_freq # distance in musteps between samples 
#nsamp= int((end_position-start_position)/dsamp) # number of samples that you will take (set in the software)



nsamp = len(x)

#Centres around 0 
#x = x - (0.5*(end_position-start_position))  #Put this because just for this one we started at different position
#x = x -1500000




x = x * 2 * metres_per_microstep




#Plots amplitude against distance

#pl.xlim(0.001015,0.00103)

pl.figure(0)

pl.plot(x,y,'bo-')
pl.xlabel("Distance from starting position(m)")
pl.ylabel("Amplitude")
pl.savefig("figures/interferogram.png")

#pl.xlim(0.005,0.00501)

print('Plotted Figure 1')


#Now analyse the peaks (this is peak)

peaks, _ = sp.signal.find_peaks(y)


peaks2, _ = sp.signal.find_peaks(y[peaks])

y_envelope = y[peaks[peaks2]]
x_envelope = x[peaks[peaks2]]

plt.plot(x_envelope,y_envelope,'x')

envMaxima = x_envelope[sp.signal.argrelmax(y_envelope, order = 25)]

#print(peaks2)

#plt.plot(envMaxima, [0]*len(envMaxima), 'x')


#Tries to fit to a function

#ini_guess = [0.0045, (2*np.pi)/0.005, 2000, 1e6, 2000] # x0, k, A, s,o

ini_guess = [0.0045, (2*np.pi)/0.005, 2000, 1e3, 2000] # x0, k, A, s,o
par_guess = [100000, 0.0015, 3500] #  a, x0, b

parax = []
paray = []

for i in range(len(x_envelope)-1):
    if x_envelope[i] > 0.001 and x_envelope[i] < 0.002:
        parax.append(x_envelope[i])
        paray.append(y_envelope[i])

ffit, ffit_cov = sp.optimize.curve_fit(parabola, parax, paray, par_guess)
paray = parabola(parax, *ffit)
plt.plot(parax,paray)

print(ffit[1])
t1=ffit[1]


par_guess = [100000, 0.005, 3500] #  a, x0, b
parax = []
paray = []

for i in range(len(x_envelope)-1):
    if x_envelope[i] > 0.0025 and x_envelope[i] < 0.0035:
        parax.append(x_envelope[i])
        paray.append(y_envelope[i])

ffit, ffit_cov = sp.optimize.curve_fit(parabola, parax, paray, par_guess)
paray = parabola(parax, *ffit)
plt.plot(parax,paray)

print(ffit[1])
t2 = ffit[1]

par_guess = [100000, 0.003, 3500] #  a, x0, b
parax = []
paray = []

for i in range(len(x_envelope)-1):
    if x_envelope[i] > 0.004 and x_envelope[i] < 0.0053:
        parax.append(x_envelope[i])
        paray.append(y_envelope[i])

ffit, ffit_cov = sp.optimize.curve_fit(parabola, parax, paray, par_guess)
paray = parabola(parax, *ffit)
plt.plot(parax,paray)


print(ffit[1])
t3 = ffit[1]


#tArray =[t1,t2,t3]

#%%

c = 299792458

lambB = [t2-t1,t3-t2]

lam1 = t2-t1
lam2 = t3-t2
lamb = np.mean(lambB)

std = np.std(lambB)
print(lamb)
print('Standard deviation: '+ str(std))
print('Standard error '+str(std/np.sqrt(2)))


olam1 = 2.9676e-11 #error in the twin gaussian fit in wavelength domain


olam2 = lam2**2 * np.sqrt(lam1**4 *olam1  + lamb**4 * std**2)





lambda1 = 5.31967e-07 #Fourier transform's value of 532 wavelength
lambda2 = 5.31915e-07 #What fourier transform thinks secondary wavelength is
errorlambda2 = 1.75651e-10 # The error in calculated secondary wavelength

Ulambda1 = 532e-9 #Actual wavelength
Ulambda2 =  0.00163798029969942 #Wavelength we got from Beat Analysis


Ulambda2 = 1/(1/Ulambda1 + 1/lamb) 

olam2 = Ulambda2**2 * np.sqrt(Ulambda1**4 *olam1**2  + lamb**4 * std**2)


print('')
print('We found that the lambda 2 value, using beat analysis was: ')
print(Ulambda2)
print('It had uncertainty of:')
print(olam2)
print('')


u1 = c/lambda1
u2 = c/lambda2
du = abs(u1 - u2)

U1 = c/Ulambda1
U2 = c/Ulambda2
dU = abs(U1 - U2)
print(dU)

ou1 = (u1/lambda1)*olam1
ou2 = (u2/lambda2)*errorlambda2

oU1 =(U1/Ulambda1)*olam1
oU2 = (U2/Ulambda2)*olam2


oDelV = np.sqrt(ou1**2+ou2**2)
UoDelV = np.sqrt(oU1**2+oU2**2)

uLo = c/(2*du)
ULo = c/(2*dU)

ouLo = (uLo/du)*oDelV
oULo = (ULo/dU)*UoDelV

print('Answer using beat analysis')
print(ULo)
print('With uncertainty')
print(oULo)
print('')



print('Answer from Fourier Twin Peaks')
print(uLo)
print('With uncertainty')
print(ouLo)
print('')













#%%


fit, fit_cov = sp.optimize.curve_fit(envelope, x_envelope, y_envelope, ini_guess)

fit_cal = envelope(x_envelope, *fit)

#pl.plot(x_envelope,fit_cal)




pl.show()

#Applies a windowing function

#Take fourier transform of interferogram
yf=spf.fft(y)
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


yaxis = abs(yf[specMin:specMax])

#repx = repx[:20000]
#yaxis = yaxis[:20000]


minLim = 300e-9
maxLim = 900e-9

newx = []
newy = []

pl.figure(1)
#pl.plot(repx,yaxis)
pl.plot(repx,yaxis)
#pl.xlim(minLim,maxLim)
#pl.xlim(3e-8, 1e-7)
#pl.ylim(0,35000)



#pl.ylim(0,0.03e11)

print('Transformed')
#'''
# Fitting Gaussian to the Fourier transform of the intensity vs. position measurements
#ini_guess = [5.5e-7, 0.1e-7, 3e+6] # mu, std, A
#fit, fit_cov = sp.optimize.curve_fit(Gau,repx,yaxis,ini_guess)
#Creates an array of y values for the fit
#fit_cal = Gau(repx,fit)
#Plots the optimised curve line.
#plt.plot(repx,fit_cal)
#'''
pl.xlabel("Wavelength (m)")
pl.ylabel("Amplitude")


pl.xlim(minLim,maxLim)

#pl.xlim(0,8e-7)


pl.savefig("figures/sim_spectrum.png")
pl.show()

print('Completed Fourier Anal')

pl.figure(2)

yf=spf.fft(y_envelope)
xf=spf.fftfreq(len(x_envelope)) 

#now some shifts to make plotting easier
xf=spf.fftshift(xf)
yf=spf.fftshift(yf)

specMin = int(len(xf)/2+1)
specMax = len(xf)

xx=xf[specMin:specMax]

distance = x_envelope[1:]-x_envelope[:-1]
repx = distance.mean()/xx

yaxis = abs(yf[specMin:specMax])



pl.plot(repx,yaxis)

pl.show()