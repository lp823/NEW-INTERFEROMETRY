"""
This code simulates an interferogram produced by various spectral sources.
Note: all lengths are given in metres; light amplitude and intensity are in arbitrary units
"""

import numpy as np
import numpy.random as npr
import scipy as sp
import pylab as pl
import scipy.fftpack as spf

def calc_gaussian_amp(nsigma,nmodes):
    """
    Calculates the amplitude of the various modes used to create a Gaussian over a
    range of 2 * nsigma standard deviations
    """
    yy=np.empty(shape=[nmodes])
    step=nsigma*2.0/nmodes
    for i in range(nmodes):
        x=-nsigma+i*step
        height=np.exp(-x*x/2)
        yy[i]=height
    return yy

def add_gaussian(x,y,wl,amp,sigma,nmodes):
    """
    This function adds the effect of a Gaussian line or spectrum to the interferogram.
    It does this by assuming that each Gaussian is made up of lots of discrete delta functions 
    and calculates to +/- 5 sigma
    Inputs:
    x is the mirror separation in the interferometer
    y is the amplitude of the interferogram (may start at zero or add to existing)
    wl is the wavelength
    amp is the amplitude of the Gaussian
    sigma is the standard deviation (width) of the Gaussian
    nmodes is the number of discrete modes used
    """
    
    nsigma=5 # the number of std devs over which the amplitude is calculated before cutting off
    amplitude=(amp*calc_gaussian_amp(nsigma,nmodes))
    
    amplitude=amp*amplitude/sum(amplitude)
    wl_step=nsigma*2.0*sigma/nmodes
    # construct the interferogram from the individual modes
    for i in range(len(amplitude)):
        wavelength=wl-nsigma*sigma+i*wl_step
        y=y+amplitude[i]*np.sin(2.0*np.pi*x/wavelength) + amplitude[i]       
    return y


  
def add_square(x,y,start,amp,width,nmodes):
    """
    This function adds the effect of a square (top-hat) spectrum to the interferogram.
    Inputs:
    x is the mirror separation separation in the interferometer
    y is the amplitude of the interferogram (may start at zero or add to existing)
    start is the wavelength to start the square from
    width is the width of the square (so it goes from wavelengths start:start+width)
    amp is the amplitude
    nmodes is the number of discrete modes used
    """
    step=width/(nmodes-1)
    amplitude=amp/nmodes
    for i in range(nmodes):
        # as we know the amplitude is constant we don't need a separate function to calculate it
        wavelength=start+i*step
        y=y+amplitude*np.sin(np.pi*2.*x/wavelength) + amplitude
    return y
    



# Describe the global calibration used
metres_per_microstep = 2e-11

# Now set up the experiment that you want to do
sampling_freq = 50 #Hz
motor_speed = 30000 # musteps per second
start_position = -1000000 # musteps
end_position = -start_position # musteps




# set up the x-grid as seen on the interferogram
dsamp= motor_speed/sampling_freq # distance in musteps between samples 

nsamp= int((end_position-start_position)/dsamp) # number of samples that you will take (set in the software)

# set up the number of modes to use under the spectral components
nmodes = 500

# construct the x-grid of the interferogram
# include factor of 2 to properly simulate path difference
x= np.linspace(start_position,end_position,nsamp) * 2.0*metres_per_microstep 
# and set the y-values to zeros
y= np.zeros(len(x)) #setting the array that will contain your results


# add a square wave
start_wavelength = 505e-9 # metres
width = 1e-9 # metres
amp = 1.0 # amplitude
y = add_square(x,y,start_wavelength,amp,width,nmodes)

wl_1 = 500e-9 # central wavelength of the Gaussian
sigma_1 = 1e-9 # standard deviation of the Gaussian
amp_1=1.0
y=add_gaussian(x,y,wl_1,amp_1,sigma_1,nmodes)

# plot the output

pl.figure(1)
pl.plot(x,y,'bo-')
pl.xlabel("Distance from null point (m)")
pl.ylabel("Amplitude")
#pl.xlim(-6e-6,6e-6)
pl.savefig("figures/interferogram.png")

# quick check by Fourier transforming the output

# take a Fourier transform
yf=spf.fft(y)
xf=spf.fftfreq(nsamp) # setting the correct x-axis for the Fourier transform. Oscillations/step

#now some shifts to make plotting easier
xf=spf.fftshift(xf)
yf=spf.fftshift(yf)


# Now try to reconstruct the original wavelength spectrum
# only take the positive part of the FT
# need to go from oscillations per step to steps per oscillation
# times by the step size
xx=xf[int(len(xf)/2+1):len(xf)]

distance = x[1:]-x[:-1]
repx = distance.mean()/xx


pl.figure(3)


yaxis = abs(yf[int(len(xf)/2+1):len(xf)])

pl.plot(repx,yaxis)
pl.xlabel("Wavelength (m)")
pl.ylabel("Amplitude")
pl.xlim(0e-9,1000e-9)
pl.savefig("figures/sim_spectrum.png")
pl.show()
