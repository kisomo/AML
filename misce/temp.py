# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import mxnet as mx

import openpyxl

def dx(f, x):
    return abs(0-f(x))
 
def newtons_method(f, df, x0, e):
    delta = dx(f, x0)
    while delta > e:
        x0 = x0 - f(x0)/df(x0)
        delta = dx(f, x0)
    print ('Root is at: ', x0)
    print ('f(x) at root is: ', f(x0))
    

def f(x):
    return 6*x**5-5*x**4-4*x**3+3*x**2
 
def df(x):
    return 30*x**4-20*x**3-12*x**2+6*x
    
x0s = [0, .5, 1]
for x0 in x0s:
    newtons_method(f, df, x0, 1e-5)
    

    """
Bisection, Secant & Newton Raphson Method.
"""

import math
"""
* Variable Description:
*
* f			:	Given function
* f_		:	Derivative of f
* [a, b]	:	End point values
* TOL		:	Tolerance
* NMAX		:	Max number of iterations
"""


def bisection(f, a, b, TOL=0.001, NMAX=100):
    	"""
	Takes a function f, start values [a,b], tolerance value(optional) TOL and
	max number of iterations(optional) NMAX and returns the root of the equation
	using the bisection method.
	"""
    n=1
	while n<=NMAX:
		c = (a+b)/2.0
		print("a=%s\tb=%s\tc=%s\tf(c)=%s"%(a,b,c,f(c)))
		if f(c)==0 or (b-a)/2.0 < TOL:
			return c
		else:
			n = n+1
			if f(c)*f(a) > 0:
				a=c
			else:
				b=c
	return False

def secant(f,x0,x1, TOL=0.001, NMAX=100):
	"""
	Takes a function f, start values [x0,x1], tolerance value(optional) TOL and
	max number of iterations(optional) NMAX and returns the root of the equation
	using the secant method.
	"""
	n=1
	while n<=NMAX:
		x2 = x1 - f(x1)*((x1-x0)/(f(x1)-f(x0)))
		if x2-x1 < TOL:
			return x2
		else:
			x0 = x1
			x1 = x2
	return False

def newtonraphson(f, f_, x0, TOL=0.001, NMAX=100):
	"""
	Takes a function f, its derivative f_, initial value x0, tolerance value(optional) TOL and
	max number of iterations(optional) NMAX and returns the root of the equation
	using the newton-raphson method.
	"""
	n=1
	while n<=NMAX:
		x1 = x0 - (f(x0)/f_(x0))
		if x1 - x0 < TOL:
			return x1
		else:
			x0 = x1
	return False

if __name__ == "__main__":
	
	def func(x):
	"""
	Function x^3 - x -2
	We will calculate the root of this function using different methods.
	"""
	return math.pow(x,3) - x -2

	def func_(x):
	"""
	Derivate of the function f(x) = x^3 - x -2
	This will be used in Newton-Rahson method.
	"""
		return 3*math.pow(x,2)-1
	
	#Invoking Bisection Method
	res = bisection(func,1,2).
	print(res)
	
	#Invoking Secant Method
	res = bisection(func,1,2).
	print(res)
	
	#Invoking Newton Raphson Method
	res = newtonraphson(func,func_,1)
	print(res)
 
 

 
 #ODE  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# y'' = -y + y^0.5
 
 import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np

pi = np.pi
sqrt = np.sqrt
cos = np.cos
sin = np.sin

def deriv_z(z, phi):
    u, udot = z
    return [udot, -u + sqrt(u)]

phi = np.linspace(0, 95*pi, 2000)
zinit = [1.49907, 0]
z = integrate.odeint(deriv_z, zinit, phi)
u, udot = z.T
plt.plot(phi, u)
fig, ax = plt.subplots()
ax.plot(1/u*cos(phi), 1/u*sin(phi))
ax.set_aspect('equal')
plt.grid(True)
plt.show()


#MANDELBROT SET +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


x,y=np.ogrid[-2:1:5000j,-1.5:1.5:5000j]

print('')
print('Grid set')
print('')

c=x + 1j*y
z=0

for g in range(500):
        #print('Iteration number: ',g)
        z=z**2 + c

threshold = 2
mask=np.abs(z) < threshold

print('')
print('Plotting using imshow()')
plt.imshow(mask.T,extent=[-2,1,-1.5,1.5])

print('')
print('plotting done')
print('')

plt.gray()

print('')
print('Preparing to render')
print('')

plt.show()


#+++++++++++++++++++++ FOURIER TRANSFORM   ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html

http://www.gaussianwaves.com/2015/11/interpreting-fft-results-obtaining-magnitude-and-phase-information/

http://www.gaussianwaves.com/2015/11/interpreting-fft-results-obtaining-magnitude-and-phase-information/

https://dsp.stackexchange.com/questions/18784/find-a-sinusoidal-fit-from-data-with-fft

https://ccrma.stanford.edu/~jos/resample/Theory_Ideal_Bandlimited_Interpolation.html



from numpy import fft
import numpy as np
import matplotlib.pyplot as plt
n = 10#00 # Number of data points
dx = 5.0 # Sampling period (in meters)
x = dx*np.arange(0,n) # x coordinates
w1 = 100.0 # wavelength (meters)
w2 = 20.0 # wavelength (meters)
y1 = np.sin(2*np.pi*x/w1)
plt.plot(x,y1)
y2 = 2*np.cos(2*np.pi*x/w2)
plt.plot(x,y2)
fx = np.sin(2*np.pi*x/w1) + 2*np.cos(2*np.pi*x/w2) # signal
plt.plot(x,fx)
Fk = fft.fft(fx)/n # Fourier coefficients (divided by n)
nu = fft.fftfreq(n,dx) # Natural frequencies
Fk = fft.fftshift(Fk) # Shift zero freq to center
nu = fft.fftshift(nu) # Shift zero freq to center
f, ax = plt.subplots(3,1,sharex=True)
ax[0].plot(nu, np.real(Fk)) # Plot Cosine terms
ax[0].set_ylabel(r'$Re[F_k]$', size = 'x-large')
ax[1].plot(nu, np.imag(Fk)) # Plot Sine terms
ax[1].set_ylabel(r'$Im[F_k]$', size = 'x-large')
ax[2].plot(nu, np.absolute(Fk)**2) # Plot spectral power
ax[2].set_ylabel(r'$\vert F_k \vert ^2$', size = 'x-large')
ax[2].set_xlabel(r'$\widetilde{\nu}$', size = 'x-large')
plt.show()


https://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html
http://bigdata-madesimple.com/how-to-run-linear-regression-in-python-scikit-learn/

from scipy.fftpack import fft, ifft
t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
x = np.array([1.0, 2.0, 1.0, -1.0, 1.5])
line1, = plt.plot(t,  x,'r',label='signal')
#line2, = plt.plot(x,  A[:,2],'g',label='1 day to expiry')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
plt.legend(loc='upper left',shadow=True)
plt.ylabel('signal')
plt.xlabel('time')
plt.title('signal')
plt.show()

y = np.fft.fft(x)
y
yinv = ifft(y)
yinv

n= x.size
f = np.fft.fftfreq(n)
restored_x = 0 # np.array([0.0, 0.0, 0.0, 0.0, 0.0])
for j in range(0,n):
    ampl = np.absolute(y[j])/n #amplitude
    phase = np.angle(y[j])     #phase
    restored_x += ampl*np.cos(2*np.pi*f[j]*t+phase)

#++++++++++++++++++++++++++++ COSINE RECONSTRUCTION +++++++++++++++++++++++++++++++

T = 1000
t = 0.01*np.arange(0,T)
TN = 0.01*np.arange(0,T*0.95)
N = TN.size
#y = np.cos(2*np.pi*t)
y = np.cos(t)
#plt.plot(t,y)
n = y.size
restored_y = 0.0 #np.arange(0.0,n) #*np.arange(0,n)
FT = np.fft.fft(y)
#N = np.ceil(n/2)
FREQ = np.fft.fftfreq(N)

#j=1
for j in range(0,N):
    ampl = np.absolute(FT[j])/n #amplitude
    phase = np.angle(FT[j])     #phase
    restored_y += ampl*np.cos(2*np.pi*FREQ[j]*t+phase)
    #restored_y += ampl*np.cos(FREQ[j]*t+phase)

restored_y.size
dif = y-restored_y
np.sqrt(sum(dif**2))

line1, = plt.plot(t, y,'r',label='original signal')
line2, = plt.plot(t,   restored_y,'b',label='reconstructed signal')
plt.legend(loc='top left',shadow=True)
plt.ylabel('signal')
plt.xlabel('time')
plt.title('Fourier Transform signal reconstruction')
plt.show()


#+++++++++++++++++++++++++++++ REAL PRICES +++++++++++++++++++++++++++++++++++++

import pandas as pd
import numpy as np
xl = pd.ExcelFile("H:\\WINDOWS\\system\\BLOOMBERG\\Historical prices\\Archive\\NASDAQ.xlsx")
xl.sheet_names
data1 = xl.parse("NASDAQ")
data1.shape
data1.head()

n_predict=12
n_harm=12
restored_ser = 0 #*np.arange(0,x.size)
horizon1 = data1.shape[0]-12 #-n_predict
horizon2 = data1.shape[0]   #+n_predict

dat = data1.iloc[0:horizon1:,0:2]
dat.shape
x1 = dat.iloc[:,0]
x2 = dat.iloc[:,1]
t = x1[::-1] #reverse
x = x2[::-1] #reverse
x.head()
t.shape
x.shape
t[-1:,] #print last row
x[-1:,] #print last row
#t.reshape(t.shape[1],1)
#x.reshape(x.shape[1],1)
t= t.reshape((t.size,1))
x= x.reshape((x.size,1))

from sklearn import linear_model
regr = linear_model.LinearRegression()
output_x = regr.fit(t, x)
regr.coef_
res = x - regr.predict(t)
plt.plot(t,res)
#np.mean((regr.predict(t) - x) ** 2))
d = data1[::-1] #reverse
newdat = d.iloc[horizon1+1:horizon2:,0]
newdat= newdat.reshape((newdat.size,1))
est = np.concatenate((regr.predict(t),regr.predict(newdat)))


def fourierExtrapolation(x,n_predict=12, n_harm=12):
    """
    param x: input series
    param n-predict: forecasting period
    param n_harm: number of harmonics in model
    return: restored series
    """
    n = res.size
    xf = np.fft.fft(res)
    f = np.fft.fftfreq(n)
    #indexes = range(n)
    indexes = list(range(n))
    indexes.sort(key=lambda i:np.absolute(f[i]))
    t2 = np.arange(0,n+n_predict)
    #t2 = list(range(0,n+n_predict,1))
    #global restored_ser
    for j in indexes[:1+n_harm*2]:
        ampl = np.absolute(xf[j])/n #amplitude
        phase = np.angle(xf[j])     #phase
        restored_ser += ampl*np.cos(2*np.pi*f[j]*t2+phase)
        #restored_ser += ampl*np.cos(2*np.pi*f[j]*t+phase)
    return restored_ser

    
restored_ser = restored_ser.reshape((restored_ser.size,1))
restored_x = est + restored_ser

diff = x-  restored_x
                       
print(fourierExtrapolation(x))

#+++++++++++++++++++++++++++++++++++++ REAL II ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
import pandas as pd
import pylab as pl
from numpy import fft
    
def fourierExtrapolation(x, n_predict, n_param):
    n = x.size
    n_harm = 510              # number of harmonics in model
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)         # find linear trend in x
    x_notrend = x - p[0] * t        # detrended x
    x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
    
    h=np.sort(x_freqdom)[-n_param]
    x_freqdom=[ x_freqdom[i] if np.absolute(x_freqdom[i])>=h else 0 for i in range(len(x_freqdom)) ]

    f = fft.fftfreq(n)              # frequencies
    indexes = list(range(n))
    # sort indexes by frequency, lower -> higher
    indexes.sort(key = lambda i: np.absolute(f[i]))
 
    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n   # amplitude
        phase = np.angle(x_freqdom[i])          # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig + p[0] * t
    
def main():
    x = np.array([669, 592, 664, 1005, 699, 401, 646, 472, 598, 681, 1126, 1260])
    #df = pd.read_csv("H:\WINDOWS\system\BLOOMBERG\Historical prices\SOURCE3.csv")
    df = pd.read_csv("C:\\Users\\y9ck3\Desktop\\Ten Yr Treasury Note (TNX).csv")
    print(df.shape)
    df = np.array(df)
    start = 3
    x1 = df[start:13720,3]
    x2 = df[start:13726,3]
    print(x2)
    
    '''
    n_predict = 100
    extrapolation = fourierExtrapolation(x, n_predict)
    pl.plot(np.arange(0, extrapolation.size), extrapolation, 'r', label = 'extrapolation')
    pl.plot(np.arange(0, x.size), x, 'b', label = 'x', linewidth = 3)
    pl.legend()
    pl.show()
    '''
    
    n_predict = 13
    n_param = 200
    n_back = 15
    extrapolation = fourierExtrapolation(x1, n_predict, n_param)
    pl.plot(np.arange(0, extrapolation.size), extrapolation, 'r', label = 'fitted')
    pl.plot(np.arange(0, x2.size), x2, 'b', label = 'original', linewidth = 3)
    pl.legend(loc='top left',shadow=True)
    pl.show()
    
    diffe = np.array(x2 - extrapolation)
    pl.plot(np.arange(0,x2.size), diffe, 'r', label = 'difference')
    pl.legend(loc = 'top left' , shadow = True)
    pl.show()
    print("============================ PREDICTION ======================================")
    print(extrapolation[(len(extrapolation)-n_back):len(extrapolation)])
    print("======================== ORIGINAL=============================================")
    print(x2[(len(x2)-n_back):len(x2)])
    print("===================== DIFFERENCE =============================================")
    print(diffe[(len(diffe)-n_back):len(diffe)])
    res = np.zeros((3,n_back),dtype = float)
    res[0,:] = extrapolation[(len(extrapolation)-n_back):len(extrapolation)]
    res[1,:] = x2[(len(x2)-n_back):len(x2)]
    res[2,:] =  diffe[(len(diffe)-n_back):len(diffe)]
    print(res)
    
if __name__ == "__main__":
    main()



















# KALMAN FILTER ON THE REAL PRICES +++++++++++++++++++++++++++++++++++++

https://github.com/mikemull/Notebooks/blob/master/Kalman-Slides-PyDataChicago2016.ipynb

https://stackoverflow.com/questions/13901997/kalman-2d-filter-in-python

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.api as smt

sigma_h = 10.0
h = np.random.normal(0, sigma_h, 110)
h[0] = 0.0
a = np.cumsum(h)

df = pd.DataFrame(a[0:100], columns=['a'])
_=df.plot(figsize=(14,6), style='b--')

#Now we introduce a second time series that's just the original series, plus some noise:

sigma_e = 15.
e = np.random.normal(0, sigma_e, 110)
df['y'] = a[0:100] + e[0:100]
_=df.plot(figsize=(14,6), style=['b--', 'g-',])

_=df.y.plot(figsize=(14,6), style=['g-',])

#y = a + e
#If we can only observe y, what can we say about α?
#This acts like a filter trying to recover a signal by filtering out noise.
#A linear filter.
# a is the state and y is the observation (equations)
import statsmodels.tsa.statespace.kalman_filter
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter

kf = KalmanFilter(1,1)

kf.obs_cov = np.array([sigma_e]) # H
kf.state_cov = np.array([sigma_h])  # Q
kf.design = np.array([1.0])  # Z
kf.transition = np.array([1.0])  # T
kf.selection = np.array([1.0])  # R

ys, ah = kf.simulate(100)
















https://pykalman.github.io/

$ easy_install pykalman
from pykalman import KalmanFilter
import numpy as np
kf = KalmanFilter(transition_matrices = [[1, 1], [0, 1]], observation_matrices = [[0.1, 0.5], [-0.3, 0.0]])
measurements = np.asarray([[1,0], [0,0], [0,1]])  # 3 observations
kf = kf.em(measurements, n_iter=5)
(filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
(smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)











#+++++++++++++++++++++++++++++++++++++++++ CIR/HESTON ++++++++++++++++++++++++++++++++

http://khandrikacm.blogspot.com/2014/03/european-style-interest-rate-swaption.html

http://gouthamanbalaraman.com/blog/quantlib-python-tutorials-with-examples.html

import numpy as np
import scipy as sc
import pandas as pd
from math import *
from scipy.stats import norm
import matplotlib.pyplot as plt


class Vasicek(object):
    """ VASICEK model
        mean reverting normal distributed process
    """
    def __init__(self, init, target, speed, vol):
        self.init   = init
        self.target = target
        self.speed  = speed
        self.vol    = vol

    def mean(self, maturity, init=None):
        if not(init):
            init=self.init
        return( init*np.exp(-self.speed*maturity) + self.target*(1-np.exp(-self.speed*maturity)) )

    def variance(self, maturity, init=None):
        if not(init):
            init=self.init
        return( self.vol**2/(2*self.speed)*(1-np.exp(-2*self.speed*maturity)) )

    def generate_brownian(self, maturity, nb_simulations, nb_steps):
        """ Generates normalized brownian increments
        """
        self.dB = np.random.normal(loc=0, scale=1, size=(nb_simulations, nb_steps*maturity))

    def generate_short_rates(self, maturity, nb_simulations, nb_steps):
        """ Simulation via explicit solution
        """
        self.generate_brownian(maturity, nb_simulations, nb_steps)
        res       = np.zeros((nb_simulations, nb_steps*maturity+1))
        res[:, 0] = self.init
        dt              = 1.0/nb_steps
        for j in range(res.shape[1]-1):
            res[:,j+1] = res[:,j] + self.speed*(self.target-res[:,j])*dt+self.vol*self.dB[:,j]
        self.short_rates = res
        return(res)

    def zero_coupon_price(self, rate, tau):
        """ price of a zero coupon bond when the short rate is a vasick process
            rate : short rate at current time
            tau  : time to maturity
        """
        b = (1-np.exp(-self.speed*tau))/self.speed
        a = np.exp((self.target-0.5*self.vol**2/(self.speed**2))*(b-tau)-0.25*self.vol**2*b**2/self.speed)
        return(a*np.exp(-rate*b))


        def acturial_rate(self, rate, tau):
        """ zc = exp (-actuarial_rate * tau)
        """
        return(-np.log(self.zero_coupon_price(rate,tau))/tau)

    def libor_rate(self, rate, tau):
        """ Libor rate at current time
            rate: rate at current time
            tau: time to maturity
        """
        return((1/self.zero_coupon_price(rate,tau)-1)/tau)

    def swap_rate(self, rate,tau, freq):
        return(0)

    def generate_zero_coupon_prices(self,tau):
        """
        """
        vectorized_zcprice = np.vectorize(lambda x: self.zero_coupon_price(x, tau))
        return(vectorized_zcprice(self.short_rates))

    def generate_discount_factor(self):
        """
        """
        return(0)
    def generate_libor_rates(self):
        return(0)
    def generate_swap_rate(self):
        return(0)
    def generate_discount_factor(self):
        return(0)

    def caplet_price(self):
        return(0)
    def putlet_price(self):
        return(0)

    def swaption_price(self):
        return(0)

class CIR(object):
    """ COX INGERSOL ROSS model
        mean reverting positive process
    """
    def __init__(self, init, target, speed, vol):
        self.init       = init
        self.target     = target
        self.speed      = speed
        self.vol        = vol

    def mean(self, maturity, init=None):
        """ mean of the process at maturity when initial point is init
        """
        if not(init):
            init=self.init
        return(self.target+(init-self.target)*np.exp(-self.speed*maturity))

    def variance(self, maturity, init=None):
        """ variance of the process at maturity when initial point is init
        """
        if not(init):
            init=self.init
        res = init*self.vol*self.vol*np.exp(-self.speed*maturity)*(1-np.exp(-self.speed*maturity))/self.speed
        res = res + 0.5*self.target*self.vol*self.vol*(1-np.exp(-self.speed*maturity))**2/self.speed
        return(res)

    def generate_brownian(self, maturity, nb_simulations, nb_steps):
        """ Generates normalized brownian increments
        """
        self.dB = np.random.normal(loc=0, scale=1, size=(nb_simulations, nb_steps*maturity))

    def generate_path(self, maturity, nb_simulations, nb_steps):
        """ Simulation with QE discretisation scheme
        """
        self.generate_brownian(maturity, nb_simulations, nb_steps)
        self.path       = np.zeros((nb_simulations, nb_steps*maturity+1))
        self.path[:, 0] = self.init
        dt        = 1.0/nb_steps
        for j in range(self.path.shape[1]-1):
            for i in range(self.path.shape[0]):
                m  = self.mean(dt, self.path[i,j])
                s2 = self.variance(dt, self.path[i,j])
                psi = s2/(m*m)
                if (psi<1.5):
                    b2  = 2/psi-1+np.sqrt(2/psi*(1/psi-1))
                    a   = m/(1+b2)
                    b   = np.sqrt(b2)
                    self.path[i,j+1] = a*(b+self.dB[i,j])**2
                else:
                    beta = 2/(m*(psi+1))
                    p    = (psi-1)/(psi+1)
                    u    = norm.cdf(self.dB[i,j])
                    self.path[i,j+1]  = self.__inverse_psi__(u,p, beta)
                    

    def zero_coupon_price(self, rate, tau):
        """ price of a zero coupon bond
            rate : short rate
            tau : time to maturity
        """
        h = sqrt(self.speed**2+2*self.vol**2)
        a = pow(2*h*exp(0.5*tau*(h+self.speed))/(2*h+(self.speed+h)*(exp(tau*h)-1)), 2*self.speed*self.target/self.vol**2)
        b = 2*(exp(h*tau)-1)/(2*h+(self.speed+h)*(exp(tau*h)-1))
        return(a*np.exp(-rate*b))


    def __generate_path_euler__(self, maturity, nb_simulations, nb_steps):
        """ Simulaton with Euler Scheme
        """
        self.generate_brownian(maturity, nb_simulations, nb_steps)
        self.path       = np.zeros((nb_simulations, nb_steps*maturity+1))
        self.path[:, 0] = self.init
        dt        = 1.0/nb_steps
        for j in range(self.path.shape[1]-1):
            self.path[:,j+1] = self.speed*(self.target-self.path[:,j])*dt+self.vol*np.sqrt(np.maximum(self.path[:,j],0)*dt)*self.dB[:,j]

    #help tools for CIR class
    def __inverse_psi__(self, u, p, beta):
        if 0<=u<=p:
            return(0)
        else:
            return(1/beta*np.log((1-p)/(1-u)))

class HestonModel(object):
    """ Heston Model
    """

    def __init__(self, init, rate, dividend, vol_init, target, speed, vol, correlation):
        self.init        = init
        self.rate        = rate
        self.dividend    = dividend
        self.volatility  = CIR(vol_init, target, speed, vol)
        self.correlation = correlation

    def generate_brownian(self, maturity, nb_simulations, nb_steps):
        """ Generates normalized brownian increments
        """
        self.dB = np.random.normal(loc=0, scale=1, size=(nb_simulations, nb_steps*maturity))

    def generate_path(self, maturity, nb_simulations, nb_steps):
        """ Simulation using Andersen Scheme
            source: Efficient Simulation of the Heston Stochastic Volatility Model
                    Leif Andersen
                    p19
        """
        self.volatility.generate_path(maturity, nb_simulations, nb_steps)
        self.generate_brownian(maturity, nb_simulations, nb_steps)
        self.path       = np.zeros((nb_simulations, nb_steps*maturity+1))
        self.path[:, 0] = self.init
        dt        = 1.0/nb_steps
        rootdt    = np.sqrt(dt)
        gamma1    = 0.5
        gamma2    = 0.5
        k0        = -self.correlation*self.volatility.speed*self.volatility.target*dt/self.volatility.vol
        k1        = gamma1*dt*(self.volatility.speed*self.correlation/self.volatility.vol-0.5)-self.correlation/self.volatility.vol
        k2        = gamma2*dt*(self.volatility.speed*self.correlation/self.volatility.vol-0.5)+self.correlation/self.volatility.vol
        k3        = gamma1*dt*(1-self.correlation**2)
        k4        = gamma2*dt*(1-self.correlation**2)
        for j in range(self.path.shape[1]-1):
            self.path[:,j+1]=self.path[:,j]*np.exp(
                                                    (self.rate-self.dividend)*dt+
                                                    k0+k1*self.volatility.sim[:,j]+k2*self.volatility.sim[:,j+1]
                                                    +np.sqrt(k3*self.volatility.sim[:,j]+k4*self.volatility.sim[:,j+1])*dB[:,j]
                                                   )

    def call_price(self, maturity, strike):
        """ Price of the Call Contract
        """
        def integrand(w):
            return(np.exp(-1j*w*np.log(strike*np.exp(-self.rate*maturity)/self.init))*(self.characteristic_function(w-1j, maturity)-1)/(1j*w*(1+1j*w)))
        #integral=sc.integrate.quad(integrand, -1000, 1000)[0]
        integral  = sc.integrate.trapz(integrand(np.arange(-100,100,0.1)),dx=0.1)
        res       = 0.5*self.init/np.pi*integral+np.max(0, self.init-strike*np.exp(-self.rate*maturity))
        return(float(res))

    def put_price(self, maturity, strike):
        """ Price of Put contract
            uses the call/put parity formula
        """
        return(self.call_price(maturity, strike)-self.init+strike*np.exp(-self.rate*maturity))

    def characteristic_function(self, w, maturity):
        """ characteristic function
            w: point of evaluation
            maturity: maturity
        """
        gamma = np.sqrt(self.volatility.vol**2*(w*w+1j*w)+(self.volatility.speed-1j*self.correlation*self.volatility.vol*w)**2)
        d     = -(w*w+1j*w)/(gamma*1/np.tanh(0.5*gamma*maturity)+self.volatility.speed-1j*self.correlation*self.volatility.vol*w)
        c     = self.volatility.speed*maturity*(self.volatility.speed-1j*self.correlation*self.volatility.vol*w-gamma)/(self.volatility.vol**2)
        c     = c-2*self.volatility.speed/(self.volatility.vol**2)*np.log(0.5*(1+np.exp(-gamma*maturity))+0.5*(self.volatility.speed-1j*self.correlation*self.volatility.vol*w)*(1-np.exp(-gamma*maturity))/gamma)
        return(np.exp(self.volatility.target*c+self.volatility.init*d))

class BlackScholes(object):
    """ Black Scholes model
    """

    def __init__(self, init=100.0, mu=0.05, sigma=0.20):
        """ instantiation of BlackScholes Object
        """
        self.mu    = mu
        self.sigma = sigma
        self.init  = init

    def generate_path(self, maturity, nb_simulations, nb_steps):
        """ generate_paths the model accross time and generates scenarios
            initial_value :  initial starting point of the simulations
            maturity : horizon of the simulation
            nb_simulations : number of simulations
            nb_steps : number of steps per year
            self_path  : numpy array with simulations whose dimension is (nb_simulations, nb_steps*maturity+1)
        """
        #brownian increments
        dB              = np.random.normal(loc=0, scale=1, size=(nb_simulations, nb_steps*maturity))
        self.path       = np.zeros((nb_simulations, nb_steps*maturity+1))
        self.path[:, 0] = self.init
        dt              = 1.0/nb_steps
        root_dt         = np.sqrt(dt)
        #loop over columns
        for j in range(self.path.shape[1]-1):
            self.path[:,j+1] = self.path[:,j]*np.exp(self.mu*dt+self.sigma*root_dt*dB[:,j]-0.5*self.sigma*self.sigma*dt)


    def fit_history(self, historical_values, time_interval=1.0):
        """ Calibrates model parameters from historical values
            inputs
             -------
            historical_values: historical observations of the variable
            time_interval: time betwen two observations (by default :1 year)
        """
        hv         = pd.Series(historical_values)
        hv         = np.log(hv).diff()
        self.mu    = hv.mean()/time_interval
        self.sigma = hv.std()/time_interval

    def call_price(self, spot, strike, maturity, dividend, rate, volatility):
        forward = spot*np.exp(rate*maturity)
        d1      = (np.log(forward/strike)+0.5*volatility*volatility*maturity)/(volatility*np.sqrt(maturity))
        d2      = d1-volatility*np.sqrt(maturity)
        res     = np.exp(-rate*maturity)*(forward*norm.cdf(d1)-norm.cdf(d2)*strike)
        return(res)
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        








 