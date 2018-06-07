import subprocess
import numpy as np
from subprocess import call
from subprocess import PIPE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

narg = 7
np0 = 1
nm0 = 1
dx = 1.0e-5

Density 	= np.genfromtxt("./cool_curve1.dat", skip_header=1)[:,0]
Temp		= np.genfromtxt("./cool_curve1.dat", skip_header=1)[:,1]
Temp = np.column_stack((Temp,np.genfromtxt("./cool_curve2.dat", skip_header=1)[:,1]))
Temp = np.column_stack((Temp,np.genfromtxt("./cool_curve3.dat", skip_header=1)[:,1]))
Temp = np.column_stack((Temp,np.genfromtxt("./cool_curve4.dat", skip_header=1)[:,1]))
Temp = np.column_stack((Temp,np.genfromtxt("./cool_curve5.dat", skip_header=1)[:,1]))
ColDens 	= np.array([1.00E+18, 3.00E+18, 1.00E+19, 1.00E+20, 1.00E+21])
logColDens     = np.zeros((5, len(Density)))
logTemp     = np.zeros((5, len(Density)))
logDensity  = np.zeros(len(Density))
Ndens		= len(Density)

for i in xrange(Ndens):
	for j in range(5):
		logColDens[j,i] = np.log10(ColDens[j])
		logTemp[j,i] = np.log10(Temp[i,j])
	logDensity[i] = np.log10(Density[i])

# Load NeuralNetwork
weightsI  = np.load("NNweightsI.npy")
weightsM  = np.load("NNweightsM.npy")
weightsO  = np.load("NNweightsO.npy")
biasI     = np.load("BiasI.npy")
biasM     = np.load("BiasM.npy")
biasO     = np.load("BiasO.npy")

#print weightsM
Nlayer    = weightsM.shape[0]+2
Ninput    = weightsI.shape[0]
Nout      = weightsO.shape[1]
LayerSize = weightsM.shape[1]

print
print "Nlayer    ", Nlayer
print "Ninput    ", Ninput
print "Nout      ", Nout
print "LayerSize ", LayerSize

ndata = 1000
X1 = np.zeros(ndata)
X2 = np.zeros(ndata)
Y = np.zeros(ndata)
#weights = np.zeros((Nlayer,Ninput,Ninput))
#print weights

def acFuA(x_):
	x = np.copy(x_)
	for i in xrange(len(x)):
		if (x[i] < 0.0):
			x[i] = 0.0
	#print "test " , x_
	return x
	#return np.tanh(x_)

def d_acFuA(x_):
	x = np.copy(x_)
	for i in xrange(len(x)):
		if (x[i] < 0.0):
			x[i] = 0.0
		else:
			x[i] = 1.0
	return x
	#return 1./(np.cosh(x_)**2)


def prediction(Xin):

	global Nlayer
	global Ninput
	global Nout
	global LayerSize
	global weightsM
	global weightsI
	global weightsO
	global biasM
	global biasI
	global biasO

	Z 	= np.zeros((Nlayer-1,LayerSize))
	ZO  = np.zeros((Nout))
	A 	= np.zeros((Nlayer,LayerSize))
	Y 	= np.zeros((Nout))


	Z[0] = Xin.dot(weightsI)+biasI

	for i in range(Nlayer-2):
		Z[i+1] = np.dot(acFuA(Z[i]),weightsM[i])+biasM[i]
	ZO 	= acFuA(Z[-1]).dot(weightsO)+biasO
	Y 	= acFuA(ZO)

	return Y



def createInitalvalues():

	global Nlayer
	global Ninput
	global Nout
	global Ndata
	global LayerSize

	#x1      	= (np.log10(1.000E+06)-np.log10(1.230E-03))*np.random.rand()+np.log10(1.230E-03)
	#x2      	= (np.log10(1.00E+21)-np.log10(1.00E+18))*np.random.rand()+np.log10(1.00E+18)

	x1 = np.random.random()
	x2 = np.random.random()

	return x1, x2


	#X[0]      	= (Density[rnd1]+1.230E-03)/(1.000E+06+1.230E-03)
	#X[1]      	= (ColDens[rnd2]-1.00E+18)/(1.00E+21-1.00E+18)
	#TargY[0]    = (Temp[rnd1, rnd2]-1.704E+01)/(5.047E+07-1.704E+01)

for i in range(ndata):
	X1[i], X2[i] = createInitalvalues()
	#X = np.array(([(X1[i]+1.230E-03)/(1.000E+06+1.230E-03), (X2[i]-1.00E+18)/(1.00E+21-1.00E+18)]))
	y = prediction(np.array([X1[i],X2[i]]))
	#print y
	#Y[i] = np.log10(y*(5.047E+07-1.704E+01)+1.704E+01)
	Y[i] = y


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1,X2,Y)
#ax.plot(logDensity, logColDens[0], zs = logTemp[0])
#ax.plot(logDensity, logColDens[1], zs = logTemp[1])
#ax.plot(logDensity, logColDens[2], zs = logTemp[2])
#ax.plot(logDensity, logColDens[3], zs = logTemp[3])
#ax.plot(logDensity, logColDens[4], zs = logTemp[4])
plt.show()
