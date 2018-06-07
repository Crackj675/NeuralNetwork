import subprocess
import numpy as np
from subprocess import call
from subprocess import PIPE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


dx = 1.0e-5
alpha = 0.001
a = 0.1

Density 	= np.genfromtxt("./cool_curve1.dat", skip_header=1)[:,0]
Temp		= np.genfromtxt("./cool_curve1.dat", skip_header=1)[:,1]
Temp = np.column_stack((Temp,np.genfromtxt("./cool_curve2.dat", skip_header=1)[:,1]))
Temp = np.column_stack((Temp,np.genfromtxt("./cool_curve3.dat", skip_header=1)[:,1]))
Temp = np.column_stack((Temp,np.genfromtxt("./cool_curve4.dat", skip_header=1)[:,1]))
Temp = np.column_stack((Temp,np.genfromtxt("./cool_curve5.dat", skip_header=1)[:,1]))
ColDens 	= np.array([1.00E+18, 3.00E+18, 1.00E+19, 1.00E+20, 1.00E+21])
Ndens		= len(Density)
plot = np.zeros((500,4))


# Setup neural network
Nlayer    = 3
Ninput    = 2
Nout      = 1
Ndata     = 1
LayerSize = 50
#values    = np.zeros((Nlayer+1,Ninput))



X         = np.zeros((Ninput))
Y         = np.zeros((Nout))
TargY     = np.zeros((Nout))
weightsM  = a*(2*np.random.random((Nlayer-2,LayerSize,LayerSize)))# - np.ones((Nlayer-2,LayerSize,LayerSize)))
weightsI  = a*(2*np.random.random((Ninput,LayerSize)))# - np.ones((Ninput,LayerSize)))
weightsO  = a*(2*np.random.random((LayerSize,Nout)))# - np.ones((LayerSize,Nout)))
biasM	  = a*(2*np.random.random((Nlayer-2,LayerSize)))# - np.ones((Nlayer-2,LayerSize)))*a
biasI 	  = a*(2*np.random.random(LayerSize))# - np.ones(LayerSize))*a
biasO 	  = a*(2*np.random.random(Nout))# - np.ones(Nout))*a
#weightsI = np.load("NNweightsI.npy")
#weightsM = np.load("NNweightsM.npy")
#weightsO = np.load("NNweightsO.npy")
#weights = np.zeros((Nlayer,Ninput,Ninput))


def acFuA(x_):
	global alpha
	x = np.copy(x_)
	for i in xrange(len(x)):
		if (x[i] < 0.0):
			x[i] = alpha*x[i]
	return x

"""
def acFuA(x_):
	x = np.copy(x_)
	for i in xrange(len(x)):
		x[i] = np.tanh(x[i])
	return x

"""
def d_acFuA(x_):
	global alpha
	x = np.copy(x_)
	for i in xrange(len(x)):
		if (x[i] < 0.0):
			x[i] = alpha
		else:
			x[i] = 1.0
	return x
"""

def d_acFuA(x_):
	x = np.copy(x_)
	for i in xrange(len(x)):
		x[i] = 1.0/np.cosh(x[i])**2
	return x
"""
def acFu(x_):
	global alpha
	if(x_ > 0.0):
		return x_
	else:
		return alpha*x_
"""
def acFu(x_):
	return np.tanh(x_)

"""
def d_acFu(x_):
	global alpha
	if(x_ > 0.0):
		return 1.0
	else:
		return alpha


"""
def d_acFu(x_):
	return 1./(np.cosh(x_)**2)
"""

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


	#print "Xin " , Xin
	#print "-------------"
	#print weightsM
	#print Xin
	Z[0] = Xin.dot(weightsI)+biasI
	#print "Z0 " , Z[0]
	for i in xrange(Nlayer-2):
		Z[i+1] = np.dot(acFuA(Z[i]),weightsM[i])+biasM[i]
		#print Z
	ZO 	= np.dot(acFuA(Z[-1]), weightsO)+biasO
	Y 	= acFuA(ZO)
	#print Z


	return Y, Z, ZO

def derivativeW(Xin, Y, Z, ZO, TargY):

	global Nlayer
	global Nout
	global weightsM
	global weightsI
	global weightsO

	dWI 	= np.zeros(weightsI.shape)
	dWM 	= np.zeros(weightsM.shape)
	dWO 	= np.zeros(weightsO.shape)
	dbI 	= np.zeros(biasI.shape)
	dbM 	= np.zeros(biasM.shape)
	dbO 	= np.zeros(biasO.shape)
	deltaO 	= np.zeros(Nout)
	deltaM 	= np.zeros(LayerSize)
	deltaSum 	= 0.0
	delta = 0.0

	for k in xrange(Nout):
		dbO[k]  = (Y[k] - TargY[k]) * d_acFu(ZO[k])

		for lc in xrange(LayerSize):
			#print delta, " ", acFu(Z[Nlayer-2,lc]), " ", Z[Nlayer-2,lc]
			dWO[lc,k] 	 = dbO[k] * acFu(Z[Nlayer-2,lc])
			deltaM[lc]  += dbO[k] * weightsO[lc,k]


	# Loop over the number of layers
	for n in xrange(Nlayer-3, -1, -1):
		deltaM_new 	= np.zeros(LayerSize)

		# Loop over the nodes in the current layer
		for lc in xrange(LayerSize):

			dbM[n,lc]  = d_acFu(Z[n,lc])*deltaM[lc]

			# Loop over the nodes in the left layer
			for li in xrange(LayerSize):

				dWM[n,li,lc] = acFu(Z[n-1,li])*dbM[n,lc]
				deltaM_new[lc] += dbM[n,lc]*weightsM[n,li,lc]

		deltaM = np.copy(deltaM_new)


	for lc in xrange(LayerSize):
		#print d_acFu(Z[n,lc]) , " ", deltaM[lc], " " , Z[n,lc]
		dbI[lc]  = d_acFu(Z[n,lc])*deltaM[lc]
		for li in xrange(Ninput):
			dWI[li,lc] 	= dbI[lc] * acFu(Xin[li])

	#print 	dWO*dx
	#print 	dWM*dx
	#print 	dWI*dx
	#print 	dbI*dx
	#print Z
	return dWI, dWM, dWO, dbI, dbM, dbO


def updateWeights(dWI, dWM, dWO, dbI, dbM, dbO, dx):

	global weightsM
	global weightsI
	global weightsO
	global biasM
	global biasI
	global biasO

	#print "----------"
	#print weightsM
	#print dWM
	weightsI 	= weightsI - dx*dWI
	weightsM 	= weightsM - dx*dWM
	weightsO 	= weightsO - dx*dWO
	biasI 		= biasI - dx*dbI
	biasM 		= biasM - dx*dbM
	biasO 		= biasO - dx*dbO
	#print weightsM

	return

def backward(Temp, Density, ColDens, dx, count):

	global Nlayer
	global Ninput
	global Nout
	global Ndata
	global LayerSize
	global weightsM
	global weightsI
	global weightsO

	Xin, TargY			= createInitalvalues()
	Y, Z, ZO			= prediction(Xin);

	if (ZO[0] > 0.0):								#Achtung Aendern wenn andere Output
		dx *=5
	else:
		print "You run into a problem!"
	dWI, dWM, dWO, dbI, dbM, dbO = derivativeW(Xin, Y, Z, ZO, TargY)
	updateWeights(dWI, dWM, dWO, dbI, dbM, dbO, dx)


	#	weightsM  = np.random.random((Nlayer-2,LayerSize,LayerSize)) #random.random
	#	weightsI  = np.random.random((Ninput,LayerSize))
	#	weightsO  = np.random.random((LayerSize,Nout))
	#	biasM	  = np.random.random((Nlayer-2,LayerSize))
	#	biasI 	  = np.random.random(LayerSize)
	#	biasO 	  = np.random.random(Nout)
	if (count == 10):
		print "error " , 0.5*(TargY-Y)**2 ," ", TargY ," ",  Y , " " , ZO
	#print "***"

	plot[count,0] =  Xin[0]
	plot[count,1] =  Xin[1]
	plot[count,2] =  Y
	plot[count,3] =  TargY
	return 0.5*(TargY-Y)**2 #abs(TargY-Y)/TargY #


def createInitalvalues():

	global Nlayer
	global Ninput
	global Nout
	global Ndata
	global LayerSize
  	global X


	rnd1        = np.random.randint(Ndens)
	rnd2        = np.random.randint(5)
	#X[0]      	= (Density[rnd1]+1.230E-03)/(1.000E+06+1.230E-03)
	#X[1]      	= (ColDens[rnd2]-1.00E+18)/(1.00E+21-1.00E+18)
	#TargY[0]    = (Temp[rnd1, rnd2]-1.704E+01)/(5.047E+07-1.704E+01)
	#X[0]      	= (logDensity[rnd1]+1.230E-03)/(1.000E+06+1.230E-03)
	#X[1]      	= (ColDens[rnd2]-1.00E+18)/(1.00E+21-1.00E+18)
	#TargY[0]    = (Temp[rnd1, rnd2]-1.704E+01)/(5.047E+07-1.704E+01)


	X[0]      	= np.log10(Density[rnd1])-np.log10(1.230E-03)
	X[1]      	= np.log10(ColDens[rnd2])
	TargY[0]    = np.log10((Temp[rnd1, rnd2])-np.log10(1.704E+01))/np.log10(5.047E+07-1.704E+01)
	#X[0]      	= Density[rnd1]
	#X[1]      	= ColDens[rnd2]
	#TargY[0]    = Temp[rnd1, rnd2]
	#print X , " " , TargY
	return X, TargY

print "----Weights----"
print weightsM
print weightsI
print weightsO
print "----Bias-------"
print biasM
print biasI
print biasO

print "----------------------------------------"
error_new = 10e6
#for i in range(125):
error = backward(Temp, Density, ColDens, dx, 0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#dx = dx0/error
#dx = 1e-8
#for j in xrange(4000):
#	error += backward(Temp, Density, ColDens, dx, 0)
#print "phase 1"


dx = 1.0e-6
for j in xrange(50):
	backward(Temp, Density, ColDens, dx, 0)
dx = 5.0e-5
epoch = 0
for i in xrange(100000):
	epoch += 1
	count = 0
	error = 0.0



	for j in xrange(500):
		error += backward(Temp, Density, ColDens, dx, count)
		count +=1
	error/=500

	#while (error_new <= error):
	#	count +=1
	#	error = 0.0
	#	for j in xrange(100):
	#		error += backward(Temp, Density, ColDens, dx, count)
	#	error /= 100
	#	if (count == 100):
	#		break

	dx *= 0.9999
	#dx = dx0/error
	#dx = min(0.99*dx, error*10e-6)
	#error_new = error
	#dx = min(dx, error*10e-6) #+ 0.001*dx
	if (epoch%20==0):
		ax.clear()
		ax.scatter(plot[:,0],plot[:,1],plot[:,2])
		ax.scatter(plot[:,0],plot[:,1],plot[:,3], color="r")
		plt.pause(1e-17)
		plt.draw()

	print error ," " , dx
	print "----Weights----"
	print weightsM
	print weightsI
	print weightsO
	print "----Bias-------"
	print biasM
	print biasI
	print biasO
	#print "Error: " , error ," " , dx
	#np.save("NNweightsI", weightsI)
	#np.save("NNweightsM", weightsM)
	#np.save("NNweightsO", weightsO)
	#np.save("BiasI", biasI)
	#np.save("BiasM", biasM)
	#np.save("BiasO", biasO)

print "----------------------------------------"


	#if (i%10 == 0):
 		#print backward(Temp, Density, ColDens, dx)
		#print dx
		#print biasO
	#backward(Temp, Density, ColDens, dx)

print "----Weights----"
print weightsM
print weightsI
print weightsO
print "----Bias-------"
print biasM
print biasI
print biasO
