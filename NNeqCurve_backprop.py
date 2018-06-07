import subprocess
import numpy as np
from subprocess import call
from subprocess import PIPE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as timer


dx = 1.0e-5
alpha = 0.01
a = 1.0

Density 	= np.genfromtxt("./cool_curve1.dat", skip_header=1)[:,0]
Temp		= np.genfromtxt("./cool_curve1.dat", skip_header=1)[:,1]
Temp = np.column_stack((Temp,np.genfromtxt("./cool_curve2.dat", skip_header=1)[:,1]))
Temp = np.column_stack((Temp,np.genfromtxt("./cool_curve3.dat", skip_header=1)[:,1]))
Temp = np.column_stack((Temp,np.genfromtxt("./cool_curve4.dat", skip_header=1)[:,1]))
Temp = np.column_stack((Temp,np.genfromtxt("./cool_curve5.dat", skip_header=1)[:,1]))
ColDens 	= np.array([1.00E+18, 3.00E+18, 1.00E+19, 1.00E+20, 1.00E+21])
Ndens		= len(Density)
plot = np.zeros((1000,4))


# Setup neural network
Nlayer    = 5
Ninput    = 2
Nout      = 1
Ndata     = 1
LayerSize = 5
#values    = np.zeros((Nlayer+1,Ninput))



X         = np.zeros((Ninput))
Y         = np.zeros((Nout))
TargY     = np.zeros((Nout))

weightsM  = a*(2*np.random.random((Nlayer-2,LayerSize,LayerSize)) - np.ones((Nlayer-2,LayerSize,LayerSize)))
weightsI  = a*(2*np.random.random((Ninput,LayerSize)) - np.ones((Ninput,LayerSize)))
weightsO  = a*(2*np.random.random((LayerSize,Nout)) - np.ones((LayerSize,Nout)))
biasM	  = a*(2*np.random.random((Nlayer-2,LayerSize)) - np.ones((Nlayer-2,LayerSize)))
biasI 	  = a*(2*np.random.random(LayerSize) - np.ones(LayerSize))
biasO 	  = a*(2*np.random.random(Nout) - np.ones(Nout))



#weightsI = np.load("NNweightsI.npy")
#weightsM = np.load("NNweightsM.npy")
#weightsO = np.load("NNweightsO.npy")
#weights = np.zeros((Nlayer,Ninput,Ninput))


def acFuA(x_):
	global alpha
	x = np.copy(x_)
	for i in range(len(x)):
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
	for i in range(len(x)):
		if (x[i] < 0.0):
			x[i] = alpha
		else:
			x[i] = 1.0
	return x
"""

def d_acFuA(x_):
	x = np.copy(x_)
	for i in range(len(x)):
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
	for i in range(Nlayer-2):
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

	for k in range(Nout):
		dbO[k]  = (Y[k] - TargY[k]) * d_acFu(ZO[k])

		for lc in range(LayerSize):
			#print delta, " ", acFu(Z[Nlayer-2,lc]), " ", Z[Nlayer-2,lc]
			dWO[lc,k] 	 = dbO[k] * acFu(Z[Nlayer-2,lc])
			deltaM[lc]  += dbO[k] * weightsO[lc,k]

	#start = timer()

	# Loop over the number of layers
	for n in range(Nlayer-3, -1, -1):
		#deltaM_new 	= np.zeros(LayerSize)

		# Loop over the nodes in the current layer
		#for lc in xrange(LayerSize):

		dbM[n,:]  = d_acFuA(Z[n,:])*(deltaM[:])
		#print  " " , (d_acFuA(Z[n,:])*(deltaM[:]))
			# Loop over the nodes in the left layer
			#for li in range(LayerSize):

		dWM[n,:,:] 	= acFuA(Z[n-1,:].reshape(LayerSize,1)).dot(dbM[n].reshape(LayerSize,1).T)

		deltaM[:] 	= (weightsM[n,:,:].sum(axis=1)).dot(dbM[n,:])

		#deltaM = np.copy(deltaM_new)

	#end = timer()
	#print(end - start)

	for lc in range(LayerSize):
		#print d_acFu(Z[n,lc]) , " ", deltaM[lc], " " , Z[n,lc]
		dbI[lc]  = d_acFu(Z[n,lc])*deltaM[lc]
		for li in range(Ninput):
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

					#Achtung Aendern wenn andere Output
	dWI, dWM, dWO, dbI, dbM, dbO = derivativeW(Xin, Y, Z, ZO, TargY)
	updateWeights(dWI, dWM, dWO, dbI, dbM, dbO, dx)




	#	weightsM  = np.random.random((Nlayer-2,LayerSize,LayerSize)) #random.random
	#	weightsI  = np.random.random((Ninput,LayerSize))
	#	weightsO  = np.random.random((LayerSize,Nout))
	#	biasM	  = np.random.random((Nlayer-2,LayerSize))
	#	biasI 	  = np.random.random(LayerSize)
	#	biasO 	  = np.random.random(Nout)
	if (count == 10):
		print ("error " , 0.5*(TargY-Y)**2 ," ", TargY ," ",  Y , " " , ZO)
	#print "***"

	plot[count,0] =  Xin[0]
	plot[count,1] =  Xin[1]
	plot[count,2] =  Y
	plot[count,3] =  TargY
	return 0.5*(TargY-Y)**2 , ZO #abs(TargY-Y)/TargY #


def createInitalvalues():

	global Nlayer
	global Ninput
	global Nout
	global Ndata
	global LayerSize
	global X


	rnd1        = np.random.randint(Ndens)
	rnd2        = np.random.randint(5)
	rnd2		= 1
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


def createNewNN():
	global Temp
	global Density
	global ColDens
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
	global a

	weightsM  = a*(2*np.random.random((Nlayer-2,LayerSize,LayerSize)) - np.ones((Nlayer-2,LayerSize,LayerSize)))
	weightsI  = a*(2*np.random.random((Ninput,LayerSize)) - np.ones((Ninput,LayerSize)))
	weightsO  = a*(2*np.random.random((LayerSize,Nout)) - np.ones((LayerSize,Nout)))
	biasM	  = a*(2*np.random.random((Nlayer-2,LayerSize)) - np.ones((Nlayer-2,LayerSize)))
	biasI 	  = a*(2*np.random.random(LayerSize) - np.ones(LayerSize))
	biasO 	  = a*(2*np.random.random(Nout) - np.ones(Nout))

	error = 101
	dxl = 1.0e-8
	while error > 100:

		count = 0
		error = 0.0
		Z = 0.0

		for j in range(100):

			error_i, Z_i = backward(Temp, Density, ColDens, dxl, count)
			error 	+= error_i[0]
			Z 		 = Z_i[0]
			count 	+=1

			if (Z<0.0):
				weightsM  = a*(2*np.random.random((Nlayer-2,LayerSize,LayerSize)) - np.ones((Nlayer-2,LayerSize,LayerSize)))
				weightsI  = a*(2*np.random.random((Ninput,LayerSize)) - np.ones((Ninput,LayerSize)))
				weightsO  = a*(2*np.random.random((LayerSize,Nout)) - np.ones((LayerSize,Nout)))
				biasM	  = a*(2*np.random.random((Nlayer-2,LayerSize)) - np.ones((Nlayer-2,LayerSize)))
				biasI 	  = a*(2*np.random.random(LayerSize) - np.ones(LayerSize))
				biasO 	  = a*(2*np.random.random(Nout) - np.ones(Nout))

		error/=100
		print (error)



	print ("----Weights----")
	print (weightsM)
	print (weightsI)
	print (weightsO)
	print ("----Bias-------")
	print (biasM)
	print (biasI)
	print (biasO)


	return

print ("----Weights----")
print (weightsM)
print (weightsI)
print (weightsO)
print ("----Bias-------")
print (biasM)
print (biasI)
print (biasO)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


#createNewNN()

dx = 5.0e-6
epoch = 0
for i in range(100000):
	epoch += 1
	count = 0
	error = 0.0
	Z = 0.0

	for j in range(1000):
		error_i, Z_i = backward(Temp, Density, ColDens, dx, count)
		error 	+= error_i[0]
		Z 		+= Z_i[0]
		count 	+=1
	error/=1000

	print ("Z " , Z)
	if (Z <= 0.0):
		print ("You run into a problem!")
		#createNewNN()

	#while (error_new <= error):
	#	count +=1
	#	error = 0.0
	#	for j in xrange(100):
	#		error += backward(Temp, Density, ColDens, dx, count)
	#	error /= 100
	#	if (count == 100):
	#		break

	dx *= 0.99999
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

	print ("Error: " , error ," " , dx)
	#np.save("NNweightsI", weightsI)
	#np.save("NNweightsM", weightsM)
	#np.save("NNweightsO", weightsO)
	#np.save("BiasI", biasI)
	#np.save("BiasM", biasM)
	#np.save("BiasO", biasO)


