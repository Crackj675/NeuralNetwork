import subprocess
import numpy as np
from subprocess import call
from subprocess import PIPE

narg = 7
np0 = 1
nm0 = 1
dx = 0.1

Density 	= np.genfromtxt("./cool_curve1.dat", skip_header=1)[:,0]
Temp		= np.genfromtxt("./cool_curve1.dat", skip_header=1)[:,1]
Temp = np.column_stack((Temp,np.genfromtxt("./cool_curve2.dat", skip_header=1)[:,1]))
Temp = np.column_stack((Temp,np.genfromtxt("./cool_curve3.dat", skip_header=1)[:,1]))
Temp = np.column_stack((Temp,np.genfromtxt("./cool_curve4.dat", skip_header=1)[:,1]))
Temp = np.column_stack((Temp,np.genfromtxt("./cool_curve5.dat", skip_header=1)[:,1]))
ColDens 	= np.array([1.00E+18, 3.00E+18, 1.00E+19, 1.00E+20, 1.00E+21])
Ndens		= len(Density)

# Setup neural network
Nlayer    = 4
Ninput    = 2
Noutout   = 1
Ndata     = 10
LayerSize = 3
#values    = np.zeros((Nlayer+1,Ninput))


X         = np.zeros((Ndata, Ninput))
Y         = np.zeros((Ndata,Noutout))
TargY     = np.zeros((Ndata,Noutout))
#weightsM  = np.ones((Nlayer-2,LayerSize,LayerSize))*0.1
#weightsI  = np.ones((Ninput,LayerSize))*0.1
#weightsO  = np.ones((LayerSize,Noutout))*0.1
weightsI = np.load("NNweightsI.npy")
weightsM = np.load("NNweightsM.npy")
weightsO = np.load("NNweightsO.npy")
#weights = np.zeros((Nlayer,Ninput,Ninput))
#print weights

def relu(x):
	#return max(0,x)
	return np.tanh(x)

def prediction(Xin):

	global Nlayer
	global Ninput
	global weightsM
	global weightsI
	global weightsO

	Values = Xin.dot(weightsI)
	for i in range(Nlayer-2):
		Values_new = relu(np.dot(Values,weightsM[i]))
		Values = np.copy(Values_new)
	Values = Values.dot(weightsO)

	return Values

def metropolis(Temp, Density, ColDens, dx):

	global Nlayer
	global Ninput
	global Noutout
	global Ndata
	global LayerSize
	global weightsM
	global weightsI
	global weightsO

	Xin, TargY		= createInitalvalues()
	pred_old 		= prediction(Xin);
	diff_old 		= calcDiff(pred_old, TargY)
	#if (pred_old == 0):
	#	nm = nm+1
	#	return False


  # Metropolis step
  #-------------------------------------------
	rnd1 = np.random.randint(Nlayer)
	rnd2 = np.random.randint(LayerSize)
	rnd3 = np.random.randint(LayerSize)
	rnd4 = dx*(-1 + np.random.rand()*2)

	if (rnd1 == 0):
		weightsI[int(rnd2/LayerSize*Ninput), rnd3] += rnd4
	elif(rnd1 == Nlayer-1):
		weightsO[rnd2, int(rnd3/LayerSize*Noutout)] += rnd4
	else:
		weightsM[rnd1-1,rnd2,rnd3] += rnd4

  # New prediction
  #-------------------------------------------
	pred_new = prediction(Xin)
	#if (pred_new == 0):
	#	weightsM[rnd1,rnd2,rnd3] -= rnd4
	#	return False

	diff_new = calcDiff(pred_new, TargY)
	#print "diff_new " , diff_new , " diff_old " , diff_old , " " , diff_old/diff_new
	if (diff_new < diff_old):
		#print weightsM
		#print weightsI
		#print weightsO
		return True
	else:
		if (np.random.rand() < 1.0-100*(diff_old/diff_new)):
			#print "case 2: " , np.random.rand() ," " , diff_old/diff_new
			return True
		else:
			if (rnd1 == 0):
				weightsI[int(rnd2/LayerSize*Ninput), rnd3] += rnd4
			elif(rnd1 == Nlayer-1):
				weightsO[rnd2, int(rnd3/LayerSize*Noutout)] += rnd4
			else:
				weightsM[rnd1-1,rnd2,rnd3] -= rnd4
	return False

def calcDiff(pred_old, TargY):
	global Ndata
	error = 0.

	for i in range(Ndata):
		error += abs(pred_old[i] - TargY[i])
	return error/Ndata

def createInitalvalues():

	global Nlayer
	global Ninput
	global Noutout
	global Ndata
	global LayerSize
  	global X

	for i in range(Ndata):
		rnd1        = np.random.randint(Ndens)
		rnd2        = np.random.randint(5)
		X[i,0]      = np.log10(Density[rnd1])
		X[i,1]      = np.log10(ColDens[rnd2])
		TargY[i]    = np.log10(Temp[rnd1, rnd2])

	return X, TargY

for i in range(1000):
	dx *= 0.99
	for j in range(int(1e5)):
	#print "------------------------------------------------------------------"

		nx = metropolis(Temp, Density, ColDens, dx)
		if (nx):
			np0 += 1.0
		else:
			nm0 += 1.0
	print np0/nm0
	print "------------------------------------------------------------------"
	print weightsM
	print "------------------------------------------------------------------"
	Xin, TargY		= createInitalvalues()
	pred 	 		= prediction(Xin);
	diff 	 		= calcDiff(pred, TargY)
	print "Diff : " , diff
	print "dx   : " , dx
	np.save("NNweightsI", weightsI)
	np.save("NNweightsM", weightsM)
	np.save("NNweightsO", weightsO)
"""
#********************************
# Species list with their indexes
1	E	krome_idx_E
2	H2	krome_idx_H2
3	H	krome_idx_H
4	CO	krome_idx_CO
5	H2+	krome_idx_H2j
6	H3+	krome_idx_H3j
7	HCO+	krome_idx_HCOj
8	CR	krome_idx_CR
9	g	krome_idx_g
10	Tgas	krome_idx_Tgas
11	dummy	krome_idx_dummy

#********************************
# Species in a Python list of strings
["E", "H2", "H", "CO", "H2+", "H3+", "HCO+", "CR", "g", "Tgas", \
 "dummy", ]


#********************************
#useful parameters
krome_nrea           = 4    !number of reactions
krome_nmols          = 7    !number of chemical species
krome_nspec          = 11   !number of species including Tgas,CR,...
krome_ndust          = 0    !number of dust bins (total)
krome_ndustTypes     = 0    !number of dust types
krome_nPhotoBins     = 0    !number of radiation bins
krome_nPhotoRates    = 0    !number of photochemical reactions


#********************************
#list of reactions (including with multiple limits)
1	H2 -> H2+ + E
2	H2 + H2+ -> H3+ + H
3	H3+ + CO -> HCO+ + H2
4	HCO+ + H2 -> H3+ + CO
"""
