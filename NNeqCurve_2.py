import subprocess
import numpy as np
from subprocess import call
from subprocess import PIPE

narg = 7
np0 = 1
nm0 = 1

Density 	= np.genfromtxt("./cool_curve1.dat", skip_header=1)[:,0]
Temp		= np.genfromtxt("./cool_curve1.dat", skip_header=1)[:,1]
Temp = np.column_stack((Temp,np.genfromtxt("./cool_curve2.dat", skip_header=1)[:,1]))
Temp = np.column_stack((Temp,np.genfromtxt("./cool_curve3.dat", skip_header=1)[:,1]))
Temp = np.column_stack((Temp,np.genfromtxt("./cool_curve4.dat", skip_header=1)[:,1]))
Temp = np.column_stack((Temp,np.genfromtxt("./cool_curve5.dat", skip_header=1)[:,1]))
ColDens 	= np.array([1.00E+18, 3.00E+18, 1.00E+19, 1.00E+20, 1.00E+21])
Ndens		= len(Density)
# Setup neural network
Nlayer = 20
Ninput = 2
values = np.zeros((Nlayer+1,Ninput))

weights = np.load("NNweights.npy")
#weights = np.ones((Nlayer,Ninput,Ninput))*0.1
#weights = np.zeros((Nlayer,Ninput,Ninput))
#print weights

def relu(x):
	#return max(0,x)
	return np.tanh(x)

def prediction(Values, weights):

	global Nlayer
	global Ninput

	Values_new = np.copy(Values)
	for i in range(Nlayer-1):
		#print Values
		for j in range(Ninput):
			Values_new[j] = relu(np.dot(weights[i,j,:],Values[:]))
			Values = np.copy(Values_new)

	return np.dot(weights[Nlayer-1,0,:],Values[:])

def metropolis(Temp, Density, ColDens, weights):

	global Nlayer
	global Ninput

	Values 			= createInitalvalues()
	TabTemp         = np.log10(Temp[Values[0],Values[1]])
	TabValues 		= np.array([ np.log10(Density[Values[0]]), np.log10(ColDens[Values[1]]) ])
	pred_old 		= prediction(TabValues, weights);
	diff_old 		= abs(TabTemp-pred_old)
	if (pred_old == 0):
		nm = nm+1
		return False

	rnd1 = np.random.randint(Nlayer)
	rnd2 = np.random.randint(Ninput)
	rnd3 = np.random.randint(Ninput)
	rnd4 = 0.001*(-1 + np.random.rand()*2)

	weights[rnd1,rnd2,rnd3] += rnd4
	pred_new = prediction(TabValues, weights)
	if (pred_new == 0):
		weights[rnd1,rnd2,rnd3] -= rnd4
		return False

	diff_new = abs(TabTemp-pred_new)
	#print "diff_new " , diff_new , " diff_old " , diff_old , " " , diff_old/diff_new
	if (diff_new < diff_old):
		return True
	else:
		if (np.random.rand() < 1.0-100*(diff_old/diff_new)):
			print np.random.rand() ," " , diff_old/diff_new
			return True
		else:
			weights[rnd1,rnd2,rnd3] -= rnd4
	return False


def createInitalvalues():

	global Ninput
	global Ndens
	indz = np.array([np.random.randint(Ndens), np.random.randint(5)])

	return indz
for i in range(100):
	for j in range(int(1e4)):
	#print "------------------------------------------------------------------"

		nx = metropolis(Temp, Density, ColDens, weights)
		if (nx):
			np0 += 1.0
		else:
			nm0 += 1.0
	print np0/nm0
	print "------------------------------------------------------------------"
	print weights
	print "------------------------------------------------------------------"
	Values 			= createInitalvalues()
	TabTemp         = np.log10(Temp[Values[0],Values[1]])
	TabValues 		= np.array( [ np.log10(Density[Values[0]]), np.log10(ColDens[Values[1]]) ])
	pred 			= prediction(TabValues, weights);
	diff			= abs(TabTemp-pred)
	print "Diff : " , diff
	np.save("NNweights", weights)

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
