import os
import sys
import numpy as np
from astropy.io import fits
from scipy.optimize import dual_annealing as D_annel
main_dir = '/Users/chimamcgruder/Research/'
sys.path.append(main_dir+'TwinPlanets/')
sys.path.append(main_dir+'iSpec_v20201001')
import ispec 
import AnalyzeDat as AD
import ModelAtmosphere as MA
import pickle 
import time
startT = time.time()
#To use simulated Annealing to determine the best stellar parameters given the data

###############========= USER INPUT =========###############
#The dictionary keys are the name of each parameter and the element are list containing the name of the prior type and that prior function input
Params = {'logg':['uniform', 4.2, 6.2], '[fe/h]':['uniform', -1, 1]} #as of now having LogG and [Fe/H] be the same for every inhomogeniety region.
# Params['temp_eff0'], Params['temp_eff1'], Params['temp_eff2'] = ['normal',5000,200], ['normal',4500,1000], ['normal',5500,1000] #['gaussian_prior', mean, std]
Params['temp_eff0'] = ['normal',5000,200]
# Params['fract0'], Params['fract1'], Params['fract2']= ['trunc-normal', .9,.3,0,1], ['trunc-normal', .1,.3,0,1], ['trunc-normal', .1,.3,0,1] #['truncated_gaussian_prior', mean, std, lower_cap, upper_cap]
Params['fract0'] = ['trunc-normal', .9,.3,0,1]
main_dir += 'KnowThyStar/'
GridPath = main_dir+'PHOENIX_R500_from2500-17100A_wavstep5.0/' #path of where model grid is located. Be cognizant to make sure the path has the proper resolution 
#CHECK UNITS!!!!! Initially PhOENIX model is in units of [erg/s/cm^2/cm], but data and sampling is done in [erg/s/cm^2/A]
Data_file = main_dir+'CombinedSpecs_S1=T4900G4.5M0.5_SNR20.fits'
# Data_file = main_dir+'CombinedSpecs_S1=T4900G4.4M0.3_S2=T3000G4.4M0.1_S3=T7000G4.4M0.3_SNR20.fits' #name (and path) of fits file want to compare data to
Wave_file = main_dir+'PHOENIX_R500_from2500-17100A_wavstep5.0/WAVE_PHOENIX-ACES-AGSS-COND-2011-R500.fits' #name (and path) of wavelength grid fits file corresponding to data's spectrum
Model_wave_file = GridPath+'WAVE_PHOENIX-ACES-AGSS-COND-2011-R500.fits' #the wavelength grid is the same for each spectra
Nlive = 4#00 #5004 # number of live points
TrueVals = [4.5, .5, 4900] #[4.4, 0.3, 4900, .8, 3000, .17, 7000, .03] #None # for tests, when we know the actual value of the parameters
interp_kind = 'cubic' # if you have to interpolate do either 'cubic' or 'linear'. Found cubic is about twice as accuate than linear from 1 specific test
nthreads = 6 #must be int
####
#########
###############

###############========= FUNCTIONS =========############### 
def InterpRange(ModWav, Wav, TheGridPath = GridPath):  #can only use wavelengths where the data and the models overlap. This function changes the data and model to have the same wavelength range
	WARNING_message = "\nWARNING!!!   WARNING!!!   WARNING!!!   WARNING!!!"
	WARNING_message += '\nRange of data: '+str(Wav[0])+'-'+str(Wav[-1])+' but range of models: '+str(ModWav[0])+'-'+str(ModWav[-1])
	WARNING_message += "\nYou can likely get a model wavelength that encompasses the FULL data range"
	WARNING_message += "\nCurrent model grid path: '"+TheGridPath+"'"
	WARNING_message += "\nWARNING!!!   WARNING!!!   WARNING!!!   WARNING!!!\n"	
	if wave[0] < mod_wav[0] or wave[-1] > mod_wav[-1]: #TODO: assuming one wavelength range fully encompasses the other NOT NECESSARILY TRUE
		print (WARNING_message)
		dat_IdxRange = np.where((mod_wav[0]<=wave) & (wave<=mod_wav[-1])) #can only use wavelengths where the data and the models overlap
		mod_IdxRange = np.arange(len(mod_wav)) #decrease data range, but keep model range the same
	else:
		mod_IdxRange = np.where((wave[0]<=mod_wav) & (mod_wav<=wave[-1])) #can only use wavelengths where the data and the models overlap
		dat_IdxRange = np.arange(len(wave)) #decrease model range, but keep data range the same
	return mod_IdxRange, dat_IdxRange

def Order_list(LIST): #to order list so each parameter is always in the same order when passed to pymultinest
	#I want the list order with 1st logG, then [Fe/H] (because I'll always have these), then Temp1, Fract1, Temp1, Fract2, etc...
	NewList = ['logg', '[fe/h]']
	temp_cnts, frac_cnts = 0,0
	for l in LIST:
		if l == 'logg' or l == '[fe/h]':
			pass # already took care of these keys
		elif 'frac' in l:
			frac_cnts+=1
		elif 'temp' in l:
			temp_cnts +=1
		else:
			print ("Given a parameter key name that this model can't handle: '"+l+"'!!!")
			sys.exit("Acceptable parameters (CASE SENSITIVE) are: 'logg', '[fe/h]', 'temp_eff'0,1,2...N, and 'fract'0,1,2...N ")
	if temp_cnts != frac_cnts and temp_cnts != 1:
		sys.exit("Number of temperature priors given are "+str(temp_cnts)+" but there are "+str(frac_cnts)+" surface fraction priors given. These must be the same!!!")
	if temp_cnts == 1:
		NewList.append('temp_eff0')
	else:
		for i in range(temp_cnts):
			NewList.append('temp_eff'+str(i)), NewList.append('fract'+str(i))
	if 'logg' not in NewList:
		sys.exit("There was no logg prior defined! This is a needed model parameter")
	if '[fe/h]' not in NewList:
		sys.exit("There was no [fe/h] prior defined! This is a needed model parameter")
	return NewList

def LogLikeli(cube, logG=None, FeH=None): #The log likelihood function. Give this to annealing to minimize
	""" Args: cube (:class:`numpy.ndarray`): an array of parameter values.
	Returns: float: the log likelihood value. 
	"""
	# calculate the model
	pass_params = {} #to keep track of each stellar parameter 
	if logG is None: #if value was not given, then assume that we are fitting for it
		logG = cube[0]
	if FeH is None:
		FeH = cube[1]
	cubN = 2
	if StellarAtmos_cnt == 1:
		pass_params['spec1'] = [cube[cubN], logG, FeH, 1]  #if there is only one stellar surface then leave covering fraction fixed to 1
	else:
		for surf in range(StellarAtmos_cnt): #spec_params = [temp, surface_G, metallicity, surface_fraction]
			pass_params['spec'+str(surf+1)] = [cube[cubN], logG, FeH, cube[cubN+1]]  #as of rn surface_G & metallicity are the same for each heterogeneity on the stell surface
			cubN+=2
	model = MA.CombinedStellAtmo(Gridpath=GridPath, Print=False, **pass_params)
	
	if model is None: #if model parameters not within bound range, then  MA.CombinedStellAtmo() returns None 
		return np.inf #and a the loglik is -inf

	# normalisation
	norm = ndata*(-np.log(2*np.pi) - np.log(sigma*2)) #sigma was estimated using iSpec outside function

	# chi-squared
	try: # if was able to do interpolation outside likelihood
		chisq = np.sum((data-model[mod_IdxRange])**2)/(sigma**2)
	except: # len(data) > len(model)
		f = interp1d(mod_wav, model[mod_IdxRange], kind = interp_kind)
		mod_interp = f(WAVE) #making more gridpoints for model
		chisq = np.sum((data-mod_interp)**2)/(sigma**2)

	return -0.5*(norm-chisq) #think want neg logliklihood, which we then minimize. 

###############========= GLOBAL VARIABLES =========############### 
"""variables that we want to set globally so pymultinest doesn't have to calculate them for each iteration""" 
params_list = Order_list(list(Params.keys())) #CASE SENSITIVE! make sure all in lower case
ndim = len(params_list)
data = fits.getdata(Data_file)#*1e-8 #initially PHOENIX models are in [erg/s/cm^2/cm], convert to [erg/s/cm^2/A]
ndata = len(data)
wave = fits.getdata(Wave_file)
mod_wav = fits.getdata(Model_wave_file)

#To save the data in a formate that iSpec can read, in order to estimate the SNR of the data
col1 = fits.Column(name='WAVE', format='D', array=wave) 
col2 = fits.Column(name='Flux', format='D', array=data)
cols = fits.ColDefs([col1, col2])
tbhdu = fits.BinTableHDU.from_columns(cols)
prihdr = fits.Header()
prihdr['TTYPE1'], prihdr['TTYPE2'] = 'WAVE',"FLUX"
prihdr['TFORM1'], prihdr['TFORM2'] = '313111D',"313111D"
prihdu = fits.PrimaryHDU(header=prihdr)
thdulist = fits.HDUList([prihdu, tbhdu])
FileName = Data_file.split('/')[-1]
TempFilName = 'TempSpec_'+FileName
thdulist.writeto(TempFilName) #to tempirarily save the spectra in a fits file, in the right formate for iSpec to read
thdulist.close()#TODO: writing it to a fits file, could probs save a couple secs per iteration if I just save data as np.recarray
SNR = AD.estimate_SNR(TempFilName, WavRange=None)
os.remove(TempFilName)
sigma = np.mean(data)/SNR #to get a mean estimate of the SNR for the entire spectrum. This may vary widely depending on wavelength range ¯\_(ツ)_/¯
if ndim == 3: #special case, when there is only one surface
	StellarAtmos_cnt = 1
else:
	StellarAtmos_cnt = int((ndim-2)/2) # the total number of stellar inhomogeneities this model will use. 1 = completely homogeneous, 2=main temp and 1 active region, etc...

#when the length of the model wavelength grid and the data wavelength grid aren't the same, interpolate data so on same scale 
from scipy.interpolate import interp1d
if len(wave) > len(mod_wav): #tough because then have to do interpolation inside logliklihood function
	mod_IdxRange, dat_IdxRange = InterpRange(mod_wav, wave)
	WAVE = wave[dat_IdxRange]
	data = data[dat_IdxRange]
	mod_wav= mod_wav[mod_IdxRange]
else: #len(wave) < len(mod_wav), meaning make wavelength grid for data larger	
	mod_IdxRange, dat_IdxRange = InterpRange(mod_wav, wave)
	f = interp1d(wave[dat_IdxRange], data[dat_IdxRange], kind=interp_kind)
	WAVE = mod_wav[mod_IdxRange]
	data = f(WAVE) #more gridpoints for the data

logg_bnds, FeH_bnds, temp_bnds = np.array([4.2, 6.2]), np.array([-1, 1]), np.array([4800, 5200])
ret = D_annel(LogLikeli, bounds=np.array([logg_bnds,FeH_bnds,temp_bnds]))
print (ret.x)
print ("total time:", time.time()-startT)
