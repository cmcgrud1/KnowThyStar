import os
import sys
import numpy as np
import pymultinest
from astropy.io import fits
main_dir = '/Users/chimamcgruder/Research/'
sys.path.append(main_dir+'TwinPlanets/')
sys.path.append(main_dir+'iSpec_v20201001')
import ispec 
import AnalyzeDat as AD
import ModelAtmosphere as MA 
import time
global startT
startT = time.time() 
#To use nested sampling to explore the inhomogeneous atmospheric model parameter space

###############========= USER INPUT =========###############
#The dictionary keys are the name of each parameter and the element are list containing the name of the prior type and that prior function input
Params = {'logg':['uniform', 4.2, 6.2], '[fe/h]':['uniform', -1, 1]} #as of now having LogG and [Fe/H] be the same for every inhomogeniety region.
Params['temp_eff0'], Params['temp_eff1'], Params['temp_eff2'] = ['normal',5000,200], ['normal',5000,1000], ['normal',5000,1000] #['gaussian_prior', mean, std]
# Params['temp_eff0'] = ['normal',5000,200]
Params['fract0'], Params['fract1'], Params['fract2']= ['trunc-normal', .9,.3,0,1], ['trunc-normal', .1,.3,0,1], ['trunc-normal', .1,.3,0,1] #['truncated_gaussian_prior', mean, std, lower_cap, upper_cap]
# Params['fract0'] = ['trunc-normal', .9,.3,0,1]
main_dir += 'KnowThyStar/'
GridPath = main_dir+'PHOENIX_R500_from2500-17100A_wavstep5.0/' #path of where model grid is located. Be cognizant to make sure the path has the proper resolution 
Data_file = main_dir+'CombinedSpecs_S1=T4900G4.4M0.3_S2=T3000G4.4M0.3_S3=T7000G4.4M0.3_SNR20.fits' #name (and path) of fits file want to compare data to
Wave_file = main_dir+'PHOENIX_R500_from2500-17100A_wavstep5.0/WAVE_PHOENIX-ACES-AGSS-COND-2011-R500.fits' #name (and path) of wavelength grid fits file corresponding to data's spectrum
Model_wave_file = GridPath+'WAVE_PHOENIX-ACES-AGSS-COND-2011-R500.fits' #the wavelength grid is the same for each spectra
nlive = 1000 #5004 # number of live points
out_path = 'ModelAtmo_NoiseTst' #directory where data will go
extenstion = '/OOT_' #additionally string to attach to sampling files. OOT stands for OutOfTransit
out_base = out_path+extenstion
TrueVals = [4.4, .3, 4900, .8, 3000, .17, 7000, .03] #None# for tests, when we know the actual value of the parameters
interp_kind = 'cubic' # if you have to interpolate do either 'cubic' or 'linear'. Found cubic is about twice as accuate than linear from 1 specific test
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

def quantile(x, q, weights=None): #Compute sample quantiles with support for weighted samples. --- ripped from coner.py
	"""
	Parameters
	----------
	x : array_like[nsamples,] --- The samples.
	q : array_like[nquantiles,] --- The list of quantiles to compute. These should all be in the range ``[0, 1]``.
	weights : Optional[array_like[nsamples,]] ---  An optional weight corresponding to each sample. 
	NOTE: When ``weights`` is ``None``, this method simply calls numpy's percentile function with the values of ``q`` multiplied by 100.

	Returns
	-------
	quantiles : array_like[nquantiles,] --- The sample quantiles computed at ``q``.

	Raises
	------
	ValueError: For invalid quantiles; ``q`` not in ``[0, 1]`` or dimension mismatch between ``x`` and ``weights``.
	"""
	x = np.atleast_1d(x)
	q = np.atleast_1d(q)

	if np.any(q < 0.0) or np.any(q > 1.0):
		raise ValueError("Quantiles must be between 0 and 1")

	if weights is None:
		return np.percentile(x, list(100.0 * q))
	else:
		weights = np.atleast_1d(weights)
		if len(x) != len(weights):
			raise ValueError("Dimension mismatch: len(weights) != len(x)")
		idx = np.argsort(x)
		sw = weights[idx]
		cdf = np.cumsum(sw)[:-1]
		cdf /= cdf[-1]
		cdf = np.append(0, cdf)
		return np.interp(q, cdf, x[idx]).tolist()

#===== TRANSFORMATION OF PRIORS: 
from scipy.stats import gamma,norm,beta,truncnorm

transform_uniform = lambda x,a,b: a + (b-a)*x

transform_normal = lambda x,mu,sigma: norm.ppf(x,loc=mu,scale=sigma)

transform_beta = lambda x,a,b: beta.ppf(x,a,b)

transform_exponential = lambda x,a=1.: gamma.ppf(x, a)

def transform_loguniform(x,a,b):
	la, lb=np.log(a), np.log(b)
	return np.exp(la + x*(lb-la))

def transform_truncated_normal(x,mu,sigma,a=-np.inf,b=np.inf):
	ar, br = (a - mu) / sigma, (b - mu) / sigma
	return truncnorm.ppf(x,ar,br,loc=mu,scale=sigma)
	#========

def Prior(cube, ndim, nparams): #The prior transform going from the unit hypercube to the true parameters.
	""" Args: cube (:class:`numpy.ndarray`): an array of values drawn from the unit hypercube. Essentially the intial parameter space
		ndim, nparams: not really sure what these are needed for but pymultinest requires it
	Returns nothing, but manipulates cube 
	"""
	def ChoosePrior(cube, param): #to select whichever prior is appropriate for a given paramaeter, based on the globally defined Params dictionary
		if param[0].lower() == 'uniform':
			return transform_uniform(cube, *param[1:])
		if param[0].lower() == 'normal':
			return transform_normal(cube, *param[1:])
		if param[0].lower() == 'beta':
			return transform_beta(cube, *param[1:])
		if param[0].lower() == 'exponential':
			return transform_exponential(cube, *param[1:])
		if param[0].lower() == 'log-uniform':
			return transform_loguniform(cube, *param[1:])
		if param[0].lower() == 'trunc-normal':
			return transform_truncated_normal(cube, *param[1:])

	#prior transformation. #params_list is a global variable defined when calling 'Order_list()'
	cube[0] = ChoosePrior(cube[0], Params[params_list[0]]) #1st parameter is always the surface gravity
	cube[1] = ChoosePrior(cube[1], Params[params_list[1]]) #2nd parameter is always the metallicity 
	
	#for the 1st stellar surface. At least one stellar surface
	cube[2] = ChoosePrior(cube[2], Params[params_list[2]]) #if there is only one stellar surface then it's covering fraction is 1. As such, not scaning param space of frac

	if StellarAtmos_cnt > 1: 
		cube[3] = ChoosePrior(cube[3], Params[params_list[3]]) 		
		fracs = cube[3] #keep track of the total amount of stellar surface we've used	
		#for the 2nd to N-1 stellar surface
		i = 4 #start at 4 because 1st 4 cubes are predetermined
		while i < (2*(StellarAtmos_cnt-2))+4: #for the additional stellar surfaces. The last surface is also not included because that's just 1-frac1-frac2....-fracN
			cube[i] = ChoosePrior(cube[i], Params[params_list[i]]) #stellar temp
			Frc_Pars = Params[params_list[i+1]]
			if Frc_Pars[0] == 'uniform': #if user already wanted this prior to be uniform, keep it that way
				if Frc_Pars[2] > 1-fracs: #DOENS'T WORK IF CUBE IS AN ARRAY
					Frc_Pars[2] = 1-fracs #only constrain prior if the priors initially given are too wide
				if Frc_Pars[1] > 1-fracs: #priors do need to be yield total fraction of 1
					Frc_Pars[1] = 0
			if Frc_Pars[0] == 'log-uniform': #changing from log-uniform to uniform. Only bound from 0 to 1 anyway, no need for the log
				Frc_Pars[0], Frc_Pars[1], Frc_Pars[2]  ='uniform', 0, 1-fracs
			if Frc_Pars[0] == 'normal': #convert normal distribution to truncated normal
				Frc_Pars[0] = 'trunc-normal'
				Frc_Pars.append(0), Frc_Pars.append(1-fracs) #can't be lower than 0 or higher than 1, after the additional fraction terms were included
			if Frc_Pars[0] == 'trunc-normal': #if user already wanted this prior to be trunc-normal, keep it that way
				if Frc_Pars[4] > 1-fracs: #DOENS'T WORK IF CUBE IS AN ARRAY
					Frc_Pars[4] = 1-fracs #only constrain prior if the priors initially given are too wide
				if Frc_Pars[3] > 1-fracs: #priors do need to be yield total fraction of 1
					Frc_Pars[3] = 0
			if Frc_Pars[0] == 'beta' or Frc_Pars[0] == 'exponential':#didn't put time in figuring out how to make it truncated, just switch to truncated normall
				Frc_Pars = ['trunc-normal', .2, .3, 0, 1-fracs] #prior is established because wouldn't expect 2ndary atmospheric components to take significant fractions of that atmo
			cube[i+1] = ChoosePrior(cube[i+1], Frc_Pars) #stellar fraction
			fracs+=cube[i+1]
			i +=2
		#for the last stellar surface
		cube[i] = ChoosePrior(cube[i], Params[params_list[i]]) #stellar temp
		cube[i+1] = 1-fracs #now subtract all previous stellar fractions, so the total stellar fraction equals 1

def LogLikeli(cube, ndim, nparams): #The log likelihood function.
	""" Args: cube (:class:`numpy.ndarray`): an array of parameter values.
	Returns: float: the log likelihood value. 
	"""
	# calculate the model
	pass_params = {} #to keep track of each stellar parameter 
	cubN = 2
	if StellarAtmos_cnt == 1:
		pass_params['spec1'] = [cube[cubN], cube[0], cube[1], 1]  #if there is only one stellar surface then leave covering fraction fixed to 1
	else:
		for surf in range(StellarAtmos_cnt): #spec_params = [temp, surface_G, metallicity, surface_fraction]
			pass_params['spec'+str(surf+1)] = [cube[cubN], cube[0], cube[1], cube[cubN+1]]  #as of rn surface_G & metallicity are the same for each heterogeneity on the stell surface
			cubN+=2
	model = MA.CombinedStellAtmo(Gridpath=GridPath, Print=False,**pass_params)
	global startT
	print ("model run time:", time.time()-startT)
	startT = time.time()
	if model is None: #if model parameters not within bound range, then  MA.CombinedStellAtmo() returns None 
		return -np.inf #and a the loglik is -inf

	# normalisation
	norm = ndata*(-np.log(2*np.pi) - np.log(sigma*2)) #sigma was estimated using iSpec outside function

	# chi-squared
	try: # if was able to do interpolation outside likelihood
		chisq = np.sum((data-model[mod_IdxRange])**2)/(sigma**2)
	except: # len(data) > len(model)
		f = interp1d(mod_wav, model[mod_IdxRange], kind = interp_kind)
		mod_interp = f(WAVE) #making more gridpoints for model
		chisq = np.sum((data-mod_interp)**2)/(sigma**2)

	return 0.5*(norm - chisq)

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

if not os.path.exists(out_path):
	os.mkdir(out_path) #make new subdiretory for the sampling info if doesn't already exists

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

###############========= RUN PYMULTINEST =========############### 
pymultinest.run(LogLikeli, Prior, ndim, n_live_points=nlive, outputfiles_basename=out_base, resume=True, verbose=True) #run the algorithm
print('\n\n\n')
output = pymultinest.Analyzer(outputfiles_basename=out_base, n_params = ndim)  # Get output
posterior_samples = output.get_equal_weighted_posterior()[:,:-1] # Get out parameters: this matrix has (samples,n_params+1)
POST = {} #dict storing all postierior samples
quntL, Mean, quntH = 0.15865,0.50, 0.84135 #upper and lower bound of 1sig and mean value
POST['logG'] = {'samples': posterior_samples[:,0], 'quantiles': quantile(posterior_samples[:,0],[quntL, Mean, quntH])} #1st parameter is always the surface gravity
POST['[Fe/H]'] = {'samples': posterior_samples[:,1], 'quantiles': quantile(posterior_samples[:,1],[quntL, Mean, quntH])} #2nd parameter is always the metallicity
unstack_files = [POST['logG']['samples'], POST['[Fe/H]']['samples']] #same as POST, but without labeling each variable
Labels = ['logG', '[Fe/H]'] #to keep track of the names, since POST doesn't preserve order
LogG, Fe_H = POST['logG']['quantiles'][1], POST['[Fe/H]']['quantiles'][1]
print ('logG='+str(LogG)+'+'+str(LogG-POST['logG']['quantiles'][0])+'-'+str(POST['logG']['quantiles'][2]-LogG))
print ('[Fe/H]='+str(Fe_H)+'+'+str(Fe_H-POST['[Fe/H]']['quantiles'][0])+'-'+str(POST['[Fe/H]']['quantiles'][2]-Fe_H))
pst_cnt = 2
if StellarAtmos_cnt == 1: #There's only one stellar surface
	POST['Temp0'] = {'samples': posterior_samples[:,pst_cnt], 'quantiles': quantile(posterior_samples[:,pst_cnt],[quntL, Mean, quntH])}	
	unstack_files.append(POST['Temp0']['samples']), Labels.append('Temp0')
	Temp = POST['Temp0']['quantiles'][1]
	print ('Temp0='+str(Temp)+'+'+str(Temp-POST['Temp0']['quantiles'][0])+'-'+str(POST['Temp0']['quantiles'][2]-Temp))
else:
	for i in range(StellarAtmos_cnt):
		POST['Temp'+str(i)] = {'samples': posterior_samples[:,pst_cnt], 'quantiles': quantile(posterior_samples[:,pst_cnt],[quntL, Mean, quntH])}
		POST['Frac'+str(i)] = {'samples': posterior_samples[:,pst_cnt+1], 'quantiles': quantile(posterior_samples[:,pst_cnt+1],[quntL, Mean, quntH])}
		unstack_files.append(POST['Temp'+str(i)]['samples']), unstack_files.append(POST['Frac'+str(i)]['samples'])
		Labels.append('Temp'+str(i)), Labels.append('Frac'+str(i))
		Temp, Frac = POST['Temp'+str(i)]['quantiles'][1], POST['Frac'+str(i)]['quantiles'][1]
		print ('Temp'+str(i)+'='+str(Temp)+'+'+str(Temp-POST['Temp'+str(i)]['quantiles'][0])+'-'+str(POST['Temp'+str(i)]['quantiles'][2]-Temp))
		print ('Frac'+str(i)+'='+str(Frac)+'+'+str(Frac-POST['Frac'+str(i)]['quantiles'][0])+'-'+str(POST['Frac'+str(i)]['quantiles'][2]-Frac))
		pst_cnt+=2
postsamples = np.vstack((unstack_files)).T

import matplotlib.pyplot as plt
import corner
fig = corner.corner(postsamples, labels=Labels, truths = TrueVals, quantiles = [quntL, Mean, quntH])
fig.savefig(out_path+'/Samples_corner3.png')
plt.close()

plt.figure(2, figsize = (20,15))
# plt.plot(wave, data_noisefree, 'k-', alpha=.5) #data without noise
plt.plot(wave, data, 'g.', label='true data') #data with noise
cubN = 2

pass_params = {}
if StellarAtmos_cnt == 1: #There's only one stellar surface
	Temp = POST['Temp0']['quantiles'][1]
	pass_params['spec1'] = [Temp, LogG, Fe_H, 1]  #if there is only one stellar surface then leave covering fraction fixed to 1
else:
	for n in range(StellarAtmos_cnt): #spec_params = [temp, surface_G, metallicity, surface_fraction]
		Temp, Frac = POST['Temp'+str(n)]['quantiles'][1], POST['Frac'+str(n)]['quantiles'][1]
		pass_params['spec'+str(n+1)] = [Temp, LogG, Fe_H, Frac]  #as of rn surface_G & metallicity are the same for each heterogeneity on the stell surface
		cubN+=2
final_model = MA.CombinedStellAtmo(Gridpath=GridPath, **pass_params)
plt.plot(mod_wav, final_model, 'r--', label='best fit mod') #best fit
plt.legend()
plt.savefig(out_path+'/bestfit_stellar_spec3.png')
plt.show()
plt.close()
