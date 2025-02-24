import os
import sys
import numpy as np
from dynesty import NestedSampler
from dynesty import DynamicNestedSampler
from dynesty.utils import resample_equal
from astropy.io import fits
main_dir = '/Users/chimamcgruder/Research/'
sys.path.append(main_dir+'TwinPlanets/')
sys.path.append(main_dir+'iSpec_v20201001')
import ispec 
import AnalyzeDat as AD
import ModelAtmosphere as MA
import pickle 
import time
startT = time.time()
#To use dynesty's nested sampling to explore the inhomogeneous atmospheric model parameter space

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
Data_file = main_dir+'SynthDat/CombinedSpecs_S1=T4900G4.5M0.5_SNR20.fits'
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

def Prior(U): #The prior transform going from the unit hypercube to the true parameters.
	""" Transforms the uniform random variables `U ~ Unif[0., 1.)` to the parameters of interest
	Returns tranformed priors 
	"""
	cube = np.array(U)  # copy U

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
	return cube

def LogLikeli(cube): #The log likelihood function.
	""" Args: cube (:class:`numpy.ndarray`): an array of parameter values.
	Returns: float: the log likelihood value. 
	"""
	# calculate the model
	pass_params = {} #to keep track of each stellar parameter 
	cubN = 2
	if StellarAtmos_cnt == 1:
		# print ('cube', cube)
		pass_params['spec1'] = [cube[cubN], cube[0], cube[1], 1]  #if there is only one stellar surface then leave covering fraction fixed to 1
	else:
		for surf in range(StellarAtmos_cnt): #spec_params = [temp, surface_G, metallicity, surface_fraction]
			pass_params['spec'+str(surf+1)] = [cube[cubN], cube[0], cube[1], cube[cubN+1]]  #as of rn surface_G & metallicity are the same for each heterogeneity on the stell surface
			cubN+=2
	model = MA.CombinedStellAtmo(Gridpath=GridPath, Print=False, **pass_params)
	
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

###############========= RUN DYNESTY =========############### 
# sampler = NestedSampler(LogLikeli, Prior, ndim, nlive=Nlive) # initialize our nested sampler
# sampler.run_nested(dlogz=0.5, maxiter=10000, maxcall=50000)
# results = sampler.results
# results.summary()
# FILE = open('Dynesty_Test1.pkl', 'wb')
# pickle.dump(results, FILE)
# FILE.close()

# dsampler = DynamicNestedSampler(LogLikeli, Prior, ndim) # initialize our nested sampler
# dsampler.run_nested(dlogz_init=0.5, maxcall=500, wt_kwargs={'pfrac': 0.8}, print_progress=False) #leave maxiter=None
# # dsampler.run_nested(dlogz_init=0.5, nlive_init=5, maxiter_init=2, nlive_batch=3, maxiter_batch=4, maxcall=200, maxiter=5, wt_kwargs={'pfrac': 0.8}, print_progress=False) #leave maxiter=None
# results = dsampler.results
# FILE = open('Dynesty_Test.pkl', 'wb')
# pickle.dump(results, FILE)
# FILE.close()
# #=== Summary:
# res = '\n=======SUMMARY=======\n'
# res +="len(results['batch_nlive']):"+str(len(results['batch_nlive']))+", len(results['ncall'])"+str(len(results['ncall']))+"\n"
# res += "niter: {:d}\n ncall: {:d}\n eff(%): {:6.3f}\n".format(results['niter'], sum(results['ncall']), results['eff'])
# res += "logz: {:6.3f} +/- {:6.3f}".format(results['logz'][-1], results['logzerr'][-1])
# print (res)
# print ("results['batch_nlive']:", results['batch_nlive'])
# print ("results['ncall']:", results['ncall'])
# print ("Total run time: "+str((time.time()-startT)/60.0)+'min')

# StartT = time.time()
# from multiprocessing import Pool
# import contextlib
# nthreads = 1
# with contextlib.closing(Pool(processes=nthreads)) as executor:
# 	dsampler = DynamicNestedSampler(LogLikeli, Prior, ndim, pool=executor, queue_size=nthreads) # initialize our nested sampler
# 	dsampler.run_nested(dlogz_init=0.5, maxcall=500, wt_kwargs={'pfrac': 0.8}, print_progress=False) #leave maxiter=None
# 	results = dsampler.results
# FILE = open('Dynesty_mthreadTest.pkl', 'wb')
# pickle.dump(results, FILE)
# FILE.close()
# #=== Summary:
# weights = np.exp(results['logwt']-results['logz'][-1])
# quntL, Mean, quntH = 15.865,50,84.135
# posterior_samples = resample_equal(results.samples, weights)
# res = '=======SUMMARY=======\n'
# res +="len(results['batch_nlive']):"+str(len(results['batch_nlive']))+", len(results['ncall'])"+str(len(results['ncall']))+"\n"
# res += "niter: {:d}\n ncall: {:d}\n eff(%): {:6.3f}\n".format(results['niter'], sum(results['ncall']), results['eff'])
# cnt = 0
# for s in range(posterior_samples.shape[1]):
# 	Quants = np.percentile(posterior_samples[:,s],[quntL, Mean, quntH])	
# 	res+= params_list[cnt]+' = '+str(Quants[1])+'+'+str(Quants[1]-Quants[0])+'-'+str(Quants[2]-Quants[1])+'\n'
# 	cnt+=1
# print (res)
# print ("results['batch_nlive']:", results['batch_nlive'])
# print ("results['ncall']:", results['ncall'])
# print ("Total run time:", (time.time()-StartT)/60 )

StartT = time.time()
ntheads = 2
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(ntheads)
dsampler = DynamicNestedSampler(LogLikeli, Prior, ndim, pool=executor, queue_size=nthreads) # initialize our nested sampler
dsampler.run_nested(dlogz_init=0.5, maxcall=500, wt_kwargs={'pfrac': 0.8}, print_progress=False) #leave maxiter=None
results = dsampler.results
FILE = open('Dynesty_mthreadTest.pkl', 'wb')
pickle.dump(results, FILE)
FILE.close()
#=== Summary:
weights = np.exp(results['logwt']-results['logz'][-1])
quntL, Mean, quntH = 15.865,50,84.135
posterior_samples = resample_equal(results.samples, weights)
res = '=======SUMMARY=======\n'
res +="len(results['batch_nlive']):"+str(len(results['batch_nlive']))+", len(results['ncall'])"+str(len(results['ncall']))+"\n"
res += "niter: {:d}\n ncall: {:d}\n eff(%): {:6.3f}\n".format(results['niter'], sum(results['ncall']), results['eff'])
cnt = 0
for s in range(posterior_samples.shape[1]):
	Quants = np.percentile(posterior_samples[:,s],[quntL, Mean, quntH])	
	res+= params_list[cnt]+' = '+str(Quants[1])+'+'+str(Quants[1]-Quants[0])+'-'+str(Quants[2]-Quants[1])+'\n'
	cnt+=1
print (res)
print ("results['batch_nlive']:", results['batch_nlive'])
print ("results['ncall']:", results['ncall'])
print ("Total run time:", (time.time()-StartT)/60 )
# ##############========= PLOTTING =========###############
# import matplotlib.pyplot as plt
# from dynesty import plotting as dyplot

# quntL, Mean, quntH = 0.15865,0.50, 0.84135 #upper and lower bound of 1sig and mean value

# # initialize figure
# fg, ax = dyplot.cornerplot(results, color='blue', show_titles=True, max_n_ticks=5, quantiles=[quntL, Mean, quntH], truths=TrueVals)
# fg.savefig('Dynesty_corner.png')
# plt.show()
# plt.close()