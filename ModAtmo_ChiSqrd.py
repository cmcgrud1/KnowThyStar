import os
import sys
import numpy as np
from dynesty import NestedSampler
from dynesty import DynamicNestedSampler
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
#To use a simple chi squared test to explore the inhomogeneous atmospheric model parameter space

###############========= USER INPUT =========###############
#The dictionary keys are the name of each parameter and the element are list containing the prior bounds. ALL PRIORS ARE UNIFORM: keys are [lower_bnd, upper_bnd, step_size]
# Params = {'logg':[3.9, 5.9, .5], '[fe/h]':[-1.1, .9, .5]} #as of now having LogG and [Fe/H] be the same for every inhomogeniety region.
# Params['temp_eff0'] = [4000,6000, 100]
global Log_G, fe_h
Log_G, fe_h = 4.4, 0.3
Params = {} #assuming surface gravity and metallicity is constrained from high res spectra
Params['temp_eff0'], Params['temp_eff1'] = [4200,5800, 50], [4000,7000, 50]
Params['fract0'] = [.9,1, .05]  #any less than .5 and it's no longer the primary atmosphere
# Params['fract0'], Params['fract1'], Params['fract2'] = [.49,1, .01], [0,.5, .01], [0,.5, .01] #Same goes for secondary atmo: any more than .49 and it's no longer the 2ndary atmo
# Params['temp_eff0'], Params['temp_eff1'], Params['temp_eff2'] = [4000,6000, 50], [3000,7000, 50], [3000,7000, 50] 
main_dir += 'KnowThyStar/'
GridPath = main_dir+'PHOENIX_R500_from2500-17100A_wavstep5.0/' #path of where model grid is located. Be cognizant to make sure the path has the proper resolution 
#CHECK UNITS!!!!! Initially PhOENIX model is in units of [erg/s/cm^2/cm], but data and sampling is done in [erg/s/cm^2/A]
Data_file = main_dir+'SynthDat/CombinedSpecs_S1=T4900G4.4M0.3_S2=T7000G4.4M0.3_SNR20.fits'
# Data_file = main_dir+'CombinedSpecs_S1=T4900G4.4M0.3_S2=T3000G4.4M0.1_S3=T7000G4.4M0.3_SNR20.fits' #name (and path) of fits file want to compare data to
Wave_file = main_dir+'PHOENIX_R500_from2500-17100A_wavstep5.0/WAVE_PHOENIX-ACES-AGSS-COND-2011-R500.fits' #name (and path) of wavelength grid fits file corresponding to data's spectrum
Model_wave_file = GridPath+'WAVE_PHOENIX-ACES-AGSS-COND-2011-R500.fits' #the wavelength grid is the same for each spectra
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

#The range of parameters available with the PHOENIX models.
def phoenix_rng(): #holding [alpha/M] fixed to 0
	T_eff1, T_eff2 = np.arange(2300, 7100, step=100),np.arange(7200, 12200, step=200)
	T_effGrid = np.append(T_eff1, T_eff2)
	log_gGrid = np.arange(0.0, 6.5, step=0.5)
	Fe_H1,Fe_H2 = np.arange(-4.0,-1.0,step=1.0), np.arange(-1.5,1.5,step=0.5)
	Fe_HGrid = np.append(Fe_H1,Fe_H2)
	return {'temp':T_effGrid, "logg":log_gGrid, "[fe/h]":Fe_HGrid}

def find_nearest(array, value): #to find the array element closest to a given value
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return array[idx]

def Order_list(LIST): #to order list so each parameter is always in the same order when passed to pymultinest
	NewList, StellParms = [], [False, False]#StellParms = used to keep track of if metallicity or logg is used
	if 'logg' in LIST: #I want the list order with 1st logG, then [Fe/H] (if they are given),
		NewList.append('logg') #then Temp1, Fract1, Temp1, Fract2, etc...
		StellParms[0] = True
	if '[fe/h]' in LIST:
		NewList.append('[fe/h]')
		StellParms[1] = True
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
	if temp_cnts != frac_cnts+1:
		sys.exit("Number of temperature priors given are "+str(temp_cnts)+" but there are "+str(frac_cnts)+" surface fraction priors given. Temp priors must be +1 frac priors!!!")
	for i in range(temp_cnts):
		NewList.append('temp_eff'+str(i)) 
		if i !=  temp_cnts-1:
			NewList.append('fract'+str(i))
	return NewList, StellParms, temp_cnts

def ChiSqrd(cube): #The ChiSqrd function = SUM((data-model)/std)^2.
	""" Args: cube (:class:`numpy.ndarray`): an array of parameter values.
	Returns: float: the log likelihood value. 
	"""
	# calculate the model
	pass_params = {} #to keep track of each stellar parameter 
	cube_cnt = 0
	global Log_G, fe_h
	if cube_start[0]: #if not given, then must define above in INPUT
		Log_G = cube[cube_cnt]
		cube_cnt +=1
	if cube_start[1]:
		fe_h = cube[cube_cnt]
		cube_cnt+=1

	if StellarAtmos_cnt == 1:
		pass_params['spec1'] = [cube[cube_cnt], Log_G, fe_h, 1]  #if there is only one stellar surface then leave covering fraction fixed to 1
	else:
		tot_cov_frac = 0 #to keep track of the total covering fraction, as it can only equal 1
		for surf in range(StellarAtmos_cnt): #spec_params = [temp, surface_G, metallicity, surface_fraction]
			if surf == StellarAtmos_cnt-1: #this is for the last stellar surface
				cov_frac = 1-tot_cov_frac #It must be the remainder of the other stellar surfaces. This allows degrees of freedom to go down by one!!!
			else:
				cov_frac = cube[cube_cnt+1] #otherwise, expecting to be given the covering fraction
				tot_cov_frac += cov_frac
			phot_temp = cube[cube_cnt]
			pass_params['spec'+str(surf+1)] = [phot_temp, Log_G, fe_h, cov_frac]  #as of rn surface_G & metallicity are the same for each heterogeneity on the stell surface
			cube_cnt+=2
	model = MA.CombinedStellAtmo(Gridpath=GridPath, Print=False, **pass_params)
	
	if model is None: #if model parameters not within bound range, then  MA.CombinedStellAtmo() returns None 
		return +np.inf #and the chi^2 is inf

	# chi-squared
	try: # if was able to do interpolation outside chi^2
		chisq = np.sum((data-model[mod_IdxRange])**2)/(sigma**2)
	except: # len(data) > len(model)
		f = interp1d(mod_wav, model[mod_IdxRange], kind=interp_kind)
		mod_interp = f(WAVE) #making more gridpoints for model
		chisq = np.sum((data-mod_interp)**2)/(sigma**2)

	return chisq #since the reduced chi^2 is just chi^2/degrees_freedom, when maximizing this it will give the same results as maximizing reg Chi^2 because degrees_freedom are the same for each model

def chunks(lst, n): #Splits list into n chunks of size approximately equal size
	if len(lst) < n:
		print ("len(lst)="+str(len(lst))+" but n="+str(n))
		print ("len(lst) < n!!!!")
		n = len(lst)
		print ("Changing n to ", len(lst))
	chnks = []
	len_chnk = int(round(len(lst)/(n*1.0))) #approximate size of each chunk. round to nearest int
	strt = 0
	for i in range(n-1): #one less than the desired number of chuncks, because the last chunck requires special consideration
		chnks.append(lst[strt:strt+len_chnk])
		strt = strt+len_chnk
	chnks.append(lst[strt:]) #for the last chunck
	return chnks

def ChiSqrdGrid(mthreads=1, limitingGrid=None): #to make a large grid of loglikihoods, and find the best gridpoint that maximizes that likelihood
	"""limitingGrid = the gridpoints that the chi^2 grid should be limited to. If provided, only makes chi^2 grid points utilizing the provided gripoints.
	This is intended to help speed up run time, so interpolations don't have to be done. MAKE SURE limitingGrid's keys match Params's keys!!!! """
	Cube = [] #must be a list of arrays because each param doesn't have the same dimensions, so can't be easily turn to an N*M array
	TotalLength = 1 #total length of Cubes (all permutations of parameters) would be product of each parameter lengths
	param_cnt = len(params_list) # number of parameters
	for p in params_list: #To make an array for each parameter range #params_list == global variable
		RANGE = np.arange(Params[p][0], Params[p][1]+Params[p][2], Params[p][2]) #Need Params[p][1]+Params[p][2], because want to include upper bound in grid. #Params == global variable
		if RANGE[-1] > Params[p][1]: #in some cases where step is not an integer and floating point round-off affects the length of out,
			RANGE = RANGE[:-1] #the last index goes past the bounds. In those cases, cut the last index
		if limitingGrid:
			KeYs = list(limitingGrid.keys())
			RANGE_refit = [] 
			if 'temp' in p: #because the key for photospheric temperatures are 'temp'+<more>, 
				p = 'temp' #need to convert it to just be 'temp' so can be found within limitingGrid's keys
			if p in KeYs: #fraction of photosphere doesn't have grid values
				for r in range(len(RANGE)):
					if p in KeYs: #fraction of photosphere doesn't have grid values
						point = find_nearest(limitingGrid[p], RANGE[r]) #to make sure each chi^2 gridpoint will conincide with the provided limitingGrid
						if point in RANGE_refit: 
							continue #if this particular gridpoint is already in there, don't add it again
						RANGE_refit.append(point)
				RANGE = np.array(RANGE_refit)
		Cube.append(RANGE) 
		TotalLength *= len(RANGE)
		# print ("'"+p+"' range:", RANGE)
	"""TODO!!!! as of right now assuming that there will only be 1 or 2 stellar photospheres. If I want it to be able to accept more, 
	need to have the stellar fraction gridpoints only include points that add to 1!!!"""
	elements = np.zeros(param_cnt) #needed for recursive function
	Cubes_track = np.zeros((TotalLength, param_cnt)) # to make an array to keep track of the parameter indeces used to make the specific ChiSqrd function 
	print ("total permutations:", TotalLength)

	global cnt 
	cnt = 0

	def Recursive(Cube, n, element):
		global cnt
		if n > 0:
			Cub = Cube[n]
			for c in range(len(Cub)):
				elements[n] = Cub[c]
				Recursive(Cube, n-1, elements)
		else:
			Cub = Cube[n]
			for c in range(len(Cub)):
				elements[n] = Cub[c]
				Cubes_track[cnt] = elements
				# CHIs[cnt] = ChiSqrd(CHIs_track[cnt]) #the ChiSqrd calcuation is the bottleneck. Do that with multthreading
				cnt += 1
	Recursive(Cube, len(Cube)-1, elements) #to calculate the slew of different parameter pertubations

	def CalcChiSqrds(tracks, retrn_dict, thread=1): #to calculate the slew of chi^2 values based on array of parameters ('Cubes_track')
		track_length = len(tracks)
		ChiSqrs = np.zeros(track_length) #To make an empyt array for the different chiSqrd options
		for t in range(track_length):
			ChiSqrs[t] = ChiSqrd(tracks[t])
		ArgMin = np.argmin(ChiSqrs) #best parameters are where the Chi^2 equation is minimized
		retrn_dict['thread'+str(thread)] = [ChiSqrs[ArgMin], tracks[ArgMin]] #to return the best chi^2 and it's corresponding parameters
		return ChiSqrs[ArgMin], tracks[ArgMin] #only need the 'return_dict' variable for multithreading. For single thread, just return the variable, like normal

	if mthreads == 0: # can't have 0 threads!!!
		mthreads = 1 
	if mthreads > 1: #to split data in chuncks for multithreading
		import multiprocessing
		manager = multiprocessing.Manager()
		return_dict = manager.dict()
		THREADs = [] #list of threads, needed to keep track of all the threads so can join them all at the end
		MultiCubes_tracks = chunks(Cubes_track, mthreads) #evenly split data so each thread can do a chunck
		thred_cnt = 1 #thead count starting from 1
		for multi_track in MultiCubes_tracks:
			print ("length of mini:", len(multi_track))
			THREADs.append(multiprocessing.Process(target=CalcChiSqrds, args=(multi_track, return_dict, thred_cnt)))
			THREADs[-1].start()
			thred_cnt += 1 
		for thread in THREADs: # now join them all so code doesn't finish until all threads are done
			thread.join()
		dict_list = list(return_dict.keys())
		Total_Chis, Total_Params = np.zeros(len(dict_list)), np.zeros((len(dict_list), param_cnt))
		for thrd in range(len(dict_list)): #now to find the global min chi^2 amongst the threaded chi^2 values
			thrd_key = dict_list[thrd]
			Total_Chis[thrd], Total_Params[thrd] = return_dict[thrd_key][0], return_dict[thrd_key][1]
		GlobMin =  np.argmin(Total_Chis)
		best_chi2, best_params = Total_Chis[GlobMin], Total_Params[GlobMin]
	else:
		best_chi2, best_params = CalcChiSqrds(Cubes_track, {})
	print ("best Chi^2:", best_chi2, 'reduced Chi^2:', best_chi2/(len(data)-param_cnt), 'and corresponding parameters: ', best_params)
	return best_chi2, best_params

###############========= GLOBAL VARIABLES =========############### 
"""variables that we want to set globally so pymultinest doesn't have to calculate them for each iteration""" 
params_list, cube_start, StellarAtmos_cnt = Order_list(list(Params.keys())) #CASE SENSITIVE! make sure all in lower case
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

###############========= RUN ChiSqrd =========############### 
results = ChiSqrdGrid(mthreads = 6, limitingGrid=phoenix_rng())
print ("Total runtime:", time.time()-startT)


