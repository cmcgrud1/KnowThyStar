import os
import sys
import glob
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
PHOENIX_path = '/Users/chimamcgruder/Research/KnowThyStar/PHOENIX/'
PHOENIXR500_5step_path = '/Users/chimamcgruder/Research/KnowThyStar/PHOENIX_R500_from2500-17100A_wavstep5.0/'
#download PHEONIX data from: ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/
#linear interpolation guide from section 2.3: https://www.sternwarte.uni-erlangen.de/docs/theses/2016-04_Kreuzer.pdf

def find_nearest(array, value): #to find the array element closest to a given value
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx #array[idx]

def UpperLowLims(array, value): #To find the upper and lower limits for each axis of my grid
	Idx = find_nearest(array, value)
	nearest =array[Idx]
	if nearest == value: #if the actual value is in our grid, then use it for both the upper and lower bounds
		Upper, Lower = nearest, nearest #this effectively reduces the dimensonality of the interpolation
	if nearest < value: #Then closest point smaller than value is nearest and closes point larger is 1 step above nearest
		Lower = nearest
		Upper = array[Idx+1]
	if nearest > value: #Then closest point greater than value is nearest and closes point smaller is 1 step below nearest
		Lower = array[Idx-1]
		Upper = nearest
	return [float(Lower), float(Upper)]

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

#The range of parameters available with the PHOENIX models.
def phoenix_rng(): #holding [alpha/M] fixed to 0
	T_eff1, T_eff2 = np.arange(2300, 7100, step=100),np.arange(7200, 12200, step=200)
	T_effGrid = np.append(T_eff1, T_eff2)
	log_gGrid = np.arange(0.0, 6.5, step=0.5)
	Fe_H1,Fe_H2 = np.arange(-4.0,-1.0,step=1.0), np.arange(-1.5,1.5,step=0.5)
	Fe_HGrid = np.append(Fe_H1,Fe_H2)
	return {'Temp':T_effGrid, "LogG":log_gGrid, "[Fe/H]":Fe_HGrid}

def PHOENIXfileCriteria(param): #Provides the specific string struturing used for PHONEIX files. Pulse out the proper formate for each variable to be consistent with the PHONEIX file
	#param is a 2 component list where the 1st element is the parameter name, and its corresponding float value is the 2nd element  
	temp_options = ['temp', 'temperature', 't_eff'] 
	if param[0].lower() in temp_options:
		T_str =str(int(param[1]))
		if len(T_str) == 4: #the string defining the temp is 5 elements. The 1st element == 0 if temp is under 10k
			T_str = '0'+T_str
		return T_str
	
	logG_options = ['logg', 'surface gravity'] #a few acceptable names for this parameter
	if param[0].lower() in logG_options:
		G_str ="{:.2f}".format(param[1]) #has to have 2 sigs after decimal
		return G_str
	
	metal_options = ['metallicity', '[fe/h]', '[m/h]']
	if param[0].lower() in metal_options: 
		M_str = "{:.1f}".format(param[1])
		if param[1] > 0: #if positive metallicity. Add '+' in front of it
			M_str = '+'+M_str
		elif param[1] == 0: #if metallicity is 0. Add '-' in front of it, because that's how PHOENIX model is orientied
			M_str = '-0.0'
		else: #if metallicity is negative, do nothing, because negative is already written in float number
			pass
		return M_str

def WidenBounds(key, Dict, paramGrid): #To widen the bounds so the edge of each parameters bound is a grid point
	lower, upper = Dict[key][0], Dict[key][1]
	Rng = [UpperLowLims(paramGrid, lower)[0]] #extend the lower bound to the nearest, lower grid point
	#if the lower bound is one of the grid points, then this parameter won't be extended
	Rng.append(UpperLowLims(paramGrid, upper)[1])  #Same, but in opposite direction for the upper bound
	return Rng	#retuns 2 element list

#1D linear introplation following the steps of section 2.3 in Simon Kreuzer's 2016 Masters Thesis: https://www.sternwarte.uni-erlangen.de/docs/theses/2016-04_Kreuzer.pdf
#alot is Hard coded specifically for PHOENIX models!!!
def Interpolate(InterValues, GridPath=PHOENIX_path, Criteria=PHOENIXfileCriteria, GridRange=phoenix_rng(), Print=True): #TODO: have work for an arbitary number of dimensions
	#InterValues = dictonary of floats, where each dict key is the specific stellar parameter for the given model and it's corresponding value to be interpolated. For PHOENIX models: [temp, surface gravity, metallicity] #holding [alpha/M] fixed to 0
	#GridRange = dictionary of arrays where each element is a range of parameters available with that models and the keys are the specific star params for the given model.
	#NOTE: Make sure all keys in dictionaries are consitent throughout all passed parameters and functions
	if GridRange['Temp'][0] > InterValues['Temp'] or InterValues['Temp'] > GridRange['Temp'][-1]:
		if Print:
			print ("Temperature for which you want to interpolate is out of model grid bounds! Temp bounds:"+str(GridRange['Temp'][0])+'-'+str(GridRange['Temp'][-1]))
		return None
	if GridRange['[Fe/H]'][0] > InterValues['[Fe/H]'] or InterValues['[Fe/H]'] > GridRange['[Fe/H]'][-1]:
		if Print:
			print ("Metallicity for which you want to interpolate is out of model grid bounds! [Fe/H] bounds:"+str(GridRange['[Fe/H]'][0])+'-'+str(GridRange['[Fe/H]'][-1]))
		return None
	if GridRange['LogG'][0] > InterValues['LogG'] or InterValues['LogG'] > GridRange['LogG'][-1]:
		if Print:
			print ("Log gravity for which you want to interpolate is out of model grid bounds! Temp bounds:"+str(GridRange['LogG'][0])+'-'+str(GridRange['LogG'][-1]))
		return None
	try: #There are edge cases where interpolation won’t work because @ end of grid bounds or special grid limits (ex. at T_eff=6900,logg=0.5 doesn't exisits)
		XYZ = {} #Directory of arrays where each element is the upper an lower bounds for a given x, y, z, ... n dimension, where each dimension is a parameter to define the model
		for interp_key in list(InterValues.keys()):
			XYZ[interp_key] = UpperLowLims(GridRange[interp_key], InterValues[interp_key])

		XYZ_variable = {} #directory of arrays, containing np.array([lower bound, upper bound, desired value to be interpolate]) for only elements of XYZ that vary. The dimensionality of the interpolation is only dictated by the number of VARYING parameters
		V = 1.0
		XYZ_str = {} #the proper PHOENIX string formating for specific parameter
		for xyz in list(XYZ.keys()):
			if XYZ[xyz][1]-XYZ[xyz][0] != 0: #if != 0 then have an upper and lower bound to be interpolated
				V *= XYZ[xyz][1]-XYZ[xyz][0] #1st element is smaller than 2nd according to UpperLowLims function
				XYZ_variable[xyz] = np.array([XYZ[xyz][0], XYZ[xyz][1], float(InterValues[xyz])])
			else:
				XYZ_str[xyz] = Criteria([xyz, XYZ[xyz][0]]) #XYZ[xyz][0]] == XYZ[xyz][1]]

		if len(list(XYZ_variable.keys())) == 0: #Then the model desired to be interpolated is a gridpoint in the selected model. Just return the extract data 
			interp_data = glob.glob(GridPath+'lte'+XYZ_str['Temp']+'-'+XYZ_str["LogG"]+XYZ_str["[Fe/H]"]+'*')
			if len(interp_data) != 1:
				sys.exit("Found "+str(len(interp_data))+" files with base name '"+GridPath+'lte'+XYZ_str['Temp']+'-'+XYZ_str["LogG"]+XYZ_str["[Fe/H]"]+".' It should only be one!")
			return fits.getdata(interp_data[0]) #TODO: need to make more general
		Interpted  = []

		global i #gonna edit this in Recursive function 
		i = 0
		XYZ_variable_i = {} #to keep track of which parameter bounds we are working with in Recursive function. dictionary of floats
		#to get each gridpoint surrounding the spectra to be interpolated
		#for 3 dimenstional param space this yields 8 components because looking for the outline points in a 3D cube. 
		#In general, for any orbitary dimensions (D) would be 2^D points
		def Recursive(Lst_of_Bnds, n, keys):
			global i
			if n > 1:
				key = keys[n-1]
				for d in range(len(Lst_of_Bnds[key])-1): #-1, because last index is the value in which we want to interpolate 
					XYZ_str[key] = Criteria([key, Lst_of_Bnds[key][d]]) #so only iterating through 1st and 2nd index, which are the bounds
					XYZ_variable_i[key] = Lst_of_Bnds[key][d]
					Recursive(Lst_of_Bnds, n-1, keys)
			else:
				key = keys[n-1]
				for d in range(len(Lst_of_Bnds[key])-1): 
					XYZ_str[key] =  Criteria([key, Lst_of_Bnds[key][d]])			
					XYZ_variable_i[key] = Lst_of_Bnds[key][d]
					N_i, i = 1, i+1
					interp_data = glob.glob(GridPath+'lte'+XYZ_str['Temp']+'-'+XYZ_str["LogG"]+XYZ_str["[Fe/H]"]+'*')
					if len(interp_data) != 1:
						sys.exit("Found "+str(len(interp_data))+" files with base name '"+GridPath+'lte'+XYZ_str['Temp']+'-'+XYZ_str["LogG"]+XYZ_str["[Fe/H]"]+".' It should only be one!")
					spec_i = fits.getdata(interp_data[0]) #TODO: need to make more general
					for k in keys:
						i_021 = np.delete(Lst_of_Bnds[k], find_nearest(Lst_of_Bnds[k], XYZ_variable_i[k])) 
						N_i *= abs(np.diff(i_021)) #equ 2.6-2.8 of Kreuzer 2016
					N_i = N_i/V
					Interpted.append(N_i*spec_i)
		XYZ_keys = list(XYZ_variable.keys())
		Recursive(XYZ_variable, len(XYZ_keys), XYZ_keys)

		Interpted = np.sum(np.array(Interpted), axis =0)
		return Interpted #returns introplated 1D-spectra, with the same wavelength grid.
	
	except: #In those special grid limit edge cases, just return None
		if Print:
			print ("Can't interpolate the spectrum with parameters: Temp="+str(InterValues['Temp'])+", LogG="+str(InterValues['LogG'])+", Metallicity="+str(InterValues['[Fe/H]']))
		return None

# # TO MAKE A GRID OF MODELS AT THE SPECIFIC RESOLUTION & wavelength range, BEFORE RUNNING PARAMETER SPACE SCAN
# #HiRes resolution of PHOENIX models is 500000
# def DegradeGridRes(NewRes, InitialRes=500000, grid_path=PHOENIX_path, New_res_path=None, WavRange=None, threads=None, PRIORS=None, wavespacing=None, IdxChunks=None): #multithread this!!!!!
# 	#PRIORS = dictionary of upper and lower bound of parameter space in sampling. each dictonary element is 1) 'metallicity', '[fe/h]', '[m/h]'; 2) 't', 'temp', 't_eff', or 'temperature', 3) 'logg' or 'surface gravity' 
# 	#import iSpec, which is needed to degrade resolution
# 	import logging
# 	ispec_dir = '/Users/chimamcgruder/Research/'
# 	sys.path.append(ispec_dir+'TwinPlanets/')
# 	sys.path.append(ispec_dir+'iSpec_v20201001')
# 	import ispec 
# 	import AnalyzeDat as AD  
# 	gridbasenames = 'PHOENIX-ACES-AGSS-COND-2011' #the name used in each grid file. TODO: make more general for other stellar models
	
# 	# global grid_cnt, thred_count #global variable that will be edited in each thread# Counting each iteration is much harder with multiprocess
# 	# grid_cnt, thred_count = 0, 0 #because run on each CPU individually. I'd have to write a way to for each core to communicate the counts. Too much effort as of now
# 	def ReMakeGrid(Models): # for loop to recreate PHOENIX models grid, but with changed resolution and wavelength range
# 		# global grid_cnt, thred_count
# 		for M in Models:
# 			M_name = M.split('/')[-1]
# 			if os.path.isfile(New_res_path+'/'+M_name): #don't recreate file if already exits
# 				# grid_cnt+=1
# 				continue
# 			# thred_count+=1
# 			spec = fits.getdata(M) # just the flux
# 			#To save data in format that iSpec can read: as a fitsrec.FITS_rec
# 			col1 = fits.Column(name='WAVE', format='D', array=wav) #original wavelength grid
# 			col2 = fits.Column(name='Flux', format='D', array=spec[Idxbounds])
# 			cols = fits.ColDefs([col1, col2])
# 			tbhdu = fits.BinTableHDU.from_columns(cols)
# 			prihdr = fits.Header()
# 			prihdr['TTYPE1'], prihdr['TTYPE2'] = 'WAVE',"FLUX"
# 			prihdr['TFORM1'], prihdr['TFORM2'] = '313111D',"313111D"
# 			prihdu = fits.PrimaryHDU(header=prihdr)
# 			thdulist = fits.HDUList([prihdu, tbhdu])
# 			TempFilName = '/TempSpec_'+str(M_name[:M_name.find('.PHOENIX')])+'.fits'
# 			thdulist.writeto(New_res_path+TempFilName, overwrite=True) #to tempirarily save the spectra in a fits file, in the right formate for iSpec to read
# 			thdulist.close()#TODO: writing it to a fits file, could probs save a couple secs per iteration if I just save data as np.recarray
# 			ResDeSpec, OGspec = AD.degrade_resolution(New_res_path+TempFilName, from_res=InitialRes, to_res=NewRes)
# 			#To save data as fits file
# 			if wavespacing:
# 				ResDeg_flux = np.zeros(len(Idx_chunks))
# 				for I in range(len(Idx_chunks)):
# 					ResDeg_flux[I] = np.median(ResDeSpec['flux'][Idx_chunks[I]])
# 			else:
# 				ResDeg_flux = ResDeSpec['flux']
# 			NewDat = fits.PrimaryHDU(ResDeg_flux)
# 			NewDat.writeto(New_res_path+'/'+str(M_name[:M_name.find('PHOENIX')])+gridbasenames+'-R'+str(NewRes)+'.fits')
# 			os.remove(New_res_path+TempFilName)
# 			# grid_cnt+=1
# 			# print ("Created "+str(grid_cnt)+"/"+str(LenM)+" files - Thread"+str(thred_count)+"\n")
# 			print ("Created "+str(M_name)+"file at reduced resolution")
# 		return None

# 	if IdxChunks:
# 		Idx_chunks = np.load(IdxChunks, allow_pickle=True) #have to do this because each bin element isn't the same length
# 	WAVs = glob.glob(grid_path+'WAVE*.fits')
# 	if len(WAVs) > 1:
# 		sys.exit("More than one wavelength grid in folder '"+grid_path+"'!!!")
# 	wav = fits.getdata(WAVs[0]) #PHOENIX wave in angstroms, which is exactly what iSpec expects
# 	Idxbounds = np.arange(len(wav))
# 	if WavRange: #truncate data to only include spectrum in desired wavelength range
# 		Idxbounds = np.where((WavRange[0]<=wav) & (wav<=WavRange[1]))[0]
# 		wav = wav[Idxbounds]
# 	Range = np.array([np.min(wav), np.max(wav)])  #just for printing and naming purposes  
# 	""" 
# 	Have the new resolution, but still in the same wavelength grid as before. 
# 	Might want to reduce the wavelength spacing to correspond to the new resolution.
# 	Only need to do this when simulating synthetic data. Not when reducing resolution grid for pymultinest
# 	""" 
# 	if wavespacing:
# 		print ("Determining line spacing for data at Resolution "+str(NewRes))
# 		Wav_diff = np.diff(wav) #wavelength spacing is not consistent throughout entire spectrum
# 		if type(wavespacing) == float or type(wavespacing) == int: #if a number is given, 
# 			pass #assuming that is the prefered wavelength spacing the data should have. Use that. Assume given in Ang
# 		else: #otherwise, determine spacing with resolution and smallest wavelength. 
# 			wavespacing = wav[0]/(3*NewRes) #Using the smallest wavelength, yields finest grid (R = lambda/Del_lambda))
# 		# if wavespacing < np.mean(Wav_diff): #if the desired wavespacing is smaller than the initial mean data spacing, 
# 		# 	wavespacing = np.round(np.mean(Wav_diff), 4) #then change spacing to min value it should be => np.mean(Wav_diff)
# 		if not IdxChunks: #if IdxChuncks is provided then don't go through this whole process agin
# 			Wav_chunks, Idx_chunks, STOP = [], [], False #this section takes ~20sec for bins of 10Ang, ~200sec with bins of 1Ang, 2000sec with bins of 0.1Ang etc.
# 			strt, end = wav[0], wav[find_nearest(wav, wav[0]+wavespacing)]
# 			if strt == end: #then the desired wavespacing is smaller than the model wave spacing
# 				end = wav[1] #in that case, just keep the same spacing of the model
# 			while not STOP:
# 				bbin = np.where((strt<=wav) & (wav<=end))[0]
# 				# if len(bbin) == 2: #check for edge case where wavespacing is smaller than the initial binnning
# 				# 	if np.diff(wav[bbin]) > wavespacing:
# 				# 		bbin = np.array([bbin[0]]) #in that case just use original wavelength grids
# 				Wav_chunks.append(wav[bbin]), Idx_chunks.append(bbin)
# 				if bbin[-1] == len(wav)-1:#at the end of the wavelength grid
# 					STOP = True
# 				elif bbin[-1] == len(wav)-2:#then one away from the edge. In that case have the last bin be by itself
# 					Wav_chunks.append(wav[len(wav)-1]), Idx_chunks.append(len(wav)-1) #len(wav)-1 = last index
# 					STOP = True
# 				else:
# 					lastWave = bbin[-1]+1 #start with one index above the last bin
# 					strt = wav[lastWave]
# 					end = wav[find_nearest(wav, wav[lastWave]+wavespacing)]
# 					if strt == end: #then the desired wavespacing is smaller than the model wave spacing
# 						end =  wav[lastWave+1] #in that case, just keep the same spacing of the model		
# 			LowResWave = np.zeros(len(Wav_chunks))
# 			for w in range(len(Wav_chunks)):
# 				LowResWave[w] = np.median(Wav_chunks[w])
# 			print ("Wavelength spacing of\033[1m", wavespacing, "\033[0myields a wavelength grid of\033[1m", len(LowResWave), "\033[0mdatapoints")
# 	else:
# 		LowResWave = wav 
# 	"""==========================="""
# 	if not New_res_path: #if didn't specify folder for new data, then name is based off the grid folder where collecting original data from
# 		if grid_path[-1] == '/':
# 			grid_path = grid_path[:-1] #-1 to get ride of '/' in path naame
# 		New_res_path = grid_path+'_R'+str(NewRes)+'_from'+str(int(round(Range[0])))+'-'+str(int(round(Range[1])))+'A'
# 		if wavespacing:
# 			New_res_path+='_wavstep'+str(np.round(wavespacing, 2))
# 	if not os.path.exists(New_res_path):
# 		os.mkdir(New_res_path) #Make new path to store PHOENIX models @ different resolution
# 	if not os.path.isfile(New_res_path+'/WAVE_'+gridbasenames+'-R'+str(NewRes)+'.fits'): #to save the new wavelength grid 
# 		if wavespacing and IdxChunks: #if Idx_chunks was provided in function call, then didn't define LowResWave above, must do it here
# 			LowResWave = np.zeros(len(Idx_chunks))
# 			for I in range(len(Idx_chunks)):
# 				LowResWave[I] = np.median(wav[Idx_chunks[I]])
# 		fits.PrimaryHDU(LowResWave).writeto(New_res_path+'/WAVE_'+gridbasenames+'-R'+str(NewRes)+'.fits')  #only have to do this once. Degrading the resolution doesn't change wavelength grid
# 	grid_models = glob.glob(grid_path+'/lte*.fits')
# 	if PRIORS: #only save data that's within specified bounds
# 		grid_models_cut = []
# 		#The range of parameters available with the PHOENIX models.
# 		Grid = phoenix_rng() #holding [alpha/M] fixed to 0

# 		for k in list(PRIORS.keys()): #first loop through the prior directory to get the range of parameterspace allowed to be explored
# 			if k.lower()=='metallicity' or k.lower()=='[fe/h]' or k.lower()=='[m/h]': #this is for the metallicity
# 				MetalRng = WidenBounds(k, PRIORS, Grid[k])
# 			if k.lower()=='temp' or k.lower()=='temperature' or k.lower()=='t_eff': #this is for the photospheric effective temperature 
# 				TempRng = WidenBounds(k, PRIORS, Grid[k]) 
# 			if k.lower()=='logg' or k.lower()=='surface gravity': #this is for the surface gravity
# 				logGRng = WidenBounds(k, PRIORS, Grid[k])
# 		print ("MetalRng:", MetalRng, "TempRng:", TempRng, "logGRng:", logGRng)
# 		#TODO: when making genaric to any model grid. Have to use the dictonary keys as guide towards the struture of my file name strings?		

# 		for mod in grid_models:
# 			file = mod.split('/')[-1] #don't want the whole path, just the file name
# 			Keep = True #if any of these below criteria are not meet, then won't add particular file to 'grid_models_cut'
# 			for k in list(PRIORS.keys()):
# 				temp = float(file[3:8]) #the string name of each PHOENIX model file is the same length, with each parameter at the same location
# 				if TempRng[0] <= temp and temp <= TempRng[1]:
# 					pass 
# 				else:
# 					Keep = False
# 				logg = float(file[9:13]) 
# 				if logGRng[0] <= logg and logg <= logGRng[1]:
# 					pass 
# 				else:
# 					Keep = False
# 				metallicity = float(file[13:17]) 
# 				if MetalRng[0] <= metallicity and metallicity <= MetalRng[1]:
# 					pass 
# 				else:
# 					Keep = False
# 			if Keep:
# 				grid_models_cut.append(mod)
# 		grid_models = grid_models_cut #reassign 'grid_models' to only include gridpoints within my bounds
# 	LenM = len(grid_models)
# 	print ("Saving \033[1m"+str(LenM)+"\033[0m files in directory \033[1m'"+New_res_path+"'\033[0m, with Resolution = \033[1m"+str(NewRes)+"\033[0m and wavelength range \033[1m"+str(np.round(Range,2))+"\033[0mAng")
# 	if threads: #to split the list of files by # of threads, that way can run each list at the same time
# 		MultiMods = chunks(grid_models, threads) #to break the list of models into sublist, which will be calculated in parallel
# 		import multiprocessing
# 		THREADs = [] #list of threads, needed to keep track of all the threads so can join them all at the end
# 		for multi in MultiMods:
# 			THREADs.append(multiprocessing.Process(target=ReMakeGrid, args=(multi,)))
# 			THREADs[-1].start()
# 		for thread in THREADs: # now join them all so code doesn't finish until all threads are done
# 			thread.join()
# 	else:
# 		ReMakeGrid(grid_models)
# 		# for g in grid_models:
# 		# 	print (g)
# 	return None 
# TO MAKE A GRID OF MODELS AT THE SPECIFIC RESOLUTION & wavelength range, BEFORE RUNNING PARAMETER SPACE SCAN
#HiRes resolution of PHOENIX models is 500000
def DegradeGridRes(NewRes, InitialRes=500000, grid_path=PHOENIX_path, New_res_path=None, WavRange=None, threads=None, PRIORS=None, wavespacing=None, IdxChunkInfo=None):
	#PRIORS = dictionary of upper and lower bound of parameter space in sampling. each dictonary element is 1) 'metallicity', '[fe/h]', '[m/h]'; 2) 't', 'temp', 't_eff', or 'temperature', 3) 'logg' or 'surface gravity' 
	#import iSpec, which is needed to degrade resolution
	#IdxChunkInfo = list where the first element is a string pointing to the .npy file containing the binning chuncks approporate for the given resolution, the 2nd element is true or false, letting the code now it the wavelength range was precut when making the .npy file
	if not IdxChunkInfo:
		IdxChunks = None
	else:
		IdxChunks = IdxChunkInfo[0]
		PreCut = IdxChunkInfo[1]
	import logging
	ispec_dir = '/Users/chimamcgruder/Research/'
	sys.path.append(ispec_dir+'TwinPlanets/')
	sys.path.append(ispec_dir+'iSpec_v20201001')
	import ispec 
	import AnalyzeDat as AD  
	gridbasenames = 'PHOENIX-ACES-AGSS-COND-2011' #the name used in each grid file. TODO: make more general for other stellar models
	# global grid_cnt, thred_count #global variable that will be edited in each thread# Counting each iteration is much harder with multiprocess
	# grid_cnt, thred_count = 0, 0 #because run on each CPU individually. I'd have to write a way to for each core to communicate the counts. Too much effort as of now
	def ReMakeGrid(Models): # for loop to recreate PHOENIX models grid, but with changed resolution and wavelength range
		# global grid_cnt, thred_count
		for M in Models:
			M_name = M.split('/')[-1]
			mod_params = M_name[:M_name.find('PHOENIX')]
			New_file_name = mod_params+gridbasenames+'-R'+str(NewRes)+'.fits'
			if os.path.isfile(New_res_path+'/'+New_file_name): #don't recreate file if already exits
				print (M_name+" file at reduced resolution previously created")
				# grid_cnt+=1
				continue
			# thred_count+=1
			spec = fits.getdata(M) # just the flux
			#To save data in format that iSpec can read: as a fitsrec.FITS_rec
			col1 = fits.Column(name='WAVE', format='D', array=wav) #original wavelength grid
			col2 = fits.Column(name='Flux', format='D', array=spec[Idxbounds])
			cols = fits.ColDefs([col1, col2])
			tbhdu = fits.BinTableHDU.from_columns(cols)
			prihdr = fits.Header()
			prihdr['TTYPE1'], prihdr['TTYPE2'] = 'WAVE',"FLUX"
			prihdr['TFORM1'], prihdr['TFORM2'] = '313111D',"313111D"
			prihdu = fits.PrimaryHDU(header=prihdr)
			thdulist = fits.HDUList([prihdu, tbhdu])
			TempFilName = '/TempSpec_'+mod_params+'fits'
			thdulist.writeto(New_res_path+TempFilName, overwrite=True) #to tempirarily save the spectra in a fits file, in the right formate for iSpec to read
			thdulist.close()#TODO: writing it to a fits file, could probs save a couple secs per iteration if I just save data as np.recarray
			ResDeSpec, OGspec = AD.degrade_resolution(New_res_path+TempFilName, from_res=InitialRes, to_res=NewRes)
			#To save data as fits file
			if wavespacing:
				ResDeg_flux = np.zeros(len(Idx_chunks))
				for I in range(len(Idx_chunks)):
					ResDeg_flux[I] = np.median(ResDeSpec['flux'][Idx_chunks[I]])
			else:
				ResDeg_flux = ResDeSpec['flux']
			NewDat = fits.PrimaryHDU(ResDeg_flux)
			NewDat.writeto(New_res_path+'/'+New_file_name)
			os.remove(New_res_path+TempFilName)
			# grid_cnt+=1
			# print ("Created "+str(grid_cnt)+"/"+str(LenM)+" files - Thread"+str(thred_count)+"\n")
			print ("Created "+str(M_name)+" file at reduced resolution")
		return None

	WAVs = glob.glob(grid_path+'WAVE*.fits')
	if len(WAVs) > 1:
		sys.exit("More than one wavelength grid in folder '"+grid_path+"'!!!")
	wav = fits.getdata(WAVs[0]) #PHOENIX wave in angstroms, which is exactly the units iSpec expects
	if WavRange: #truncate data to only include spectrum in desired wavelength range.
		Idxbounds = np.where((WavRange[0]<=wav) & (wav<=WavRange[1]))[0]
		wav = wav[Idxbounds] 
	Range = np.array([np.min(wav), np.max(wav)])  #just for printing and naming purposes  

	if IdxChunks and wavespacing:
		Idx_chunks = np.load(IdxChunks, allow_pickle=True) #have to set 'allow_pickle=True' because each bin element isn't the same length
		if not PreCut:
			#now to cut these index chuncks to only inlude wavelengths of interest
			mn_idx, mx_idx = Idxbounds[0], Idxbounds[-1]
			for ic in range(len(Idx_chunks)):
				if mn_idx in Idx_chunks[ic]: #to identify which chunck contains the lower limit of the wavelength range
					frst_chnk = ic #to keep track of which chuncks I even need 
					start = np.where(Idx_chunks[ic] == mn_idx)[0][0] #to keep track of which index of the 1st useable chunck, is usable
				if mx_idx in Idx_chunks[ic]: #to identify which chunck contains the lower limit of the wavelength range
					lst_chnk = ic 
					end = np.where(Idx_chunks[ic] == mx_idx)[0][0] #to keep track of which index of the last useable chunck, is usable
			Idx_chunks = Idx_chunks[frst_chnk:lst_chnk+1] #to cut out all index chuncks that don't encompass useable wavelength ranges
			Idx_chunks[0] = Idx_chunks[0][start:]
			Idx_chunks[-1] = Idx_chunks[-1][:end+1] #to cut out all indeces in the 1st and last chunck that don't encompass the wavelength ranges
			#Now to reset this for the 1st index of the 1st chunk == 0. Need this because we cut wav to only be wavelengths of interests
			for chnk in range(len(Idx_chunks)):
				Idx_chunks[chnk] = Idx_chunks[chnk]-mn_idx
	""" 
	Have the new resolution, but still in the same wavelength grid as before. 
	Might want to reduce the wavelength spacing to correspond to the new resolution.
	Only need to do this when simulating synthetic data. Not when reducing resolution grid for pymultinest
	""" 
	if wavespacing:
		print ("Determining line spacing for data at Resolution "+str(NewRes))
		Wav_diff = np.diff(wav) #wavelength spacing is not consistent throughout entire spectrum
		if type(wavespacing) == float or type(wavespacing) == int: #if a number is given, 
			pass #assuming that is the prefered wavelength spacing the data should have. Use that. Assume given in Ang
		else: #otherwise, determine spacing with resolution and smallest wavelength. 
			wavespacing = wav[0]/(3*NewRes) #Using the smallest wavelength, yields finest grid (R = lambda/Del_lambda))
		# if wavespacing < np.mean(Wav_diff): #if the desired wavespacing is smaller than the initial mean data spacing, 
		# 	wavespacing = np.round(np.mean(Wav_diff), 4) #then change spacing to min value it should be => np.mean(Wav_diff)
		if not IdxChunks: #if IdxChuncks is provided then don't go through this whole process again
			Wav_chunks, Idx_chunks, STOP = [], [], False #this section takes ~20sec for bins of 10Ang, ~200sec with bins of 1Ang, 2000sec with bins of 0.1Ang etc.
			strt, end = wav[0], wav[find_nearest(wav, wav[0]+wavespacing)]
			# if strt == end: #then the desired wavespacing is smaller than the model wave spacing
			# 	end = wav[1] #in that case, just keep the same spacing of the model
			while not STOP:
				bbin = np.where((strt<=wav) & (wav<=end))[0]
				# if len(bbin) == 2: #check for edge case where wavespacing is smaller or equal to the initial binnning
				# 	if np.diff(wav[bbin]) > wavespacing:
				# 		bbin = np.array([bbin[0]]) #in that case just use original wavelength grids
				Wav_chunks.append(wav[bbin]), Idx_chunks.append(bbin)
				if bbin[-1] == len(wav)-1:#at the end of the wavelength grid
					STOP = True
				# elif bbin[-1] == len(wav)-2:#then one away from the edge. In that case have the last bin be by itself
				# 	Wav_chunks.append(wav[len(wav)-1]), Idx_chunks.append(len(wav)-1) #len(wav)-1 = last index
				# 	STOP = True
				else:
					lastWave = bbin[-1]+1 #start with one index above the last bin
					strt = wav[lastWave]
					end = wav[find_nearest(wav, wav[lastWave]+wavespacing)]
					# if strt == end: #then the desired wavespacing is smaller or equal to the model wave spacing
					# 	end =  wav[lastWave+1] #in that case, just keep the same spacing of the model	
			LowResWave = np.zeros(len(Wav_chunks))
			for w in range(len(Wav_chunks)):
				LowResWave[w] = np.median(Wav_chunks[w])
			print ("Wavelength spacing of\033[1m "+str(wavespacing)+"/"+str(np.mean(np.diff(LowResWave)))+" \033[0myields a wavelength grid of\033[1m", len(LowResWave), "\033[0mdatapoints")
	else:
		LowResWave = wav 
	"""==========================="""
	if not New_res_path: #if didn't specify folder for new data, then name is based off the grid folder where collecting original data from
		if grid_path[-1] == '/':
			grid_path = grid_path[:-1] #-1 to get ride of '/' in path naame
		New_res_path = grid_path+'_R'+str(NewRes)+'_from'+str(int(round(Range[0])))+'-'+str(int(round(Range[1])))+'A'
		if wavespacing:
			New_res_path+='_wavstep'+str(np.round(np.mean(np.diff(LowResWave)), 3)) #Save actual mean wavelength spacing as name, rather than idea wavelength spacing ('wavespacing')
	if not os.path.exists(New_res_path):
		os.mkdir(New_res_path) #Make new path to store PHOENIX models @ different resolution
	if not os.path.isfile(New_res_path+'/WAVE_'+gridbasenames+'-R'+str(NewRes)+'.fits'): #to save the new wavelength grid 
		if wavespacing and IdxChunks: #if Idx_chunks was provided in function call, then didn't define LowResWave above, must do it here
			LowResWave = np.zeros(len(Idx_chunks))
			for I in range(len(Idx_chunks)):
				LowResWave[I] = np.median(wav[Idx_chunks[I]])
			print ("Wavelength spacing of\033[1m "+str(wavespacing)+"/"+str(np.mean(np.diff(LowResWave)))+" \033[0myields a wavelength grid of\033[1m", len(LowResWave), "\033[0mdatapoints")
		fits.PrimaryHDU(LowResWave).writeto(New_res_path+'/WAVE_'+gridbasenames+'-R'+str(NewRes)+'.fits')  #only have to do this once. Degrading the resolution doesn't change wavelength grid
	grid_models = glob.glob(grid_path+'/lte*.fits')
	if PRIORS: #only save data that's within specified bounds
		grid_models_cut = []
		#The range of parameters available with the PHOENIX models.
		Grid = phoenix_rng() #holding [alpha/M] fixed to 0

		for k in list(PRIORS.keys()): #first loop through the prior directory to get the range of parameterspace allowed to be explored
			if k.lower()=='metallicity' or k.lower()=='[fe/h]' or k.lower()=='[m/h]': #this is for the metallicity
				MetalRng = WidenBounds(k, PRIORS, Grid[k])
			if k.lower()=='temp' or k.lower()=='temperature' or k.lower()=='t_eff': #this is for the photospheric effective temperature 
				TempRng = WidenBounds(k, PRIORS, Grid[k]) 
			if k.lower()=='logg' or k.lower()=='surface gravity': #this is for the surface gravity
				logGRng = WidenBounds(k, PRIORS, Grid[k])
		print ("MetalRng:", MetalRng, "TempRng:", TempRng, "logGRng:", logGRng)
		#TODO: when making genaric to any model grid. Have to use the dictonary keys as guide towards the struture of my file name strings?		

		for mod in grid_models:
			file = mod.split('/')[-1] #don't want the whole path, just the file name
			Keep = True #if any of these below criteria are not meet, then won't add particular file to 'grid_models_cut'
			for k in list(PRIORS.keys()):
				temp = float(file[3:8]) #the string name of each PHOENIX model file is the same length, with each parameter at the same location
				if TempRng[0] <= temp and temp <= TempRng[1]:
					pass 
				else:
					Keep = False
				logg = float(file[9:13]) 
				if logGRng[0] <= logg and logg <= logGRng[1]:
					pass 
				else:
					Keep = False
				metallicity = float(file[13:17]) 
				if MetalRng[0] <= metallicity and metallicity <= MetalRng[1]:
					pass 
				else:
					Keep = False
			if Keep:
				grid_models_cut.append(mod)
		grid_models = grid_models_cut #reassign 'grid_models' to only include gridpoints within my bounds
	LenM = len(grid_models)
	print ("Saving \033[1m"+str(LenM)+"\033[0m files in directory \033[1m'"+New_res_path+"'\033[0m, with Resolution = \033[1m"+str(NewRes)+"\033[0m and wavelength range \033[1m"+str(np.round(Range,2))+"\033[0mAng")
	if threads: #to split the list of files by # of threads, that way can run each list at the same time
		MultiMods = chunks(grid_models, threads) #to break the list of models into sublist, which will be calculated in parallel
		import multiprocessing
		THREADs = [] #list of threads, needed to keep track of all the threads so can join them all at the end
		for multi in MultiMods:
			THREADs.append(multiprocessing.Process(target=ReMakeGrid, args=(multi,)))
			THREADs[-1].start()
		for thread in THREADs: # now join them all so code doesn't finish until all threads are done
			thread.join()
	else:
		ReMakeGrid(grid_models)
		# for g in grid_models:
		# 	print (g)
	return None 

def LinInterpAccuracy(Range, WaveRange=None, ModelsDri=PHOENIX_path): #to test the accuracy of the linear interpolations
	"""
	Range = dictonary of 2 element lists. Where each dict key is the specific PHOENIX model parameter and their corresponding list components are the upper and lower bounds of the range of parameter space you'd like to test the accuracy of
	a Range key can also be one number (int or float), which means that we are not varying the models along that parameter
	"""
	if ModelsDri[-1] != '/':
		ModelsDri+='/'
	if WaveRange: #list of upper an lower wavelength bounds (angstroms)
		wave_data = fits.getdata(ModelsDri+'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
		Indx_bnds = np.where((WaveRange[0]<=wave_data)&(wave_data<=WaveRange[1])) #same wavegrid for each spectra, so wavelengths want to include is captured here

	print ('initial Range', Range) 
	Grid = phoenix_rng() #The range of parameters available with the PHOENIX models. #holding [alpha/M] fixed to 0 #TODO: Change to handle any arbitrary stellar spectrum model

	def AccuracyTestGrid(Gridpoint): # the gride of phoneix models datapoints, minus the specific parameters for the gridpoint I'm testing
		#Gridpoint = a dict containing the specific phoneix model parameter gridpoints, of the parameters that are being varied
		#DO NOT include parameters that you're holding fixed for the accuracy test. Those will stay in the final grid, allowing for Interpolate() to hold that parameter fixed in the interpolation proccess
		new_Grid = phoenix_rng()
		for gp_k in list(Gridpoint.keys()):
			indx = np.where(new_Grid[gp_k] == Gridpoint[gp_k])[0]
			new_Grid[gp_k] = np.delete(new_Grid[gp_k], indx) #remove gridpoint parameter from list of grid values to trip the Interpolation function
		return new_Grid

	for k in list(Range.keys()): #first loop through the Range directory to get the range of parameterspace allowed to be explored
		if type(Range[k]) == float or type(Range[k]) == int:
			Range[k] = [float(Range[k]), float(Range[k])] #not varying models along this parameter
			continue
		if np.diff(Range[k])[0] == 0: #if the upper and lower bound of the range to scan are the same, then leave this as a fixed parameter for the test
			Range[k][0], Range[k][1] = float(Range[k][0]), float(Range[k][0])
			continue
		WidenBnd = WidenBounds(k, Range, Grid[k])
		print (k+" widen bounds: "+str(WidenBnd), end =', ')
		nearestLow = find_nearest(Grid[k], WidenBnd[0]) #nearest gridpoint to the given lower bound 
		if nearestLow != 0: #NOT at the edge of the model grid's lower bound, so can go lower
			WidenBnd[0] = Grid[k][nearestLow-1] #therefore changing lower bound to one gridpoint lower. This is needed for interpolation
		nearestUp = find_nearest(Grid[k], WidenBnd[1]) 
		if nearestUp != len(Grid[k])-1: #NOT at the edge of the model grid's upper bound, so can go higher
			WidenBnd[1] = Grid[k][nearestUp+1] #therefore changing upper bound to one gridpoint larger. This is needed for interpolation
		Range[k] = [float(WidenBnd[0]), float(WidenBnd[1])]
	print ("\nextra wide bounds: "+str(Range))

	#TODO: when making genaric to any model grid. Have to use the dictonary keys as guide towards the struture of my file name strings?		
	grid_models_cut = []
	grid_models = glob.glob(ModelsDri+'/lte*.fits')
	temp_options = ['temp', 'temperature', 't_eff'] #viable names for PHOENIX models' parameters
	logG_options = ['logg', 'surface gravity']
	metal_options = ['metallicity', '[fe/h]', '[m/h]']
	for mod in grid_models:
		file = mod.split('/')[-1] #don't want the whole path, just the file name
		Keep = True #if any of these below criteria are not meet, then won't add particular file to 'grid_models_cut'
		temp = float(file[3:8]) #the string name of each PHOENIX model file is the same length, with each parameter at the same location
		logg, metallicity = float(file[9:13]), float(file[13:17])
		for k in list(Range.keys()):
			if k.lower() in temp_options:
				if Range[k][0] <= temp and temp <= Range[k][1]:
					pass 
				else:
					Keep = False
			if k.lower() in logG_options:
				if Range[k][0] <= logg and logg <= Range[k][1]:
					pass 
				else:
					Keep = False
			if k.lower() in metal_options:
				if Range[k][0] <= metallicity and metallicity <= Range[k][1]:
					pass 
				else:
					Keep = False
		if Keep:
			grid_models_cut.append(mod)
	grid_models = grid_models_cut #reassign 'grid_models' to only include gridpoints within my bounds

	SPECTRA = {} #dict of arrays where each key is the temp, logg, & [fe/h] for a specific model and it's corresponding intropolated spectrum
	PerErr_dict, PerErr_list  = {}, []#dict of the percent error for each model tested. List version is used to get the average per err for all models tested
	XYZ_str = {'Temp':0, 'LogG':0, '[Fe/H]':0} #the proper PHOENIX string formating for specific parameter
	cntTotal, cntTrue = 0, 0 # to count the number of spectra attempting to estimate the accuracy of and the actual number I ended up being able to calculate, given the edge cases
	for gridpnt_file in grid_models: #test one gridpoint at a time  
		Keep = True #if any of these below criteria are not meet, then won't interpolate spectrum at given gridpoint
		#use the adjacent gridpoints to interpolate the spectrum of the given gidpoint. Thus, can't intropolate the edge of the bounds (corresponds to 8 gridpoints in 3D)
		gridpnt = gridpnt_file.split('/')[-1]
		temp, logg, metallicity = float(gridpnt[3:8]), float(gridpnt[9:13]), float(gridpnt[13:17]) #the string name of each PHOENIX model file is the same length, with each parameter at the same location
		PHONEIX_bnds = {} #dict of lists containing the upper and lower bounds needed to interpolate values for this gridpoint
		for key in list(Range.keys()):
			if not Keep: #No need to check the other parameters if we've already flagged that it's not going to be used
				continue
			if key.lower() in temp_options:
				if np.diff(Range[key])[0] == 0:
					pass
				elif np.diff(Range[key])[0] != 0 and Range[key][0] == temp: #at the lower edge of the temp bound AND temp value isn't being held fixed. 
					Keep = False # don't interpolate the model  
				elif np.diff(Range[key])[0] != 0 and Range[key][1] == temp: #at the upper edge of the temp bounds AND temp value isn't being held fixed
					Keep = False# don't interpolate the model  
				else:
					Nearest = find_nearest(Grid[key],temp)
					PHONEIX_bnds['Temp'] = temp #use the next surrounding gridpoint for interpolation
			if key.lower() in logG_options:
				if np.diff(Range[key])[0] == 0:
					pass
				elif np.diff(Range[key])[0] != 0 and Range[key][0] == logg: #at the lower edge of the logg bound AND logg value isn't being held fixed 
					Keep = False # don't interpolate the model  
				elif np.diff(Range[key])[0] != 0 and Range[key][1] == logg: #at the upper edge of the logg bounds AND logg value isn't being held fixed
					Keep = False # don't interpolate the model 
				else:
					Nearest = find_nearest(Grid[key],logg)
					PHONEIX_bnds['LogG'] = logg #use the next surrounding gridpoint for interpolation
			if key.lower() in metal_options:
				if np.diff(Range[key])[0] == 0:
					pass
				elif np.diff(Range[key])[0] != 0 and Range[key][0] == metallicity: #at the lower edge of the metallicity bound AND metallicity value isn't being held fixed
					Keep = False # don't interpolate the model  
				elif np.diff(Range[key])[0] != 0 and Range[key][1] == metallicity: #at the upper edge of the metallicity bounds AND metallicity value isn't being held fixed
					Keep = False # don't interpolate the model 
				else:
					Nearest = find_nearest(Grid[key],metallicity)
					PHONEIX_bnds['[Fe/H]'] = metallicity #use the next surrounding gridpoint for interpolation
		if Keep: # not at the edge of the parameter space
			cntTotal +=1
			# print ("temp, logg, metallicity", temp, logg, metallicity)
			try: #There are edge cases where interpolation won’t work because @ end of grid bounds or special grid limits (ex. at T_eff=6900,logg=0.5 doesn't exisits)
				# print ("AccuracyTestGrid(PHONEIX_bnds)", AccuracyTestGrid(PHONEIX_bnds))
				SPECTRA = Interpolate({'Temp':temp, 'LogG':logg, '[Fe/H]':metallicity}, GridRange=AccuracyTestGrid(PHONEIX_bnds))[Indx_bnds]		
				# SPECTRA['T'+str(temp)+'_G'+str(logg)+'_M'+str(metallicity)] = Interpolate({'Temp':temp, 'LogG':logg, '[Fe/H]':metallicity}, GridRange=AccuracyTestGrid(PHONEIX_bnds))[Indx_bnds]
				phx_str = {}
				phx_str['Temp'], phx_str['LogG'] = PHOENIXfileCriteria(['Temp', temp]), PHOENIXfileCriteria(['LogG', logg])
				phx_str['[Fe/H]'] =  PHOENIXfileCriteria(['[Fe/H]', metallicity])
				true_data = fits.getdata(ModelsDri+'lte'+phx_str['Temp']+'-'+phx_str['LogG']+phx_str['[Fe/H]']+'.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')[Indx_bnds]
				AbsDiff = abs((SPECTRA-true_data))# The absolute difference of each spectra datapoint compared to the corresponding model's spectral datapoints
				PerErr_dict['T'+str(temp)+'_G'+str(logg)+'_M'+str(metallicity)] = np.sum(AbsDiff)/np.sum(true_data)*100.0 #to estimate the mean percent error of the interpolation compared to the true model values
				#^^^ doing sum/sum because edge cases cause massive errors when doing np.mean(spectrum/spectrum). sum/sum == mean/mean
				PerErr_list.append(PerErr_dict['T'+str(temp)+'_G'+str(logg)+'_M'+str(metallicity)])
				cntTrue+=1
			except: #In those special grid limit edge cases, just return ignore the accuracy calculation for those specific parameters
				print ("Can't interpolate the spectrum with parameters: Temp=", temp, "LogG=",logg, "Metallicity=",metallicity)
	print ("calculated the percision of\033[1m", cntTrue, "\033[0mspectra, but attempted to calculate", cntTotal)
	return PerErr_dict, np.array(PerErr_list) #SPECTRA #saving the spectra requires too much memory

def SavePHOENIXasFits(FLUX, SaveName, WAVE=None, ERR=None, path = ''): #To save PHOENIX model as .fits files cause it's faster to read and manipulate
	#don't necessarily need WAVE if the .txt file (which has flux and wave) is loaded
	if path is None or path == '': #if no path info provided, save data in current working directory
		path = os.getcwd()
	if path[-1] != '/': #make sure we got the backslash
		path += '/'
	if type(FLUX) == str and '.fits' in FLUX: #if file type is a string with 'fits' in it, given the original PHOENIX models fits file
		spectrum = fits.getdata(FLUX)*1e-8 #initially in [erg/s/cm^2/cm], convert to [erg/s/cm^2/A]
		Wav = fits.getdata(WAVE) #leave in terms of Angstroms, cause that's what all the HST conversions are in
	elif type(FLUX) == str and '.txt' in FLUX: #Assuming given a txt file with wave[A], flux[erg/s/cm^2/A], and error [0] all in same file
		Wav, spectrum, err = np.loadtxt(FLUX, unpack=True)
	else: #otherwise assumming given array components individually
		Wav, spectrum, err = WAVE, FLUX, ERR
	if WAVE is not None: #if want to save the wavelength info and spectral info in one fits file, save in formate that iSpec can read: as a fitsrec.FITS_rec
		col1 = fits.Column(name='WAVE', format='D', array=Wav) #column data
		col2 = fits.Column(name='Flux', format='D', array=spectrum)
		COLs = [col1, col2]
		prihdr = fits.Header()
		prihdr['TTYPE1'], prihdr['TTYPE2'] = 'WAVE',"FLUX" #heaeer info
		prihdr['TFORM1'], prihdr['TFORM2'] = '313111D',"313111D"
		if ERR:
			col3 = fits.Column(name='ERR', format='D', array=err)
			COLs.append(col3)
			prihdr['TTYPE3'], prihdr['TFORM3'] = 'ERR', '313111D'
		cols = fits.ColDefs(COLs)
		tbhdu = fits.BinTableHDU.from_columns(cols)
		prihdu = fits.PrimaryHDU(header=prihdr)
		thdulist = fits.HDUList([prihdu, tbhdu])
		thdulist.writeto(path+SaveName+'.fits', overwrite=True)
	else: #Otherwise, if no wavelength is given, then just save spectra as 1D array fits file
		fits.PrimaryHDU(FLUX).writeto(path+SaveName+'.fits', overwrite=True)
	print ('Saved PHOENIX spectrum as "'+SaveName+'.fits'+'" in directory "'+path+'"')
	return None

#To determine the combined stellar atmosphere, given PHEONIX parameters of 3 stars 
def CombinedStellAtmo(Save=False, save_path='', Noise=None, Gridpath=PHOENIX_path, Print=True, **Specs): 
	#Noise = flag for if you want to add noise to data. if so pass the SNR to the variable
	#specs = list of stellar parameters [temp, surface gravity, metallicity, fraction of photosphere]
	#Same as Equ 2 from Wakeford et. al 2019, where Spec1, 2, & 3 
	#are the model spectra files of different stars. And Frac2, & 3 are the surface covering fraction of those spectra
	#Save = Flag if you want to save the final spectrum. Can also provide a string that acts as a base name of the file to be saved
	if Gridpath[-1] != '/':
		Gridpath+='/'
	temp, surf_grav, Fe_h, Frac = [], [], [], []
	for spc in list(Specs.keys()):
		temp.append(Specs[spc][0]), surf_grav.append(Specs[spc][1]), Fe_h.append(Specs[spc][2]), Frac.append(Specs[spc][3])
	remain, old_sum = 1-sum(Frac), sum(Frac)
	if abs(remain) > 0.00000999: #shouldn't be greater/less than 1 up to the 5th decimal
		sys.exit("Fraction of stellar surfaces = "+str(sum(Frac))+". It needs to equal 1.\t Remainder past 1:"+str(remain))
	else: #if it's close enough to 1, round it to 1 by adding/subtracting the excess/deficiency to the highest fraction
		Frac[np.argmax(Frac)] += remain #do this because the slight change wouldn't affect the main surface of the star at all
		if remain != 0 and Print:
			print ("Total fraction of star now is "+str(sum(Frac))+", but was "+str(old_sum))
	for s in range(len(Frac)):
		if s == 0: #to initiate Final_Spec
			try: #if Interpolate can return a spectrum, quit out with returning nothing
				Final_Spec = Interpolate({'Temp':temp[s], "LogG":surf_grav[s], "[Fe/H]":Fe_h[s]}, GridPath=Gridpath, Print=Print)*Frac[s]
			except:
				return None
		else:
			try: #TODO: EDIT CODE LATER TO INCORPERATE THE THROUGHPUT OF THE INSTURMENT USED
				Final_Spec += Interpolate({'Temp':temp[s], "LogG":surf_grav[s], "[Fe/H]":Fe_h[s]}, GridPath=Gridpath, Print=Print)*Frac[s] 
			except:
				return None
	Final_Spec = Final_Spec*1e-8 #initially PHOENIX models are in [erg/s/cm^2/cm], convert to [erg/s/cm^2/A]
	if Noise: #add gaussian noise with given SNR. Gaussian noise is only valid when have high enough counts. Photon noise distribution is actually poisson
		snr = Noise
		Norm_apprx = (np.mean(Final_Spec)/(snr**2)) #snr = sqrt(N), so in order to have given counts correspond to appropriate SNR we have SNR = <sqrt(N*Norm_apprx)>
		Norm_range = np.linspace(Norm_apprx-(Norm_apprx*.1), Norm_apprx+(Norm_apprx*.1), num=1000) #but we want mean SNR to equal sqrt(N*Norm_apprx), which is NOT
		Norm_range[Norm_range <= 0.0] = 1.0e-10 #can't divide by 0
		mean_SNRs = np.zeros(len(Norm_range)) #the same as SNR = sqrt(<N>*Norm_apprx). Thus, numerically solve for the appropriate Normalization term, using 
		for N in range(len(Norm_range)): #Norm_apprx derived from SNR = sqrt(<N>*Norm_apprx) as a starting point
			# print ("Final_Spec/Norm_range[N]:", Final_Spec/Norm_range[N])
			mean_SNRs[N] = np.mean(np.sqrt(Final_Spec/Norm_range[N]))
			# print ("mean_SNRs[N]", mean_SNRs[N])
		Best_Norm = Norm_range[find_nearest(mean_SNRs, snr)]
		# print ("Best_Norm", Best_Norm)
		SNRs = np.sqrt(Final_Spec/Best_Norm) #Now use the count dependent SNR to determine the noise levles  		
		sigma = Final_Spec/SNRs #N/sqrt(N) = sigma = snr = sqrt(N), there4 to include snr: counts/snr = sigma
		sigma[sigma <= 0.0] = 1.0e-10 #minimum sig
		Final_Spec += np.random.normal(0, sigma, len(Final_Spec))
	if Save is not False and Save is not None:
		if type(Save) == str: #if Save is a string, then use that as the start of the file name
			Save_name = Save
		else:
			Save_name = ''
		Save_name+='CombinedSpecs'
		for S in range(len(temp)): #probably too many loop, but only ~3 elements so no time at ll to run through em
			Save_name += '_S'+str(S+1)+"=T"+str(temp[S])+'G'+str(surf_grav[S])+'M'+str(Fe_h[S])
		if Noise:
			Save_name +='_SNR'+str(snr)
		SavePHOENIXasFits(Final_Spec, Save_name, path=save_path) 
	return Final_Spec 

######===== Run Functions =====######
if __name__ == '__main__':
	# prior_test = {'Temp':[4150, 4470], '[Fe/H]':[0, 0], 'LogG':[4.5, 4.5]}
	DegradeGridRes(10, InitialRes=500000, grid_path=PHOENIX_path, New_res_path='PHOENIX_R1000_from3000-55000A_wavstep1.0', WavRange=[2997,55000], wavespacing=False, IdxChunkInfo=['PHOENIX_R1000_from3000-55000A_wavstep1.0/Idx_chunks_R1000.npy', True])	
	# To_interp = {'Temp':4900, "LogG":4.4, "[Fe/H]":0.1}
	# InterpSpec1 = Interpolate(To_interp)*1e-8 #*1e-8 because initially PHOENIX models are in [erg/s/cm^2/cm], convert to [erg/s/cm^2/A]
	# CombinedSpec = CombinedStellAtmo(spec1=[4900, 4.4, 0.1, .2],spec3=[7000, 4.4, 0.1, .8])
	# To_interp = {'Temp':5400, "LogG":0.0, "[Fe/H]":1.0}
	# try:
	# 	InterpSpec2 = Interpolate(To_interp)*1e-8 #*1e-8 because initially PHOENIX models are in [erg/s/cm^2/cm], convert to [erg/s/cm^2/A]
	# except:
	# 	print ("couldn't interpolate model:", To_interp)
	"""
	Accuracy_dict, Accuracy_arry = LinInterpAccuracy({'Temp':[2400, 7000], "LogG":[3.0, 5.5], "[Fe/H]":[-2.0,.5]}, WaveRange=[2500,17100])
	print ("Mean percent error:", np.mean(Accuracy_arry), '\n') #2.2565528755252724 #calculated the percision of 1692 spectra, but attempted to calculate 1692
	Accuracy_dict, Accuracy_arry = LinInterpAccuracy({'Temp':[2400, 7000], "LogG":[1.0, 3.0], "[Fe/H]":[-2.0,.5]}, WaveRange=[2500,17100])
	print ("Mean percent error:", np.mean(Accuracy_arry), '\n') #2.9607657946515546 #calculated the percision of 1341 spectra, but attempted to calculate 1410
	Accuracy_dict, Accuracy_arry = LinInterpAccuracy({'Temp':[2400, 7000], "LogG":[1.0, 3.0], "[Fe/H]":[-3.0,-2.0]}, WaveRange=[2500,17100])
	print ("Mean percent error:", np.mean(Accuracy_arry), '\n') #2.7213615608764483  #calculated the percision of 450 spectra, but attempted to calculate 470
	Accuracy_dict, Accuracy_arry = LinInterpAccuracy({'Temp':[2400, 7000], "LogG":[3.0, 5.5], "[Fe/H]":[-3.0,-2.0]}, WaveRange=[2500,17100])
	print ("Mean percent error:", np.mean(Accuracy_arry), '\n') #2.9181302426714426 #calculated the percision of 564 spectra, but attempted to calculate 564

	Accuracy_dict, Accuracy_arry = LinInterpAccuracy({'Temp':[7000, 11800], "LogG":[1.0, 3.0], "[Fe/H]":[-2.0,.5]}, WaveRange=[2500,17100])
	print ("Mean percent error:", np.mean(Accuracy_arry), '\n') #1.436245839557691 #calculated the percision of 380 spectra, but attempted to calculate 563
	Accuracy_dict, Accuracy_arry = LinInterpAccuracy({'Temp':[7000, 11800], "LogG":[3.0, 5.5], "[Fe/H]":[-2.0,.5]}, WaveRange=[2500,17100])
	print ("Mean percent error:", np.mean(Accuracy_arry), '\n') #0.9345617570400903 #calculated the percision of 895 spectra, but attempted to calculate 900
	Accuracy_dict, Accuracy_arry = LinInterpAccuracy({'Temp':[7000, 11800], "LogG":[1.0, 3.0], "[Fe/H]":[-3.0,-2.0]}, WaveRange=[2500,17100])
	print ("Mean percent error:", np.mean(Accuracy_arry), '\n') #1.3544053704644108 #calculated the percision of 117 spectra, but attempted to calculate 190
	Accuracy_dict, Accuracy_arry = LinInterpAccuracy({'Temp':[7000, 11800], "LogG":[3.0, 5.5], "[Fe/H]":[-3.0,-2.0]}, WaveRange=[2500,17100])
	print ("Mean percent error:", np.mean(Accuracy_arry), '\n') #0.8322635992816926  #calculated the percision of 261 spectra, but attempted to calculate 300

	Accuracy_dict, Accuracy_arry = LinInterpAccuracy({'Temp':[2400, 4500], "LogG":[4.5, 4.5], "[Fe/H]":[0.0,0.0]}, WaveRange=[2500,17100])
	print ("Mean percent error:", np.mean(Accuracy_arry), '\n') #1.2291326378623346 #calculated the percision of 22 spectra, but attempted to calculate 22
	Accuracy_dict, Accuracy_arry = LinInterpAccuracy({'Temp':[4500, 7000], "LogG":[4.5, 4.5], "[Fe/H]":[0.0,0.0]}, WaveRange=[2500,17100])
	print ("Mean percent error:", np.mean(Accuracy_arry), '\n')	#0.4469124245713021 #calculated the percision of 26 spectra, but attempted to calculate 26
	Accuracy_dict, Accuracy_arry = LinInterpAccuracy({'Temp':[7000, 9000], "LogG":[4.5, 4.5], "[Fe/H]":[0.0,0.0]}, WaveRange=[2500,17100])
	print ("Mean percent error:", np.mean(Accuracy_arry), '\n') #0.38278040571906613 #calculated the percision of 11 spectra, but attempted to calculate 11
	Accuracy_dict, Accuracy_arry = LinInterpAccuracy({'Temp':[9000, 11800], "LogG":[4.5, 4.5], "[Fe/H]":[0.0,0.0]}, WaveRange=[2500,17100])
	print ("Mean percent error:", np.mean(Accuracy_arry), '\n') #0.10343352916693911 #calculated the percision of 15 spectra, but attempted to calculate 15
	"""
	"""
	Accuracy_dict, Accuracy_arry = LinInterpAccuracy({'Temp':[5000, 5200], "LogG":[4.0, 4.0], "[Fe/H]":[0.0,0.0]}, WaveRange=[2500,17100])
	print ("Percent error:", Accuracy_arry, '\n')
	print ("Mean percent error:", np.mean(Accuracy_arry), '\n')
	data1 = fits.getdata('PHOENIX/lte05000-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')
	data2 = fits.getdata('PHOENIX/lte05200-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')
	AbsDiff = abs(data1-data2)
	print ("mean percent difference in between adjacent models", np.sum(AbsDiff)/np.sum(data1)*100.0)
	print ("This test gives a ",(np.sum(AbsDiff)/np.sum(data1)*100.0)/np.mean(Accuracy_arry),"times increase in precision")
	"""
	# data1 = CombinedStellAtmo(Gridpath=PHOENIXR500_5step_path, Noise=400.03321621031404, spec1=[3030.71780439874, 4.5, 0.0, 0.5282827022605407], spec2=[1510.4397160009332, 4.5, 0.0, 0.4717172977394593])
	# data2 = CombinedStellAtmo(spec1=[4900, 4.4, 0.3, 1], Gridpath=PHOENIXR500_5step_path, Noise=20)
	# Wave1 = fits.getdata("PHOENIX_R500_from2500-17100A/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
	# Wave2 = fits.getdata(PHOENIXR500_5step_path+"WAVE_PHOENIX-ACES-AGSS-COND-2011-R500.fits")
	# data1R500 = fits.getdata("Res500_2500-17100Ang/SYNTHSPECmet0.0_grav4.5tempA2961tempB1505frac0.45484.fits")
	# data2R500 = fits.getdata('PHOENIX_R500_from2500-17100A/lte04600-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')
	# WaveR500 = fits.getdata("PHOENIX_R500_from2500-17100A/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
	# Combined = fits.getdata('CombinedSpecs_S1=T4900G4.4M0.3_S2=T3000G4.4M0.1_S3=T7000G4.4M0.3.fits')
	# Combined_noise = fits.getdata('CombinedSpecs_S1=T4900G4.4M0.3_S2=T3000G4.4M0.1_S3=T7000G4.4M0.3_SNR20.fits')
	# data1 = fits.getdata('PHOENIX_R500_from2500-17100A/lte04900-4.50+0.5.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')*1e-8
	# plt.figure(1)
	# plt.plot(WaveR500, Combined, 'g--')
	# plt.plot(WaveR500, Combined_noise, 'r.', markersize=.1)
	
	# plt.figure(2)
	# plt.plot(WaveR500, data1, 'r')
	# plt.plot(WaveR500, Combined, 'g--')
	
	# plt.figure(3)
	# plt.plot(Wave, data2, 'g')
	# plt.plot(Wave, data1, 'r--')
	
	# plt.figure(4)
	# plt.plot(data1R500*1e-8, 'b')
	# plt.plot(Wave2, data2, 'r.')
	# # plt.plot(Wave, T5800G45Mmin05, 'b')
	# Temp, Log_G, Fe_H = 5700, 4.5, -0.5
	# InterpSpec = Interpolate([Temp, Log_G, Fe_H], GridPath=PHOENIX_path)
	# plt.figure(1)
	# plt.plot(Wave, InterpSpec1, 'b')
	# plt.xlim([2500,17100])
	# plt.figure(2)
	# plt.plot(Wave, InterpSpec1-CombinedSpec, 'k')
	# plt.xlim([2500,17100])
	# plt.figure(5)
	# Interp1 = Interpolate({'Temp':3030.71780439874, "LogG":4.5, "[Fe/H]":0.0}, GridPath=PHOENIXR500_5step_path)*0.5282827022605407
	# plt.plot(Interp1)
	# plt.figure(6)
	# Interp2= Interpolate({'Temp':1510.4397160009332, "LogG":4.5, "[Fe/H]":0.0}, GridPath=PHOENIXR500_5step_path, Print=True)*0.4717172977394593
	# plt.plot(data1R500)
	# plt.figure(7)
	# plt.plot(Interp1+Interp2)
	# plt.show()
	# plt.close()


