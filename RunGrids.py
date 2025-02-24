#Code to run my grid of tests
import pickle
import os
import sys
import glob
import time
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
main_dir = '/Users/chimamcgruder/Research/'
sys.path.append(main_dir+'TwinPlanets/')
sys.path.append(main_dir+'iSpec_v20201001')
import ispec 
import ModelAtmosphere as MA	
PHOENIX_constant = 'PHOENIX-ACES-AGSS-COND-2011-' #string that's found in all PHOENIX data
PHOENIXR500_5step_path = '/Users/chimamcgruder/Research/KnowThyStar/PHOENIX_R500_from2500-17100A_wavstep5.0/'
SavePath = 'Res500'
#Have to do each resolution seperately  

# ((2*3*2*1*2*100*20)/60)/20 = 40hrs
def MakeSpectrum(Temp1s, Temp2s, Frac2s, WaveRanges, SNRs, Fe_H, Log_G, ModelsPerType=100, varyConts=False, varyScale = .3,\
	TempStep=75, MetalStep=0.5, GravStep=0.5, FracStep=0.05, SNRstep=5, SynthPath=SavePath, GridPath=PHOENIXR500_5step_path):
	"""To make an array of synthetic data based on the range of parameters provided. Will vary the stellar paramters slightly per iteration
	varyScale = fraction of variable, relative to provieded stepsize, in which the std of that variable will be varied. Varying all variable 
				as a gaussian with given mean & std calculated from varyScale. If you you don't like the std of one paraticular parameter, then 
				change the Step size of that variable. Only change varyScale when trying to change std of most parameters.
	varyConts = If you want to vary the terms that aren't going to be explored in the posterior. This would simulate our inherent uncertaintiy in those parameters
	"""
	Temp_limits = [2300, 12000] #temperatures outside of this bound will be changed to closest value
	TotalModels = len(Temp1s)*len(Temp2s)*len(Frac2s)*len(SNRs)*ModelsPerType #number of synthetic spectra that must be made given the specified parameters (not including range of wavelengths!!!)
	#For the PHONEIX model data will need all of these paramters. TODO: have the loops for each parameter, arbitarty so can take in any number of params for any given model	
	#To generate the different parameters, with the specified level of variance for each parameter
	start, start1, start2 = 0, 0, 0
	def SynthDatParam(surfaceG, metallicity,wav_index,File_Dict, SynthPath, OG_grav=None,OG_Met=None): 
		"""call this function for each wavelength range and each 'ModelsPerType' TotalModels, ModelsPerType, parameters (i.e. Temp1s, Frac2s), and 
		parameter step sizes (i.e. TempStep), don't change throughout. Thus, don't need to pass it in the fucntoin and just keep it as a global variable
		OG_grav/OG_Met = have to keep track of the original metallicitiy and gravity before noise was added for the 'File_Dict' """
		# cnt = 0
		if not OG_grav:
			OG_grav = surfaceG
		if not OG_Met:
			OG_Met = metallicity			
		ModelTypes = TotalModels/ModelsPerType
		for T1 in Temp1s: 
			for T2 in Temp2s:
				for F in Frac2s:
					for snr in SNRs:
						t1 = np.random.normal(T1, TempStep*varyScale) # extract parameters at the end because want the random iteration to be unique for EACH permutation of each param
						t2 = np.random.normal(T1+(T1*T2), TempStep*varyScale)
						f = abs(np.random.normal(F, FracStep*varyScale))
						s_n_r = np.abs(np.random.normal(snr, SNRstep*varyScale))
						if t2 < Temp_limits[0]:
							t2 = Temp_limits[0]
						if t2 > Temp_limits[1]:
							t2 = Temp_limits[1]
						# cnt +=1
						# print ("t1:", t1, "t2:", t2, "f:", f, 's_n_r:', s_n_r)
						# print ("cnt:", cnt)
						Final_Spec = MA.CombinedStellAtmo(Noise=s_n_r, Gridpath=GridPath, spec1=[t1, surfaceG, metallicity,1-f], spec2=[t2, surfaceG, metallicity, f], Print=True)
						# print ("Temp1:", drawnTemp1s[i], "Temp2:", drawnTemp2s[i], "Frac2:", drawnFracs[i], "SNR:", drawnSNRs[i])#, "\nClean Values:", drawnTemp1s_clean[i], drawnTemp2s_clean[i], drawnFracs_clean[i], drawnSNRs_clean[i],"\n")
						File_name = 'SYNTHSPECmet'+str(np.round(metallicity,2))+'_grav'+str(np.round(surfaceG,2))+\
							'_tempA'+str(int(t1))+'_tempB'+str(int(t2))+'_frac'+str(np.round(f,5))+'_snr'+str(np.round(s_n_r,1))+'.fits'
						fits.PrimaryHDU(Final_Spec[wav_index]).writeto(SynthPath+'/'+File_name)
						Key = "met"+str(OG_Met)+"_grav"+str(OG_grav)+"_tempA"+str(int(T1))+"_tempB"+str(int(T1+(T1*T2)))+"_frac"+str(F)+"_snr"+str(int(snr))
						if Key in list(File_Dict.keys()):
							File_Dict[Key].append(File_name)
						else:
							File_Dict[Key] = [File_name]
		return File_Dict

	if varyConts: #only vary metellicity and gravity for each iteration. So these variables don't vary per wavelength or specific parameter permutation
		Metallicities = np.random.normal(Fe_H, MetalStep*varyScale, ModelsPerType)
		Gravities = np.random.normal(Log_G, GravStep*varyScale, ModelsPerType)

	wavefile = glob.glob(GridPath+"WAVE*.fits")
	if len(wavefile) > 1:
		sys.exit("Multiple wavelength grids found in '"+GridPath+",' don't know which to use")
	WaveGrid = fits.getdata(wavefile[0])
	for wr in WaveRanges:
		FilDir = {} #to keep track of the original (before adding noise) parameter set up each file has. New file for each wave range and resolution
		Full_synthPath = SynthPath+'_Wav'+str(int(wr[0]))+'-'+str(int(wr[1]))+'A'
		WavIndx = np.where((wr[0]<=WaveGrid)&(WaveGrid<=wr[1]))[0]
		print ("Making ", TotalModels, "synthetic spectra for wavelength range of: "+str(int(wr[0]))+'-'+str(int(wr[1])))
		if not os.path.exists(Full_synthPath): #make the folder where all data with common wavelength range and Resoltion will be located (resolution specified in 'SynthPath')
			os.mkdir(Full_synthPath)
		for i in range(ModelsPerType): #same base parameters, but only vary based on the noise interoduced in each parameter
			if varyConts:
				FilDir = SynthDatParam(Gravities[i], Metallicities[i], WavIndx, FilDir, Full_synthPath, OG_grav=Log_G, OG_Met=Fe_H)
			else:
				FilDir = SynthDatParam(Log_G, Fe_H, WavIndx, FilDir, Full_synthPath) #1st pass an empty 'FilDir', edit it for that run then repass it in the 'ModelsPerType' loop
		FILE = open(Full_synthPath+'/FileDirectory.pkl', 'wb')
		pickle.dump(FilDir, FILE)
		FILE.close()		
	return None

def AnalyzeSpec(PickleFiles, jobQuota, threads=20, Maxcalls=60000, GridPath=PHOENIXR500_5step_path, wait_T=10): #to submit jobs to analyze synthetic data
	#jobQuota = number of sampling runs that should be submitted in one job
	JobCnt, JobSubmissions, totalCnt = 0, 0, 0 # to keep track of the number of sampling runs in one code, the number of jobs submitted, & the total counts, respectively
	for P in PickleFiles: #for each folder name given (a different folder is used for differing res/wave-range)
		NewFolder = True
		print ("Working with files in '"+P.split('/')[-2]+"' directory")
		File = pickle.load(open(P, "rb"))
		for typ in list(File.keys()): #for each parameter permutation in the specific folder
			typ_cnt = 0 #to keep track of what count of that specific parameter permutation
			for fil in File[typ]: #for each iteration with the given parameter type
				if JobCnt == 0: # This will be the start of a new file for a new job submission
					JobTitle = 'DynSampGrid_Num'+str(JobSubmissions)
					Py_script = open(JobTitle+'.py', 'w')
					Py_contents ="import os\n"
					Py_contents +="import sys\n"
					Py_contents +="import numpy as np\n"
					Py_contents +="import time\n"
					Py_contents +="main_dir = /Users/chimamcgruder/Research/'\n"
					Py_contents +="sys.path.append(main_dir+'/KnowThyStar/')\n"
					Py_contents +="import ModAtmo_Dynesty_class as SpecDecomp\n"
					Py_contents +="###############========= USER INPUT =========###############\n"
					Py_contents +="#The dictionary keys are the name of each parameter and the element are lists containing the name of the prior type and that prior function input\n"
					Py_contents +="#CHECK UNITS!!!!! Initially PHOENIX model is in units of [erg/s/cm^2/cm], but data and sampling is done in [erg/s/cm^2/A]\n"
					Py_contents +="main_dir += 'KnowThyStar/'\n"
					Py_contents +="GridPath = '"+GridPath+"' #path of where model grid is located. Be cognizant to make sure the path has the proper resolution\n"
					NewFolder = True #if having to initilize a new file then by default using a new folder

				if NewFolder: #everytime working with a new folder, add this information
					wavefile = glob.glob(GridPath+'/WAVE*.fits') #assumming the wavelength grid for synthetic data is same as what models used to sample posterior will have
					if len(wavefile) != 1:
						sys.exit("Expecting to find 1 wavelength grids file in '"+GridPath+".' Found "+str(len(wavefile)))
					Py_contents +="Wave_file = '"+wavefile[0]+"' #name (and path) of wavelength grid fits file corresponding to data's spectrum\n"
				
				#Do this for every iteration 	
				met, grav = typ[typ.find('met')+3:typ.find('_grav')], typ[typ.find('_grav')+5:typ.find('_tempA')]
				tempA, tempB = typ[typ.find('_tempA')+6:typ.find('_tempB')], typ[typ.find('_tempB')+6:typ.find('_frac')]
				frac, snr = typ[typ.find('_frac')+5:typ.find('_snr')], typ[typ.find('_snr')+4:]
				outfile =  "M"+met+"_G"+grav+"_Ta"+tempA+"_Tb"+tempB+"_F"+frac+"_S"+snr
				PATH = '' #to get full path name of where synthetic data is stored 
				patH = P.split('/')[:-1]
				for p in patH:
					PATH += p+'/'
				Py_contents +="####\n#########\n###############\n"
				Py_contents +="startT = time.time()\n"
				Py_contents +="Data_file = '"+PATH+fil+"'\n"
				Py_contents += "Params = {} #assuming surface gravity and metallicity is constrained from high res spectra\n"
				temp_initiation = int(np.round(np.random.uniform(float(tempA)-50, float(tempA)+50)/10)*10) # to signify a 100K uncertainty in our starting temp prior. rounding to nearest 100th
				Py_contents += "Params['temp_eff0'], Params['temp_eff1'] = ['normal',"+str(temp_initiation)+",200], ['normal',"+str(temp_initiation)+",1000]\n"	 #assuming only fitting for 2 surfaces and no metallicity or surface gravity
				Py_contents += "Params['fract0'], Params['fract1'] = ['trunc-normal', .6,.3,0,1], ['trunc-normal', .4,.3,0,1]\n" #keep fraction priors pretty wide
				Py_contents += "ResultsFileName = '"+PATH+"Dynesty_"+outfile+"run"+str(typ_cnt)+"'\n"
				Py_contents += "initilization"+str(JobCnt)+"= SpecDecomp.DynestySampling(Params, GridPath, Data_file, Wave_file, ResultsFileName, Log_G="+grav+", fe_h="+met+")\n"
				Py_contents += "initilization"+str(JobCnt)+".Run(nthreads="+str(threads)+", MaxCalls="+str(Maxcalls)+")\n"
				Py_contents += "print ('Run time for run "+str(JobCnt)+": '+str((time.time()-startT)/60.0)+'min')\n"

				if JobCnt == jobQuota-1:# thus this is the last element of this specific python script
					Py_script.write(Py_contents) #write and 
					Py_script.close() #close up the file 
					#then create the corresponding .job file
					job_script = open(JobTitle+'.job', 'w')
					job_contents ="# /bin/sh\n# ----------------Parameters---------------------- #\n#$ -S /bin/sh\n"
					job_contents += "#$ -pe mthread "+str(threads)+"\n#$ -q sThC.q\n#$ -l mres="+str(int(threads*5))+"G,h_data=5G,h_vmem=5G\n#$ -cwd\n#$ -j y\n"
					job_contents +="#$ -N "+JobTitle+"\n#$ -o "+JobTitle+".log\n#$ -m bea\n#$ -M chima.mcgruder@cfa.harvard.edu\n"
					job_contents +='#\n# ----------------Modules------------------------- #\nexport PATH="/home/mcgruderc/.conda/envs/iSpec3/bin/:$PATH"\n'
					job_contents +="export PATH=$PATH:$HOME/.local/bin/\n#\n# ----------------Your Commands------------------- #\n#\n"
					job_contents +="echo + `date` job $JOB_NAME started in $QUEUE with jobID=$JOB_ID on $HOSTNAME\n#\n"
					job_contents +="python "+JobTitle+".py\n#\n"+repr("echo = \n")[1:-1]+"\necho = `date` $JOB_NAME done\n~"
					job_script.write(job_contents)
					job_script.close()
					
					#and submit the job
					# os.system('qsub '+JobTitle+'.job')
					# print ("\n")# since make a new job file for every submission, don't have to wait for the job to finish before moving on. 
					# time.sleep(60) #Just wait 60sec so there's a lag between job submissions					
					JobSubmissions += 1
					JobCnt = -1 #preventing JobCnt to ever equal or exceede jobQuota
				JobCnt +=1
				typ_cnt +=1
				totalCnt +=1
				NewFolder = False
	try: #if a file is still opened, finish writing the contents and close it here
		Py_script.write(Py_contents)
		Py_script.close()
	except:
		pass		
	return None

######===== Run Functions =====######
if __name__ == '__main__':
	# Main_temps = [3000, 8500]#[3000, 4500, 6000, 7500, 9000] #for temperature @ 3000K, CAN'T go down to -.5 heterogenity temp. Will have to be creative to ignore those regions
	# Hetero_temps = [.4, -.05, -.2]#[.5,.2,.05,-.05,-.2,-.5] #percent difference of heterogeniety temperature
	# stellar_fracs =  [.45, .02]#[.45, .25, .1, .02] #fractional size of heterogeniety
	# Wav_range = [[2500,17100],] #[[2500,4500], [4500.5900], [5900-7000], [7000,17000], [17000,50000]]
	# SNR_range = [400, 10]#[400, 200, 75, 35, 10]
	# fe_h, logG = 0.0, 4.5
	# MakeSpectrum(Main_temps, Hetero_temps, stellar_fracs, Wav_range, SNR_range, fe_h, logG, ModelsPerType=5)

	# File = pickle.load(open("Res500_Wav2500-17100A/FileDirectory.pkl", "rb"))
	# f_cnt = 0
	# for f in list(File.keys()):
	# 	print ("'"+f+"': "+str(File[f]))
	# 	f_cnt +=1
	# print ("f_cnt", f_cnt)

	# AnalyzeSpec(['Res500_Wav2500-17100A/FileDirectory.pkl'], 24)

