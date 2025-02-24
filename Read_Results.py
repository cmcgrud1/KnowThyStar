import numpy as np
import pickle 
from dynesty import NestedSampler
import matplotlib.pyplot as plt
from dynesty import plotting as dyplot
from dynesty.utils import resample_equal
import os
import sys
import glob
from astropy.io import fits
main_dir = '/Users/chimamcgruder/Research/'
sys.path.append(main_dir+'KnowThyStar/')
import ModelAtmosphere as MA	

def PosteriorResults(File, Quantiles, SaveCorn=False, Names=None, TrueVals=None): #to print the results of one posterior distribution of one specific (synthetic) dataset
	#SaveCorn = if you want to save the corner plot. Also doubles up as the label for the cornerplot. If SaveCorn is a string then will save the cornerplot with SaveCorn as the file name
	PATH = ''#to get full path name of where synthetic data is stored 
	patH = File.split('/')[:-1]
	for p in patH:
		PATH += p+'/'
	results = pickle.load(open(File, 'rb'))
	fg, ax = dyplot.cornerplot(results, color='blue', show_titles=True, max_n_ticks=5, quantiles=np.array(Quantiles)/100.0, truths=TrueVals, labels=Names,\
	title_kwargs={'fontsize':12,'fontweight':'bold'}, label_kwargs={'fontsize':15,'fontweight':'bold'})
	weights = np.exp(results['logwt']-results['logz'][-1])
	posterior_samples = resample_equal(results.samples, weights)
	i = 0
	for s in range(posterior_samples.shape[1]):
		if Names:
			Name = Names[s]
		else: #if the names of each parameters is not specified, then just label it Xi...Xn
			Name = 'X'+str(i)
		Quants = np.percentile(posterior_samples[:,s],Quantiles)
		print (Name+": "+str(Quants[1])+"+"+str(Quants[2]-Quants[1])+"-"+str(Quants[1]-Quants[0]))
		i +=1
	if SaveCorn:
		if type(SaveCorn) == str:
			FigName = SaveCorn
		else: #if no specific file name for the cornerplot is given, just save it as the same name as the pickle file data was extracted from
			fileName = File.split('/')[-1]
			FigName =  fileName[:fileName.find('.pkl')]
		fg.savefig(PATH+FigName+'.png')
		print ("Saved corner plot as '"+FigName+".png' in directory: '"+PATH+"'")
	plt.show()
	plt.close()

def Data_vs_Fit(Data, Best_parms, PHOENIX_path): #To plot the data vs best fit nested sampling results
	#Data = the file names for the synthetic data and it's corresponding wavelength range
	#PHOENIX_path = path of phoenix models, in which you should collect the data from
	#Best_parms =  Stellar params found for best fit: [Ta, 1-spot_covering_frac, Tb]
	data, wave = fits.getdata(Data[0]), fits.getdata(Data[1])
	WavRange = [wave[0], wave[-1]]
	FracB = 1-Best_parms[1]
	BestFitDat = MA.CombinedStellAtmo(spec1=[Best_parms[0],4.5,0.0,Best_parms[1]], spec3=[Best_parms[2],4.5,0.0,FracB], Gridpath=PHOENIX_path) #log G and metallicity are held fixeed at 4.5 and 0, respectively
	print ("BestFitDat", BestFitDat)
	WAVs = glob.glob(PHOENIX_path+'/WAVE*.fits')
	BestFitWav = fits.getdata(WAVs[0])
	Idxbounds = np.where((WavRange[0]<=BestFitWav) & (BestFitWav<=WavRange[1]))[0]
	plt.plot(wave, data, 'g.', label='synthetic data')
	plt.plot(BestFitWav[Idxbounds], BestFitDat[Idxbounds], 'r--', label='best fit')
	plt.legend(prop={'size':20, 'weight':'bold'})
	plt.xlabel("Wavlength [$\AA$]", fontsize=20,fontweight='bold')
	plt.ylabel("Flux [erg/s/cm^2/$\AA$]", fontsize=20,fontweight='bold')
	plt.xticks(fontsize=16, fontweight='bold')
	plt.yticks(fontsize=16, fontweight='bold')
	plt.show()
	plt.close()

def Histograms(DataFile, display_paramets=None, display_type=None, TrueValues=True, maxColumns=4): #to display histagrams of the results of the slew of grid points
	#display_paramets = list of string, where each element is the same key that was used to save the grid results for each parameter perturbation. ex. 'met0.0_grav4.5_tempA8500_tempB6800_frac0.02_snr10'
	#display_type = list of strings, where each element is the same key that was used to save the specific statistic. i.e. 'Primary_Temp',  'Primary_Temp_Accuracy', 'Primary_Temp_uncertainties'
	#TrueValues = Boolean, if you want to plot the true value on the parameter histograms. This is only applicable for 'Primary_Temp', 'Secondary_Temp', 'Primary_SurfaceFraction', and 'Secondary_SurfaceFraction'
	#DataFile = the results files produced by the 'Statistics' function in the 'RunGrids.py' script
	Parameters = []
	if TrueValues: #all of the parameters where it makes sense to plot a true value on the histogram
		Parameters = ['Primary_Temp', 'Secondary_Temp', 'Primary_SurfaceFraction','Secondary_SurfaceFraction']
	PATH = '' #to get full path name of where the grid test results are stored 
	patH = DataFile.split('/')[:-1]
	for p in patH:
		PATH += p+'/'
	RESULTS = pickle.load(open(DataFile, 'rb'))
	for typ in list(RESULTS.keys()): #display each parameter perturbation specified. If no display_parameters are specified, then display them all
		if not display_paramets or typ in display_paramets:	
			fig = plt.figure(typ) #New figure for each parameter perturbation
			met, grav = typ[typ.find('met')+3:typ.find('_grav')], typ[typ.find('_grav')+5:typ.find('_tempA')] #but rather label each iteration with a different count: 'typ_cnt'
			tempA, tempB = typ[typ.find('_tempA')+6:typ.find('_tempB')], typ[typ.find('_tempB')+6:typ.find('_frac')]
			frac, snr = typ[typ.find('_frac')+5:typ.find('_snr')], typ[typ.find('_snr')+4:]
			Params = [int(tempA), int(tempB), 1-float(frac), float(frac)] #Only used when TrueValues flag is on. in same order as 'Parameters' 
			Main_title = '[Fe/H]='+met+', logG='+grav+', Main T_eff='+tempA+', Secondary T_eff='+tempB
			Main_title += ',\n Main covering fraction='+str(1-float(frac))+', Secondary covering fraction='+frac
			stats_list = list(RESULTS[typ].keys())
			Main_title += ', SNR='+snr+', iterations='+str(len(RESULTS[typ][stats_list[0]])) #assuming that there's the same number of iterations for each stat (should be the case)
			fig.suptitle(Main_title, fontsize=20)
			cnt = 1
			for stat in stats_list: #subfigures for each specific stat
				if not display_type:
					SubplotS = len(stats_list)#number of subplots
				elif stat in display_type:
					SubplotS = len(display_type)#number of subplots
				else:
					continue #otherwise don't use this specific stat, and skip this loop iteration
				if SubplotS < 2*maxColumns: #if less than 2*desired number of columns (meaning can't have 2 full rows), just let len(columns)~len(rows)
					rows = int(round(np.sqrt(SubplotS)))
					columns = rows
				else:
					rows, columns = int(round(SubplotS/maxColumns)), int(maxColumns) #want no more than maxColumns columns per row
				RCI = [rows, columns, cnt] #subplot format: number of rows, number of columns, and specfiic index
				ax = plt.subplot(RCI[0], RCI[1], RCI[2])
				ax.set_title(stat)
				statistic = RESULTS[typ][stat]
				if stat in Parameters:
					plt.axvline(x=Params[Parameters.index(stat)], linewidth=4, color='r')
				ax.hist(np.array(statistic), bins='auto')
				if (cnt-1)%maxColumns == 0: # to label the y axis on every 1st element of each row
					ax.set_ylabel('counts')
				cnt += 1
			fig.set_figheight(int(5*rows)) #figure size set based on the number of columns and rows
			fig.set_figwidth(int(5*columns)) #times 5 for each row/column in the height/width directions
			saved_fig = PATH+subdir+'/'+typ+'.png'
			fig.savefig(saved_fig) #have to save the figures, otherwise would have to look at each figure 1 at a time as the code runs
			print ("saved histograms as '"+PATH+typ+".png'")
			plt.close()

def TrianglularPlots(DataFiles, AxisLabeling=None, PlotShape=['v', 'o', '*'], plotStep=.05, Print=True, SavePath='', ExtraNaming= ['',''], skip_params=None, Stellar_surfaces=2, overwrite=False, MinScales=None, plot_scaling=5, FontSize=13): 
	"""
	To help visulize the results. These are not quite conner plots, but similar idea. Think name is 'Giant Triangle Confusogram' (GTC) They show how two parameters correlate with a 'good fit'. 
	Essentially a 3d plot where the x and y D are the 2 parameters that we are comparing and the z D is either how close the mean is to the ture value, or the std of the draws
	*skip_params == list of strings, where each element is a specific parameter to skip in the GTC. by default includes ['Res','SNR','Wave','Primary_Temp', and stellar parameters]
	*SavePath == path of where to store store resulting figures. By default stored in cwd
	*ExtraNaming == prefix and suffix for the name of the save triangle plot .png image
	*PlotShape = shapes used for plotting. Needs to be the same length as the number of stellar parameters
	*plotStep = how much one stellar parameter should be offset from another
	*MinScales == two element list. if provided this will be the minumum accuracy and precision, respectively to be plotted on the z-axis color scale
	*AxisLabeling == a dictonary of strings. Where each key is the naming scheme of every parameter that'll be used, and it's corresponding string value is the name that you want for that axis
	*plot_scaling == How each corner plot will be scaled based on in size. This number is multipled by the number of coner plot, subplots (shape)
	"""

	WaveRanges = [] #list of strings that provide the wavelength range of specific samples. This is needed so we can write range 1, 2, 3 etc on the axes
	"""to keep track of the parmaeter names that will be displayed (the components of the Giant Triangle Confusogram)"""
	StellarParams = ['Primary_Temp'] #to keep track of just the stellar params, which will be the basis of each plot
	Params_2_display = ['Res','SNR','Wave','Primary_Temp'] #to keep track of all parameters affecting the fit (i.e. stellar and observational)
	if Stellar_surfaces > 1: #to get a list of all stellar surfaces to plot. TODO: also include capability of adding other model parameters (i.e. metallicity/ surface grav)
		StellarParams.append('Primary_SurfaceFraction'), Params_2_display.append('Primary_SurfaceFraction') #don't need to include 2ndary surface fraction if only 1 stellar surface
		StellarParams.append('Secondary_Temp'), Params_2_display.append('Secondary_Temp')
	if Stellar_surfaces > 2:
		StellarParams.append('Secondary_SurfaceFraction'), Params_2_display.append('Secondary_SurfaceFraction')
		cnt = 2 # cause already did 1st 2 surfaces
		while cnt < Stellar_surfaces+1:
			StellarParams.append('Temp_'+str(cnt+1)), Params_2_display.append('Temp_'+str(cnt+1))
			if Stellar_surfaces > cnt+1: #have one less surface fraction from the total number of photospheres
				StellarParams.append('SurfaceFraction_'+str(cnt+1)), Params_2_display.append('SurfaceFraction_'+str(cnt+1))
	
	"""To make a list of all the different parameter combinations, after removing parameters in 'skip_params' from my 'Params_2_display' list. This will be how I use to organize the data for plotting"""
	if skip_params:
		for skp in skip_params:
			Params_2_display.remove(skp)
	tot_parms = len(Params_2_display)
	tri_num = sum(range(tot_parms)) # to calculate the triangle number for a specific number of parameters. i.e N + N-1 + N-2 + N-3.... + 0. but here N = tot_parms-1 because GTC has 1 less than total params
	p2d_cnts = 0
	ParamCombos = []
	for p2d in range(tot_parms-1): #this nested for loop should yield dictonary element length of tri_num(tot_parms-1)
		base_i = Params_2_display[p2d] #this will be combined with all preceding parameters
		list_i = Params_2_display[p2d+1:]
		for p2d_i in list_i: #below x = base_i, y = p2d_i, and z = the scatter point color
			ParamCombos.append(base_i+' vs. '+p2d_i) 

	"""initiate the dictionaries storing all parameter results. Each dir element will be a x2 array. One for the value of the dic key and another for the specific Accuracy/Percision values"""
	Accuracies, Precisions = {}, {} #dictionaries of dictionaries of arraies. where the 1st dict layer is the specific stellar parameter info, 2nd is the retrieval parameter, and 3 are the accuracy results
	R = pickle.load(open(DataFiles[0], 'rb')) #assuming that all subsequent files in 'DataFiles' has the same length as the 1st one, which should be the case if I run the RunGrids.py script symetrically
	Num_ParamPertubations = len(DataFiles)*len(R) #to keep track of how many different parameter perturbation files there are. This is [Num_ParamPertubations*=n, for n in Params_2_display]
	for stel_par in StellarParams:
		Accuracies[stel_par], Precisions[stel_par] = {}, {} #defining the stellar param dict layer
		for disp in Params_2_display:
			Accuracies[stel_par][disp], Precisions[stel_par][disp] = np.zeros((2, Num_ParamPertubations)), np.zeros((2, Num_ParamPertubations))
	"""To sort all the data based on parameters"""
	cnt = 0
	max_str = 0 # to get the max length of the stellar param names
	for s in StellarParams: #so can make sure all strings are printed in alignment with the longest Stellar_surface string name
		s_len = len(s)
		if s_len > max_str:
			max_str = s_len
	print_str = "Param"+' '*(max_str-len('Param'))+"\t\t Model \t\t Mean Accuracy \t\t Mean Precision\n"  #just for printing purposes #only gonna be used if Print==True
	WaveCnt = 0 #to keep track of the wavelength number we on
	for D in DataFiles: #there's a specific wavelength range and resolution for each file, D
		results = {} #To keep track of the specific parameters for this draw, and the counts
		RESULTS = pickle.load(open(D, 'rb'))
		results['Res'] = int(D[D.find('Res')+3:D.find('_Wav')]) #key name of each dictionary element is so we have the string name associated with this variable. Used as ParamCombos keys
		Wave_Rng = D[D.find('Wav')+3:D.find('Ang')] 
		slash = Wave_Rng.find('-') 
		Wave_Rng = Wave_Rng[:slash]+'--\n'+Wave_Rng[slash+1:]#to make the start and end of the wavelength range on seperate lines, for plotting purposes
		if Wave_Rng in WaveRanges: #if the specific wavelength range is already specified, use it's the same indexing to label the wavelength range again
			results['Wave'] = WaveRanges.index(Wave_Rng)
		else: #otherwise assign a uniqure number 0 to len(WaveRanges)-1 to the specific wavelength range. So just using the order in which that wavelength range was used
			results['Wave'] = WaveCnt
			WaveRanges.append(Wave_Rng)
			WaveCnt +=1
		MeanAccuracies = {} # dictonary to store all different types of accuracy (i.e. TempA, B, FracA, etc.). #Here the accuracy is 100 - %error: 100 -  (|distrubition_mean-True_value|/True_value)*100
		MeanPercisions = {} #NOTE: don't need to worry about the fact that the created synthetic data has a variance, because the uncertainty is determined by the varicance around the specific posteriors mean. This is relative mean precision = (mean(std)/mean)*100
		for typ in list(RESULTS.keys()): #for each parameter perturbation of a given pickle file
			met, grav = typ[typ.find('met')+3:typ.find('_grav')], typ[typ.find('_grav')+5:typ.find('_tempA')] #TODO: need to edit this to take an arbitray amount of temperatures and stellar fractions!!!!
			results['Primary_Temp'] = int(typ[typ.find('_tempA')+6:typ.find('_tempB')])
			Secondary_Temp = int(typ[typ.find('_tempB')+6:typ.find('_frac')]) #T1+(T1*T2) #the actual secondary temp is used to calculate the accuracy of the retireval, but for plotting purposes saving the temperature contras
			results['Secondary_Temp'] = [Secondary_Temp, (Secondary_Temp-results['Primary_Temp'])/results['Primary_Temp']] #storing both the true seconday temp and the secondary temp relative to the primary temp
			results['Primary_SurfaceFraction'], results['SNR'] = 1-float(typ[typ.find('_frac')+5:typ.find('_snr')]), int(typ[typ.find('_snr')+4:])
			for StelPar in StellarParams: #here, even though the individual parameter used for a specific synthetic test data was not the same, because it was varied with a guassian distribution.
				MeanAccuracies[StelPar] = 100 - (np.mean(RESULTS[typ][StelPar+'_Accuracy'])) #Taking the mean of accuracies, where each accuracy is calculated from the difference of that specific variants mean and the retrieval result
				Uncertainties = np.array(RESULTS[typ][StelPar+'_uncertainties']) #need to conver to array so can multiply every value by .5
				if 'Temp' in StelPar and 'Primary' not in StelPar: #for non-primary photospheric temps, there are 2 numbers saved. The actual temp of the 2ndary photosphere and the temp relative to the primary temp
					MeanPercisions[StelPar] = 100 - (np.mean(Uncertainties*.5)/results[StelPar][0])*100 #uncertainty times .5, because want the std of data, which is half the error width (if gaussian)
					print_str += StelPar+' '*(max_str-len(StelPar))+' \t '+str(results[StelPar])+' \t '+str(MeanAccuracies[StelPar])+' \t '+str(MeanPercisions[StelPar])+'\n'  #need the actual temp for calculating the relatiave precission (results[StelPar][0]), above. Will use the relative temp (results[StelPar][1]) in plotting, below
				else:
					MeanPercisions[StelPar] = 100 - (np.mean(Uncertainties*.5)/results[StelPar])*100  # NOTE: this assumes that the retrieval precision is also gaussian. MIGHT NOT BE TRUE
					print_str += StelPar+' '*(max_str-len(StelPar))+' \t '+str(results[StelPar])+' \t\t '+str(MeanAccuracies[StelPar])+' \t '+str(MeanPercisions[StelPar])+'\n'
				# print_str += str(MeanPercisions[StelPar])+'\n'
				for disp in Params_2_display: #to store the precisions/accuracies of the results and their corresponding parameters
					if type(results[disp]) == list: #for non-primary photospheric temps, there are 2 numbers saved. The actual temp of the 2ndary photosphere and the temp relative to the primary temp
						Accuracies[StelPar][disp][0][cnt], Accuracies[StelPar][disp][1][cnt] = results[disp][1], MeanAccuracies[StelPar] #Need the relative temp ([1]) in plotting, here. Used the actual temp for calculating the mean ([0]), above
						Precisions[StelPar][disp][0][cnt], Precisions[StelPar][disp][1][cnt] = results[disp][1], MeanPercisions[StelPar]
					else:
						Accuracies[StelPar][disp][0][cnt], Accuracies[StelPar][disp][1][cnt] = results[disp], MeanAccuracies[StelPar]
						Precisions[StelPar][disp][0][cnt], Precisions[StelPar][disp][1][cnt] = results[disp], MeanPercisions[StelPar]
			print_str += '\n'
			cnt +=1
	if Print:
		print (print_str)

	""" To save all of the data for each triangle plot, subplot """
	ParmCombos_Acc, ParmCombos_Prec = {}, {}
	for StePr in StellarParams:
		ParmCombos_Acc[StePr], ParmCombos_Prec[StePr] = {}, {}
	MinAcc, MinPrec = 100, 100 # to record the min accuracy and precision for plotting purposes
	print_str = ""
	for pc in ParamCombos:
		pc_i = pc.split(' vs. ') #all of this data will be used multiple times, since we are marganalizing over only 2 (out of len(Params_2_display)) parameters
		for SP in StellarParams:
			x = list(set(Accuracies[SP][pc_i[0]][0])) #to get the unique values in the list of the 1st parameter
			y = list(set(Accuracies[SP][pc_i[1]][0])) #to get the unique values in the list of the 2nd parameter
			x.sort(), y.sort() #and sort the values in ascending order, which set() default does in decending order
			tot_elements = len(y)*len(x)
			ParmCombos_Acc[SP][pc], ParmCombos_Prec[SP][pc] = np.zeros((3,tot_elements)), np.zeros((3,tot_elements)) # the 3 dimensions are the 1st parm, the 2nd, and the corresponding mean accuracy. Same for precisions too
			it_cnt = 0
			for x_i in x:
				for y_i in y:
					index = np.where((Accuracies[SP][pc_i[0]][0] == x_i) & (Accuracies[SP][pc_i[1]][0] == y_i)) #only have to do this once because same ordering for Precsions/Accs/TA/TB/FA
					ParmCombos_Acc[SP][pc][0][it_cnt], ParmCombos_Prec[SP][pc][0][it_cnt] = x_i, x_i
					ParmCombos_Acc[SP][pc][1][it_cnt], ParmCombos_Prec[SP][pc][1][it_cnt] = y_i, y_i       
					ParmCombos_Acc[SP][pc][2][it_cnt], ParmCombos_Prec[SP][pc][2][it_cnt] = np.mean(Accuracies[SP][pc_i[0]][1][index]), np.mean(Precisions[SP][pc_i[0]][1][index]) # pc_i value doesn't actually matter, because AccuraciesXX[X][1] is always the same, no matter what X is
					if ParmCombos_Acc[SP][pc][2][it_cnt] < MinAcc:
						MinAcc = ParmCombos_Acc[SP][pc][2][it_cnt]
					if np.min(ParmCombos_Prec[SP][pc][2][it_cnt]) < MinPrec:
						MinPrec = ParmCombos_Prec[SP][pc][2][it_cnt]					
					it_cnt += 1
			print_str += str(SP)+": Accuracy['"+pc+"'] = "+str(ParmCombos_Acc[SP][pc])
			print_str += str(SP)+": Precisions['"+pc+"'] = "+str(ParmCombos_Prec[SP][pc])
		print_str += '\n'
	if Print:
		print (print_str)
	if MinScales:
		MinAcc, MinPrec = MinScales[0], MinScales[1]
	print ("ParamCombos:", ParamCombos)
	print ("Minimum accuracy:", MinAcc, "Minimum precisions:", MinPrec)
	WaveRanges = np.array(WaveRanges) #need the wavelength range to be an array for plotting purposes, later on
	def SaveTrianglePlt(FigName, SavePath): # since saving the precisions and accuries are nearly exactly the same, make a function that I can call twice. One for the precision and one for the accuracy
		if FigName.lower() == 'accuracy' or FigName.lower() == 'accuracies':
			DataStats, Param_combos, MinVal = Accuracies, ParmCombos_Acc, MinAcc
		if FigName.lower() == 'precision' or FigName.lower() == 'precisions':
			DataStats, Param_combos, MinVal = Precisions, ParmCombos_Prec, MinPrec
		fig = plt.figure(FigName)
		suptitle = ''
		for s in range(len(StellarParams)): #to quickly print a legend for the figures
			ParamName = StellarParams[s]
			if AxisLabeling and StellarParams[s] in list(AxisLabeling.keys()):
				ParamName = AxisLabeling[StellarParams[s]] # use the same naming schme that the axis are given
			suptitle += PlotShape[s] + ' = '+ ParamName+'\n'
		#acceptable size parameters: xx-small, x-small, small, medium, large, x-large, xx-large, larger, smaller, None
		# fig.suptitle(suptitle, fontsize="medium")#fontsize="x-large") #To make a global legend instead: https://stackoverflow.com/questions/9834452/how-do-i-make-a-single-legend-for-many-subplots-with-matplotlib
		shape = tot_parms-1 #there are # of total parameters-1 columns and rows in GTCs. The rows and columns are sysmetric
		fig.set_figheight(int(plot_scaling*shape)) #figure size set based on the number of columns and rows
		fig.set_figwidth(int(plot_scaling*shape)) #times 5 per each row/column in the height/width directions
		if SavePath != '' and SavePath[-1] != '/':
			SavePath+= '/'
		Fig_Cnt =0 # to keep track of how many of this specific figure has been made
		SavedFig = lambda fig_cnt, fig_type: SavePath+ExtraNaming[0]+fig_type+'_'+str(tot_parms)+'Params'+ExtraNaming[1]+'_GTCv'+str(fig_cnt)+'.png' #to get the string name of file to be saved
		saved_fig = SavedFig(Fig_Cnt, FigName)
		if not overwrite: #if don't want to overwrite the GTC figures, then add new version number to figure name
			while os.path.isfile(saved_fig):
				end = saved_fig.find('_GTC')+4
				saved_fig = saved_fig[:end]
				saved_fig+='v'+str(Fig_Cnt)+'.png'
				Fig_Cnt+=1
				saved_fig = SavedFig(Fig_Cnt, FigName)
		#if AxisLabeling was defined make a new list that maps the previous naming scheme of my parameters to the new one defined with 'AxisLabeling'
		if AxisLabeling: 
			Parms2Disp = []
			axisLabels = list(AxisLabeling.keys())
			for Par2dis in Params_2_display:
				if Par2dis in axisLabels:
					Parms2Disp.append(AxisLabeling[Par2dis])
				else:
					Parms2Disp.append(Par2dis)
		else: # otherwise keep the maping the same as the original naming scheme
			Parms2Disp = Params_2_display
		#here making the appropriate amount of subplots, in the correct order
		repeats1, repeats2 = Params_2_display[2:], Parms2Disp[2:] #The first 2 elements aren't going to be repeated in GTC
		VerticalTitles, VerticalNames = [Params_2_display[0]]+repeats1, [Parms2Disp[0]]+repeats2
		VerticalTitles, VerticalNames = VerticalTitles[::-1],  VerticalNames[::-1] #have to invert list, because printing label from top to bottom
		HorizTitles, HorizName = [Params_2_display[1]]+repeats1[::-1], [Parms2Disp[1]]+repeats2[::-1]
		PC_cnt = 1 #subplot figure indexing starts at 1 (very unpython like)
		row_cnt = 1 #to keep track of which row the loop is on
		row_extras = 0 #to keep track of the PC_cnt counts from the previous rows, so when doing PC_cnt > row_cnt, they both on the same scale
		New_Row = True #to keep track if on new row for y-axis plotting
		len_stelParm = len(StellarParams)
		steps = np.arange(0,len_stelParm)-np.floor(len_stelParm/2.0) #to have integer steps centered around 0
		for PC in range(shape**2):
			if PC_cnt > row_cnt+row_extras: #Each row should have it's row number as the number of used columns (starting from 1)
				pass
			else: #fill this corner plot
				plt.subplot(shape,shape,PC_cnt)
				PC_colm_cnt = (PC_cnt)-((row_cnt-1)*shape) #the column we are on is the count - all counts accumulated from previous row
				xlab_i = HorizTitles[PC_colm_cnt-1] #-1 because python and matplotlib.pyplot have diff index start (0 and 1)
				x_parm = list(set(DataStats[SP][xlab_i][0])) #just using the previous called SP value, because the specific stellar parameter doesn't matter, only the parameters being compared (2nd layer of dict)
				x_parm.sort() #to get the sorted unique values for this parameter #set() default sorts in decending order, want it in ascending order
				if xlab_i == 'Wave':
					for w in range(len(x_parm)): 
						x_parm[w] = int(x_parm[w]) #to convert the float values indicating which wavelength range we are working on to an int, than can be used to reference the wavelength range
					x_parm = list(WaveRanges[x_parm]) #need  to convert this back to a list for the plot labeling
				x_parm_ticks = len(x_parm) #and the corresponding count
				x_parm_ticks = np.linspace(0, 1, num=x_parm_ticks+2) #+2 because 0 and 1 are supposed to be edges of plot						
				if New_Row: #the y axis is the same for each row. When at a new row, label it. Otherwise keep the same labeling while on the same row
					ylab_i = VerticalTitles[row_cnt-1]
					plt.ylabel(VerticalNames[row_cnt-1], fontsize=FontSize, fontweight='bold') #row_cnt-1 because python and matplotlib.pyplot have diff index start (0 and 1)
					y_parm = list(set(DataStats[SP][ylab_i][0])) #same as for x_parm, only the 2nd layer of the dictonary matters
					y_parm.sort() #to get the sorted unique values for this parameter #set() default sorts in decending order, want it in ascending order
					if ylab_i == 'Wave':
						for w in range(len(y_parm)): 
							y_parm[w] = int(y_parm[w]) #to convert the float values indicating which wavelength range we are working on to an int, than can be used to reference the wavelength range
						y_parm = list(WaveRanges[y_parm]) #need to convert this back to a list for the plot labeling
					y_parm_ticks = len(y_parm) #and the corresponding count
					y_parm_ticks = np.linspace(0, 1, num=y_parm_ticks+2) #+2 because 0 and 1 are supposed to be edges of plot
					if ylab_i == 'Secondary_Temp':
						y_parm = list(np.array(y_parm)*100) #quick fix to convert 2ndary temperature from fraction relative to photospheric time to % realtive to phot temp. DON'T LIKE IT
					my_yticks = ['']+y_parm+[''] #[''] in front & back of list are just place holders, so my_ticks and param_ticks are the same length. They won't be shown in the figures
					plt.yticks(y_parm_ticks, my_yticks, fontsize=FontSize, fontweight='bold')
					New_Row = False
				else: #only have tick marks on the 1st column
					plt.yticks([])
				if row_cnt == shape: #At the last row. Since x axis is the same for each column, label all columns here
					plt.xlabel(HorizName[PC_colm_cnt-1], fontsize=FontSize, fontweight='bold') 
					if xlab_i == 'Secondary_Temp':
						x_parm =list(np.array(x_parm)*100) #quick fix to convert 2ndary temperature from fraction relative to photospheric time to % realtive to phot temp. DON'T LIKE IT
					my_xticks = ['']+x_parm+[''] #[''] in front & back of list are just place holders, so my_ticks and param_ticks are the same length. They won't be shown in the figures
					plt.xticks(x_parm_ticks, my_xticks, fontsize=FontSize, fontweight='bold')
				else: #only have ticks marks on the last row
					plt.xticks([])
				for ParmComb in ParamCombos:
					if ylab_i in ParmComb and xlab_i in ParmComb:
						params_i = ParmComb
				Params_i = params_i.split(' vs. ') 
				row = Params_i.index(ylab_i) # index of y component
				colm = Params_i.index(xlab_i) #index of x component
				for sp_i in range(len(StellarParams)): #loop to do every stellar parameter
					SP_i = StellarParams[sp_i]
					rawX_accs, rawY_accs = Param_combos[SP_i][params_i][colm], Param_combos[SP_i][params_i][row] #the x and y componenets of the scatter plot
					Z_accs = Param_combos[SP_i][params_i][2] # the z component of the scatter plot
					#since actual gridpoints go from 0.05-0.95 (contrary to axis labels), need to scale my data to that
					rawX_Uniques, rawY_Uniques = list(set(rawX_accs)), list(set(rawY_accs)) #get the unique occurances of the x and y values
					rawX_Uniques.sort(), rawY_Uniques.sort() #set() default sorts in decending order, want it in ascending order
					parameterized_x, parameterized_y = np.zeros(len(rawX_accs)), np.zeros(len(rawY_accs))
					scaleX, scaleY = x_parm_ticks[1:-1], y_parm_ticks[1:-1] #the 1st and last element isn't used in the plot. Should be same length of corresponding raw_unique!!!
					for r in range(len(rawX_Uniques)): #we've already sorted this
						convert_idx = np.where(rawX_accs == rawX_Uniques[r]) #so now go by and assigning each element a number based on the tick mark scaling
						parameterized_x[convert_idx] = np.ones(len(convert_idx))*scaleX[r]
					for r in range(len(rawY_Uniques)): #repeat process for x on y component
						convert_idx = np.where(rawY_accs == rawY_Uniques[r]) 
						parameterized_y[convert_idx] = np.ones(len(convert_idx))*scaleY[r]	
					plt.scatter(parameterized_x,parameterized_y+(steps[sp_i]*plotStep),c=Z_accs,  marker=PlotShape[sp_i], cmap='RdYlGn', vmin=MinVal, vmax=100,edgecolors='k')
					for xyz in range(len(parameterized_x)):
						plt.text(parameterized_x[xyz]+.01,parameterized_y[xyz]+(steps[sp_i]*plotStep),np.round(Z_accs[xyz],1))
				plt.ylim([0.05,.95]) #only plotting from 0.05 to .95
				plt.xlim([0.05,.95])
				if PC_cnt == row_cnt+row_extras: #only show colorbar on the last used column of each row
					cbar = plt.colorbar()
					cbar.set_label(FigName, rotation=270, fontsize=FontSize, fontweight='bold')
			if PC_cnt%shape == 0:
				row_cnt +=1
				New_Row = True
				row_extras = PC_cnt
			PC_cnt += 1	
		plt.subplots_adjust(wspace=0.03, hspace=0.03) #to reduce the spacing between each subplot
		# fig.savefig(saved_fig) #have to save the figures, otherwise would have to look at each figure 1 at a time as the code runs
		print ("saved histograms as '"+saved_fig+"'")
		plt.show()
		plt.close()
		return None
	FigNames = ['Accuracy', 'Precision'] #to keep track  of if this is the Accuracy or Precision figure
	for fiG in FigNames:
		SaveTrianglePlt(fiG, SavePath)
	return None

def PlankSpecIrrad_vtil(T=290): #T in Kelvin, v in Hz # Just plotting a simple black body function
	h = 6.6260701e-34 # SI, J*s
	k = 1.38064852e-23 #SI, J/K
	c = 299792458 #m/s
	vtil = np.linspace(0.1, 2.0e3, num = 2000)
	vtil = vtil*100.0 # to convert from cm-1 to m-1
	expC = (h*vtil*c)/(k*T)
	B_v = (100*c**2*2*h*vtil**3)/(np.exp(expC)-1)
	plt.plot(vtil, B_v)
	plt.plot(vtil, B_v*.8)
	plt.xticks([])
	plt.yticks([])
	plt.show()
	plt.close()
	return None

######===== Run Functions =====######
if __name__ == '__main__':
	Dat = ['Res500_Wav2900-5700Ang/Dynesty_Results_stats_call1.pkl', 'Res500_Wav2000-8000Ang/Dynesty_Results_stats.pkl', 'Res60000_Wav2900-5700Ang/Dynesty_Results_stats.pkl']
	# trueVals = [5900, .93, 4000, .07] #HiRes test
	trueVals = [2990, 0.53885, 4211] #LowRes test
	names = ['Temp1', 'frac1', 'Temp2']
	# quantiles = [15.865,50,84.135] #upper and lower bound of 1sig and mean value
	quantiles = [15.865,50,84.135] #upper and lower bound of 2sig and mean value
	DisplayParamets = ['met0.0_grav4.5_tempA8500_tempB8075_frac0.02_snr400']
	# Histograms('Res500_Wav2500-17100A/ChiSqrd_Results_stats.pkl', subdir='Dyn_InterpResults')#, display_paramets=DisplayParamets)
	All_parms = ['Res', 'SNR', 'Wave', 'Primary_Temp', 'Primary_SurfaceFraction', 'Secondary_Temp'] #to help me keep track of all of the parmaters
	Axis_Labeling={"Res": 'Resolution (R)', "Secondary_Temp": 'Spot Contrast \n(% of photosphere temperature)', "Primary_Temp":'Photospheric Temperature (K)', "Wave": 'Wavelength range ($\AA$)', "Primary_SurfaceFraction": '1 - covering_fraction'}
	# TrianglularPlots(Dat, AxisLabeling=Axis_Labeling, Print=False)
	# TrianglularPlots(Dat, plot_scaling=5, MinScales = [66,48], skip_params=['Res', 'SNR', 'Primary_SurfaceFraction', 'Secondary_Temp'], AxisLabeling=Axis_Labeling, Print=False, FontSize=13) #['2900-5700']
	# PosteriorResults("InterpResults/Dynesty_M0.0_G4.5_Ta8500_Tb6800_F0.1_S400run150.pkl", quantiles, SaveCorn='Ta=8484_F=1-0.08474_Tb=6797', Names=['Photospheric Temp', '1 - spot fraction', 'Spot Temp'], TrueVals=[8484, 1-0.08474, 6797])
	Synth_dat = ['InterpResults/SYNTHSPECmet0.0_grav4.5_tempA8484_tempB6797_frac0.08474_snr403.0run150.fits', 'InterpResults/WAVE_PHOENIX-ACES-AGSS-COND-2011-Res500.fits']
	# Data_vs_Fit(Synth_dat, [8490.415606883249, 0.9071162868821602, 6908.04466832706], 'PHOENIX_R500_from2000-17000A_wavstep4.0')
