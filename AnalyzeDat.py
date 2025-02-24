#!/usr/bin/env python
#
#    This file is part of iSpec.
#    Copyright Sergi Blanco-Cuaresma - http://www.blancocuaresma.com/s/
#
#    iSpec is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    iSpec is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with iSpec. If not, see <http://www.gnu.org/licenses/>.
#
import os
import sys
import numpy as np
import logging
import pickle
import multiprocessing
from multiprocessing import Pool
import matplotlib.pyplot as plt
import glob
from astropy.time import Time

from uncertainties import ufloat #used for error propergation
from uncertainties import unumpy #uses linear error propagation theory
from uncertainties import umath #documentation: https://pythonhosted.org/uncertainties/_downloads/uncertaintiesPythonPackage.pdf
################################################################################
#--- iSpec directory -------------------------------------------------------------
ispec_dir = '/Users/chimamcgruder/Research/iSpec_v20201001'
sys.path.insert(0, ispec_dir)
import ispec

#--- Change LOG level ----------------------------------------------------------
#LOG_LEVEL = "warning"
LOG_LEVEL = "info"
logger = logging.getLogger() # root logger, common for all
logger.setLevel(logging.getLevelName(LOG_LEVEL.upper()))
################################################################################
report_progress = lambda current_work_progress, last_reported_progress: (int(current_work_progress) % 10 == 0 and current_work_progress - last_reported_progress > 10) or last_reported_progress < 0 or current_work_progress == 100 #True every 10% of progress

def FindSigFigs(nominal, sigma, sifdig=2): # to find the significant digits in a number series. Base the significant digits on the uncertainty. The 1st 'sifdig' non-0s are the significant digits
    str_cnt, sig_cnt, str_sigm = -1, 0, str(sigma)
    whrDec = str_sigm.index('.') 
    whlNums = whrDec #this is also the length of whole numbers
    if whlNums == 1 and str_sigm[0] == '0': #except when there is a there are no whole numbers
        whlNums = whrDec-1 #then will be one off
    if whrDec > sifdig: #allow one more significant figure if the uncertainties is deep in the whole numbers
        sifdig +=1
    start_sig = False #to keep track of when to start counting significant figures, i.e. after the 1st non-0
    while sig_cnt < sifdig:
        str_cnt+=1 #wanna add the cnt 1st, so can get the proper index of that number when out of string
        if str_cnt >= len(str_sigm): #if surpassed the size of the string, before finding all the significant figures
            sig_cnt += 1 #break out, but add the extra last number for the sig figs count
            break 
        s = str_sigm[str_cnt]
        if s == '.':
            pass
        elif s == '0' and not start_sig:
            pass
        else:
            start_sig = True #as soon as you see 1 non-0 start the sig count
            sig_cnt += 1
    if sig_cnt <= whlNums: #if the significant figures are all within the whole numbers
        decimal_rnd = str_cnt+1-whrDec #then count backwards to round the proper number of significant digits (i.e. minus decimals)
        return int(round(nominal, decimal_rnd)), int(round(sigma, decimal_rnd))#, decimal_rnd
    else: #otherwise the significant digits are in the decimals,
        decimal_rnd = str_cnt-whrDec #and count how many digits needed after the decimal place
        return round(nominal, decimal_rnd), round(sigma, decimal_rnd)#, decimal_rnd

def degrade_resolution(dat_file, from_res=115000, to_res=65000, Save=False, File_Base=None, ErrLim=1e-8, LOG=True):
    #File_Bas=starting string of reduced res file to be saved. This should include the path and begining of file name (e.g. 'HD23249/older_data/ADP.2014-09-18T12_13_19.393_RVcorr').
    if type(dat_file) == str: #if data given is string. then assuming it's a string of the file name. Will read out the actual spectrum
        star_spectrum = ispec.read_spectrum(dat_file)
        if np.min(star_spectrum['waveobs']) > 1000: #Since assuming working with near UV to near IR data. Then a min wavelength greater than 1k must be in Ang, convert to nm
            star_spectrum['waveobs'] = star_spectrum['waveobs']/10
    if not File_Base: #if didn't specify the base name, use the same basename that the file has
        File_Base = dat_file
    else: #if not string assumming its the extracted iSpec data
        star_spectrum = dat_file

    OGstar_spectrum = np.copy(star_spectrum)
    MinFlux = np.min(star_spectrum['flux']) #if min flux is less than 0, add the min flux to all elements. The contiuumm fitting messes up when there's negative flux.   
    if MinFlux < 0:
        star_spectrum['flux'] = star_spectrum['flux']+abs(MinFlux) #since normalizing, don't need to shift back    
    #--- Resolution degradation ----------------------------------------------------
    if LOG:
        logging.info("Resolution degradation...")
    convolved_star_spectrum = ispec.convolve_spectrum(star_spectrum, to_res, from_resolution=from_res)
    if MinFlux < 0:
        convolved_star_spectrum['flux'] = convolved_star_spectrum['flux']-abs(MinFlux) #To shift back after oberations are done
    if Save:
        #To save reduced resolution data in same path as the original data
        path = ''
        split_path, dataFile = File_Base.split('/')[:-1], File_Base.split('/')[-1] #to get the path info and the base file name
        for s in split_path:
            path +=s+'/'

        if type(dat_file) != str and not File_Base: #if file given isn't a file name and the name of the reduced resoultion isn't provided complain
            sys.exit("Type of file is "+str(type(dat_file))+" and 'File_Base' is not defined. Need file to be string or the 'File_Base' to save files!")
        elif type(dat_file) == str: # if the file passes is a string. Save the reduced res with the same file name, but with the reduced resolution saved
            if dat_file[-5:] == '.fits': #spectral file is .fits file
                dataFile, Type0 = dataFile[:dataFile.find('.fits')], '.fits' #also remove path of file
            if dat_file[-4:] == '.txt': #spectral file is .txt file
                dataFile, Type0 = dataFile[:dataFile.find('.txt')], '.txt'
            FileName = path+dataFile+'Res'+str(to_res)+'.txt'
        else: #if a specific file name is given as 'File_Base'
            FileName = path+dataFile+'Res'+str(to_res)+'.txt' #assuming that in the case that 'File_Base' is given, File_Base.split('/')[-1] == the info of the start of the file to be saved (like asked for in variable definition)
        f = open(FileName, 'w')
        f.write('# waveobs   flux   err\n')
        if np.nanmean(convolved_star_spectrum['err']) < ErrLim: #If errors are greater than this, errors are actually given. Otherwise set errors equal to 0
            convolved_star_spectrum['err'] = np.zeros(len(convolved_star_spectrum['err']))
        for w in range(len(convolved_star_spectrum['waveobs'])):
            f.write(str(convolved_star_spectrum['waveobs'][w])+'   '+str(convolved_star_spectrum['flux'][w])+'   '+str(convolved_star_spectrum['err'][w])+'\n') 
        f.close()
        if LOG:
            logging.info("Saved spectra degraded from a resolution of "+str(from_res)+" to "+str(to_res)+" as '"+FileName+"'...")
    return convolved_star_spectrum, OGstar_spectrum

def create_segments_around_linemasks(Lines, Marg =0):
    #Marg = extra cushion
    #---Create segments around linemasks -------------------------------------------
    line_regions = ispec.read_line_regions(Lines)
    segments = ispec.create_segments_around_lines(line_regions, margin=Marg)
    return segments


def find_continuum_regions(dat_file, Model="Splines", IgnRegion=None, Res=115000, Deg=2, MedianWavRng=0.05, MaxWavRag=1.0, Nsplines=None, Normalize=True, PLOT=False, UseErr=False, plot_save_path=None):
    #NOTE: The continuum messes up if there are cosmic rays near the bluer, low throughput, low counts end of the spectrum. Will have to manually remove cosmic rays in those rare cases
    #TODO!! highlight line regions and ignore it in contiuum fitting. think right?
    #model = "Splines" or "Polynomy". Could also fit continuum with a teplate or fixed value, but those require seperate code
    #Nsplines = number of splines. If None, will automatically set 1 spline every 5 nm
    #UseErr == if you want to weight the spectrum with its errors. I find that this causes the continuum fit to be more uneven
    #plot_save_path = path where to save the figure. Can also add a prefix to the figure name as path+'/'<prefix>+'_'
    if type(dat_file) == str: #if data given is string. then assuming it's a string of the file name. Will read out the actual spectrum
        star_spectrum = ispec.read_spectrum(dat_file)
        if np.min(star_spectrum['waveobs']) > 1000: #Since assuming working with near UV to near IR data. Then a min wavelength greater than 1k must be in Ang, convert to nm
            star_spectrum['waveobs'] = star_spectrum['waveobs']/10
        if PLOT and not plot_save_path: #to get the path name of figures to be saved from the origianl files
            plot_save_path = ''
            splitfile = dat_file.split('/')
            for sp in range(len(splitfile)-1):
                plot_save_path +=  splitfile[sp]+'/'
    else: #if not string assumming its the extracted iSpec data
        star_spectrum = dat_file
        if PLOT and not plot_save_path:
            sys.exit("No path name to save figures specified!!!!")
    OGspectrum = np.copy(star_spectrum) #need to make a copy because ispec's corrections are still linked to the original file 
    MinFlux = np.min(star_spectrum['flux']) #if min flux is less than 0, add the min flux to all elements. The contiuumm fitting messes up when there's negative flux.   
    if MinFlux < 0:
        star_spectrum['flux'] = star_spectrum['flux']+abs(MinFlux) #since normalizing, don't need to shift back  
    #--- Continuum fit -------------------------------------------------------------

    # Strategy: Filter first median values and secondly MAXIMUMs in order to find the continuum
    order='median+max'
    if IgnRegion:
        Ign = ispec.read_segment_regions(ispec_dir+'/input/regions/'+IgnRegion)
    else:
        Ign = None
    continuum_model = ispec.fit_continuum(star_spectrum, from_resolution=Res, nknots=Nsplines, degree=Deg,ignore =Ign,\
                                median_wave_range=MedianWavRng, max_wave_range=MaxWavRag, model=Model, order=order,\
                                automatic_strong_line_detection=True, strong_line_probability=0.5,use_errors_for_fitting=UseErr)
    if PLOT:    
        plt.figure('continuum_model')
        if Ign is not None:
            for I in Ign:
               plt.axvspan(I[0], I[1], facecolor='r', alpha=.5)
        plt.plot(star_spectrum['waveobs'], star_spectrum['flux'], 'b')
        plt.fill_between(star_spectrum['waveobs'], star_spectrum['flux']-star_spectrum['err'], star_spectrum['flux']+star_spectrum['err'], alpha=0.5)
        plt.plot(star_spectrum['waveobs'], continuum_model(star_spectrum['waveobs']), 'g')
        plt.savefig(plot_save_path+'continuum_model.png')

    #--- Continuum normalization ---------------------------------------------------
    if Normalize:
        logging.info("\033[1mContinuum normalization...\033[0m")
        normalized_spectrum = ispec.normalize_spectrum(star_spectrum, continuum_model, consider_continuum_errors=False)
        if PLOT:
            # Use a fixed value because the spectrum is already normalized
            norm_continuum_model = ispec.fit_continuum(star_spectrum, fixed_value=1.0, model="Fixed value")
            plt.figure('normalized_spectrum')
            plt.plot(normalized_spectrum['waveobs'], norm_continuum_model(normalized_spectrum['waveobs']), 'g')
            plt.plot(normalized_spectrum['waveobs'], normalized_spectrum['flux'], 'b')
            plt.savefig(plot_save_path+'normalized_spectrum.png')
        return normalized_spectrum, continuum_model, OGspectrum
    else:
        return continuum_model, OGspectrum

def PlotSpec(Spectra, filename=None, title=1, ylabel='Flux', xlabel='Wavelength (nm)', PltStyle=None, alphas=None, FontSizes=None, Labels=None, ylim=None, xlim =None, Interactive=False):
    """
        Plot a spectrum or array of spectra.
        If filename (i.e. plot.png) is specified, then the plot is saved into that file
        and not shown on the screen.
        Spectra = list of ispec.read_spectrum files
        IF plotting multiple spectra (i.e. calling this function multiple times) and want to save the figure. Specify filename on last call
    """
    if type(Spectra) != list:
        sys.exit("Variable type passed as 'Spectra' is "+str(type(Spectra))+'. Need it to be a list of the different iSpec spectra you want to plot')
    if not FontSizes:
        FontSizes = 10
    if not Labels:
        LabCnts, Labels = range(len(Spectra)), []
        for l in LabCnts:
            Labels.append('Spectra'+str(1+l))
    if Interactive: # then save interactive plot as .pkl
        fig,ax = plt.subplots()
    else:
        figure = plt.figure(title)
    plt.title(title, fontsize=FontSizes)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=FontSizes)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=FontSizes)


    cnt, clr, styl = 0,0,0
    if not PltStyle:
        colors = ['b', 'g', 'r', 'c', 'm', 'k', 'y']
        linestyles = ['-', '--', ':', '.', '^', '.-', 'o']
    if alphas is not None:
        pass
    else:
        alphas = np.ones(len(Spectra))
    for spec in Spectra:
        if type(spec) == str:
            str_spec = spec
            spec = ispec.read_spectrum(spec)
            if np.min(spec['waveobs']) > 1000: #Since assuming working with near UV to near IR data. Then a min wavelength greater than 1k must be in Ang, convert to nm
                spec['waveobs'] =  spec['waveobs']/10.0
        if not PltStyle:
            Style = colors[clr]+linestyles[styl]
            if clr%(len(colors)-1) == 0 and clr > 0: #reset clr cnt and incriment style cnt 
                styl, clr = styl+1, 0 #This will lead to a total of 49 deffernt spectra plotting options
            else:
                clr+=1
        else:
            Style = PltStyle[cnt]
        plt.plot(spec['waveobs'], spec['flux'], Style, label=Labels[cnt], alpha=alphas[cnt])
        cnt +=1
    plt.legend()
    if ylim:
        plt.ylim((ylim[0],ylim[1]))
    if xlim:
        plt.xlim((xlim[0],xlim[1]))
    if filename: #giving a file name means that you want the figure to be saved
        if not Interactive:
            if filename[-4:] == ".png": #if ending in .png assuming that file path and full file name
                pass
            else:
                if filename[-1] != '/':
                    filename +='/' #to makes sure we have the back slash between file and folder names
                filename = filename+title+".png"
            plt.savefig(filename) #otherwise, use the figure title as the file name
            print ("Saved static plot as '"+filename+"'")
        else:
            if filename[-4:] == ".png": #not using the .png extension when saving as interactive
                filename = filename[:-4]
            else:
                if filename[-1] != '/':
                    filename +='/' #to makes sure we have the back slash between file and folder names
                filename = filename+title+'fig.pkl'
            pickle.dump(fig, open(filename, 'wb'))
            print ("Saved interactive plot as '"+filename+"'")
    return None

def GetDateTime(FILES, DataType='HARPS', extras=None): #To pull out the data and time from the data file name. Can only be used for ESO archive data
    #extras = to remove any excess naming of base file name in  README.txt
    #DataType= string of insturment where the data came from i.e. 'HARPS', 'CORALIE', 'FEROS', etc.
    #FILES = list of file names (strs)
    ###================For HARPS data================###
    if DataType.lower() == 'harps' or 'feros':
        #To get path info containing data and readme
        Dirs, PATH = FILES[0].split('/'), '' #assuming data is all located in same folder
        for d in range(len(Dirs)-1): #last element in Dirs is the file name
            PATH += Dirs[d]+'/'
        #To read info in README.txt file
        readme = glob.glob(PATH+'README*')
        if len(readme) != 1:
            readme = glob.glob(PATH+'readme*') #after the ESO archive update, the readme files are title with lower case
            if len(readme) != 1:
                sys.exit("Number of README.txt (or readme.txt) files in '"+PATH+"' is "+str(len(readme))+". There needs to be one and one only!")
        read = open(readme[0], 'r')        
        FitsNames, FitsStartTimes, FitsStartDates = [], [], []
        for line in read:
            if 'SCIENCE.SPECTRUM' in line:
                numb = line.split()
                if len(numb) != 5:
                    sys.exit('Length of split line is '+str(len(numb))+' Expecting 5!')
                FitsNames.append(numb[1].replace(":", "_")), FitsStartDates.append(numb[2][6:16]), FitsStartTimes.append(numb[2][17:29]) #Assuming Date string is of same length everytime. Think that's the case
        #To get base of file and order the file names with the dates. Have to remove <.type> from file. Assuming can only be .fits or .txt file
        ReOrderTimes, ReOrderDates, intFile, Order = [], [], FILES[0], []
        if intFile[-5:] == '.fits': #spectral file is .fits file
            base, Type0 = intFile[len(PATH):intFile.find('.fits')], '.fits' #also remove path of file
        if intFile[-4:] == '.txt': #spectral file is .txt file
            base, Type0 = intFile[len(PATH):intFile.find('.txt')], '.txt'
        if extras is not None:
            base = base[:base.find(extras)]
        Order.append(FitsNames.index(base+'.fits')) #assuming that all data was initially saved as .fits file in REAMME.txt
        for f in range(len(FILES)-1):
            file = FILES[f+1]
            if file.find('.txt') == -1: #spectral file is .fits file
                base, Type = file[len(PATH):file.find('.fits')], '.fits' #also remove path of file
            if file.find('.fits') == -1: #spectral file is .txt file
                base, Type = file[len(PATH):file.find('.txt')], '.txt'
            if extras is not None:
                base = base[:base.find(extras)]
            if Type != Type0:
                sys.exit("File type of current file: '"+Type+".' Previous type of current file: '"+Type0+".' ALL files need to be the same type!!" )
            Order.append(FitsNames.index(base+'.fits')) #assuming that all data was initially saved as .fits file in REAMME.txt
            Type0 = Type 
        for o in Order:
            ReOrderDates.append(FitsStartDates[o]), ReOrderTimes.append(FitsStartTimes[o])
        return ReOrderDates, ReOrderTimes #spits date and time out in order of FILES is given

    ###================================================###

def calculate_barycentric_velocity(File, Date, Time, RA, Dec, Correct=True, log=False):
    #Date='yyyy-mm-dd', Time='hh:mm:sss.sss', RA = [RAhours, RAmins, RAsec], Dec = [Decdeg, Decmin, Decsec]
    if type(File) == str:
        mu_cas_spectrum = ispec.read_spectrum(File)
        if np.min(mu_cas_spectrum['waveobs']) > 1000: #Since assuming working with near UV to near IR data. Then a min wavelength greater than 1k must be in Ang, convert to nm
            mu_cas_spectrum['waveobs'] = mu_cas_spectrum['waveobs']/10
    else:
        mu_cas_spectrum = File
    spectrum_uncorrected = np.copy(mu_cas_spectrum) #need to make a copy because ispec's corrections are still linked to the original file
    #--- Barycentric velocity correction from observation date/coordinates ---------
    if log:
        logging.info("\033[1mCalculating barycentric velocity correction...\033[0m")

    # Project velocity toward star
    date = Date.split('-')
    time = Time.split(':')
    barycentric_vel = ispec.calculate_barycentric_velocity_correction((date[0], date[1], date[2], time[0], time[1], \
                                    time[2]), (RA[0], RA[1], RA[2], Dec[0], Dec[1], Dec[2]))
    if log:
        logging.info("\033[1mFound barycentric velocity: "+str(barycentric_vel)+" [km/s]\033[0m")
    #--- Correcting barycentric velocity -------------------------------------------
    if Correct:
        if log:
            logging.info("\033[1mRadial velocity correction...\033[0m")
        corrected_spectrum = ispec.correct_velocity(mu_cas_spectrum, barycentric_vel)
        # print ("spectrum_uncorrected['waveobs']", spectrum_uncorrected['waveobs'])
        # print ("corrected_spectrum['waveobs']", corrected_spectrum['waveobs'])
        # PlotSpec([spectrum_uncorrected, corrected_spectrum], PltStyle = ['b-', 'r--'], Labels = ['Uncorrected', 'BaryCen Corrected'])
        return barycentric_vel, corrected_spectrum
    return barycentric_vel

def determine_radial_velocity_with_mask(dat_file, mask_file, Correct=True, LwrVlim=-200, UpVlim=200,Vstp=1.0, Msk_dpth=0.01, Fourier=False, PLOT=False, log=False, fileName=None, Sigs=2, Interact=False):
    #Sigs= sololy for plotting significant figures of the results
    #fileName = the path of where the figure will be save, could also be the full path and file name
    #Interact = if PLOT is True and fileName is not None, then if you want to save the figure as an interactive figure or not
    if type(dat_file) == str:
        mu_cas_spectrum = ispec.read_spectrum(dat_file)
        if np.min(mu_cas_spectrum['waveobs']) > 1000: #Since assuming working with near UV to near IR data. Then a min wavelength greater than 1k must be in Ang, convert to nm
            mu_cas_spectrum['waveobs'] = mu_cas_spectrum['waveobs']/10
        if PLOT and not fileName: #to get the path name of figures to be saved from the origianl files
            fileName = ''
            splitfile = dat_file.split('/')
            for sp in range(len(splitfile)-1):
                fileName +=  splitfile[sp]+'/'
    else: #if not string assumming its the extracted iSpec data
        mu_cas_spectrum = dat_file
        if PLOT and not fileName:
            sys.exit("No path name to save figures specified!!!!")        
    spectrum_uncorrected = np.copy(mu_cas_spectrum) #need to make a copy because ispec's corrections are still linked to the original file   
    #--- Radial Velocity determination with linelist mask --------------------------
    if log:
        logging.info("\033[1mRadial velocity determination with linelist mask...\033[0m")
    # - Read atomic data
    # mask_file = ispec_dir + "input/linelists/CCF/HARPS_SOPHIE.G2.375_679nm/mask.lst"
    ccf_mask = ispec.read_cross_correlation_mask(ispec_dir+'/'+mask_file)

    models, ccf = ispec.cross_correlate_with_mask(mu_cas_spectrum, ccf_mask, lower_velocity_limit=LwrVlim, \
                            upper_velocity_limit=UpVlim, velocity_step=Vstp, mask_depth=Msk_dpth, fourier=Fourier)

    # Number of models represent the number of components
    components = len(models)
    # First component:
    if components < 1:
        print ("broken datafile!")
        if fileName:
            print( "\033[1m'"+fileName+"'\033[0m")
        print (type(models))
        print (models)
        PlotSpec([mu_cas_spectrum])
        plt.show()
        plt.close()
        sys.exit()
    rv, rv_err = models[0].mu(), models[0].emu()# km/s
    rv_rnd, rv_err_rnd = FindSigFigs(rv, rv_err, sifdig=Sigs)
    if log:
        logging.info("\033[1mFound radial velocity: "+str(rv_rnd)+"+/-"+str(rv_err_rnd)+" [km/s]\033[0m")
    if Correct:
        if log:
            logging.info("\033[1mRadial velocity correction...\033[0m")
        corrected_spectrum = ispec.correct_velocity(mu_cas_spectrum, rv)
        if PLOT:
            PlotSpec([spectrum_uncorrected, corrected_spectrum], PltStyle=['b-', 'r--'], Labels=['Uncorrected', 'RadV Corrected'], filename=fileName, title="RVcorrectedvsUncorrected_Spectrum", Interactive=Interact)
        return rv, rv_err, corrected_spectrum
    return rv, rv_err 

def Estimate_SNR(File, NumPoints=10, Est_Err=False, ErrLim=1e-8, WavRange=None): #WavRange=[500,680]
    #Est_Err= if you want to produce and return the spectrum with errors estimated from the snr
    #ErrLim = if the mean errors are lower than this, we assume the error values are invalid and will calculate it on our own
    #NumPoints = How large the block are for estimating the snr. Found that 10 give the most consistent SNR to the ESO archive (used the 3 H29 HARPS spec)
    if type(File) == str:
        star_spectrum = ispec.read_spectrum(File)
        if np.min(star_spectrum['waveobs']) > 1000: #Since assuming working with near UV to near IR data. Then a min wavelength greater than 1k must be in Ang, convert to nm
            star_spectrum['waveobs'] = star_spectrum['waveobs']/10
    else:
        star_spectrum = File
    NaNs = np.argwhere(np.isnan(star_spectrum['err']))[:,0] #Assuming star_spectrum['err'] is a 1D array. Pretty safe assumption
    star_spectrum['err'][NaNs] = np.zeros(len(NaNs))
    
    #To screen counts that are less than 0. When count is less than 0, the snr estimation messes up and can't properly determine Ca H and K lines
    Zeros = len(np.where(star_spectrum['flux'] == 0)[0]) # Exactly 0 counts is a gap in data. Want to save the number of datapoints that are gaps
    NegScreen = np.where(star_spectrum['flux'] > 0)[0]
    CleanedSpectrum = np.copy(star_spectrum)
    CleanedSpectrum = CleanedSpectrum[NegScreen] 
    og_len = len(star_spectrum['flux'])
    screened = og_len-Zeros-len(NegScreen)
    if screened > 0: #if had to screen more than the gaps, print update on what was screened
        print("had to screen "+str(screened)+"("+str(screened/(og_len-Zeros))+"%) negative datapoints") 

    #--- Estimate SNR within a given wavelength range. In this case, the errors aren't estimated -------------------#
    if WavRange is not None and not Est_Err: #Only use the specified wavelength range to calculate the spectrum SNR. When Est_Err == True, want to use the whole spectra to save error estimates
        Range = np.where((CleanedSpectrum['waveobs']>WavRange[0]) & (CleanedSpectrum['waveobs']<WavRange[1]))[0]
        cut_spectrum = CleanedSpectrum[Range] #Only analyze data within given range
        if np.nanmean(cut_spectrum['err']) > ErrLim: #if there are actually errors given in the original data, use that to calculate error
            logging.info("\033[1mEstimating SNR from errors...\033[0m")
            efilter = np.where(cut_spectrum['err'] > 0)[0] #don't use regions where the error is negative (unphysical error)
            efiltered_star_spectrum = cut_spectrum[efilter]
            estimated_snr = np.mean(efiltered_star_spectrum['flux']/efiltered_star_spectrum['err']) 
        else:
            CleanedSpectrum= CleanedSpectrum[Range] #Only analyze data within given range
            estimated_snr = ispec.estimate_snr(CleanedSpectrum['flux'], num_points=NumPoints)
        logging.info("\033[1mEstimated SNR = "+str(estimated_snr)+"\033[0m, from wavelength range:"+str(WavRange[0])+"-"+str(WavRange[1])+"\n")
        return estimated_snr
    
    #--- Estimate SNR from errors -------------------(Only do so if there are errors provided and it's not just 0s or NaNs)-------------------#
    if np.nanmean(CleanedSpectrum['err']) > ErrLim: #if there are actually errors given in the original data
        efilter = np.where(CleanedSpectrum['err'] > 0)[0] #don't use regions where the error is negative (unphysical error)
        efiltered_star_spectrum = CleanedSpectrum[efilter]
        estimated_snr = np.mean(efiltered_star_spectrum['flux']/efiltered_star_spectrum['err'])
        logging.info("\033[1mEstimated SNR = "+str(estimated_snr)+"\033[0m, from errors.")
        return estimated_snr # don't need to calculate errs, if already given
    
    #--- Estimate SNR from flux -----------------------------------------------------#
    ##!!! WARNING: To compare SNR estimation between different spectra, they should be homogeneously sampled (consider a uniform re-sampling) !!!##  
    elif not Est_Err: #if don't want to save the noise based off the estimated SNR, then just run ispec's routine on the data to get global estimate of SNR
        estimated_snr = ispec.estimate_snr(CleanedSpectrum['flux'], num_points=NumPoints)
        logging.info("\033[1mEstimated SNR = "+str(estimated_snr)+"\033[0m, from fluxes.")
        return estimated_snr
    else:
        #COMPARE WITH FULL SPECTRA SNR ESTIMATES (look under the hood of ispec.estimate_snr, to see best way to compare)!!!!
        num_splits = int(np.ceil(len(CleanedSpectrum['flux'])/(NumPoints*2))) #Split the spectra into N/(NumPoints*2), where N is the number of datapoints. Need (NumPoints*2) because ispec.estimate_snr needs at least 2 wavelength blocks of size NumPoints to work
        FinalSpectrum = np.copy(CleanedSpectrum)
        Wav, Flux, Errs = CleanedSpectrum['waveobs'], CleanedSpectrum['flux'], CleanedSpectrum['err'] #use the shifted spectra to estimate snr
        strt, estimated_SNRs, frame, last_reported_progress, weights = 0, [], None, -1, []#-1 because starting at 0
        for split in range(num_splits): #Then get an estimate of the SNR for that localized region, rinse and repeat until done the full spectrum.
            start_wave = Wav[strt] #for calculating weights the starting wavelength should always be the previous end wavelength (i.e. don't include the overlapping wavelengths for the last block)
            if split == num_splits-1: #if at last index
                strt, end = len(Wav)-(2*NumPoints), len(Wav) #make sure at least have NumPoints of spectra datapoints to estimate SNR, it's okay if the last few datapoints overlap
            else:
                end = (2*NumPoints)+strt
            end_wave = Wav[end-1] #-1 because not actually including last index when do [strt:end]
            estimated_snr = ispec.estimate_snr(Flux[strt:end], num_points=NumPoints, updates=False) #updates == False, because don't want update for every single split
            FinalSpectrum['err'][strt:end] = abs(FinalSpectrum['flux'][strt:end]/estimated_snr) #use the unshifted spectra to calculate final error
            
            #To give updates on how the overall error estimation is going
            current_work_progress = ((split*1.0/ num_splits) * 100.0)
            if report_progress(current_work_progress, last_reported_progress):
                last_reported_progress = current_work_progress
                logging.info("%.2f%%" % current_work_progress)
                if frame is not None:
                    frame.update_progress(current_work_progress)
            # estimated_SNRs.append(estimated_snr)
            # weights.append(end_wave-start_wave) #use the wavelengths as weights, because even though it's using the same number of datapoints, the wavelength range might not be consistent
            strt = end
        estimated_snr = np.mean(abs(FinalSpectrum['flux'])/FinalSpectrum['err']) #just estimate global snr from the snr of each datapoint, where the snr of each datapoint was determined from the 'split' process
        # estimated_snr = np.average(np.array(estimated_SNRs), weights=np.array(weights)) #have opposite weighting scheme than normal. Give values with larger wavelength ranges to have a larger weight
        logging.info("SNR = %.2f" % estimated_snr)
        # print ("\nestimating error using whole spectrum... ")
        # estimated_snr = ispec.estimate_snr(FinalSpectrum['flux'], num_points=NumPoints)
        return estimated_snr, FinalSpectrum

def Add_Noise(File, SNR, distribution = "gaussian"):
    """
    Add noise to a spectrum (ideally to a synthetic one) based on a given SNR.
    """
    #distribution options: "poisson" or "gaussian"
    #NOTE: something is wrong with the poisson distribution. lambda = sig2 = sqrt(N) = 1/snr
    if type(File) == str:
        star_spectrum = ispec.read_spectrum(File)
    else:
        star_spectrum =File
    OGspectrum = np.copy(star_spectrum)
    if distribution.lower() == "gaussian":
        noisy_star_spectrum = ispec.add_noise(star_spectrum, SNR, distribution)
    #TODO: The poisson function doesn't make sense!!!! :? NEED TO FIX!!!
    #Myabe just can't work right with super high counts???
    if distribution.lower() == "poisson":
        noisy_star_spectrum = np.copy(star_spectrum)
        lambd = SNR**2 #with shot noise the sigma = snr = sqrt(lambda)
        noise = np.random.poisson(lambd*np.ones(len(star_spectrum['flux']))) #make array of noise based off poisson distribution
        JustNoise = noise-lambd*np.ones(len(star_spectrum['flux'])) #in poisson distribution the mean = lambda. We want to just capture the noise effect, therefore subtract mean from noise
        noisy_star_spectrum['flux'] = noisy_star_spectrum['flux']+JustNoise #add noise component
        noisy_star_spectrum['err'] = np.sqrt(lambd) #lambda = variance = sigma^2 = snr^2 = std^2
    return noisy_star_spectrum, OGspectrum



def CombineSpectra(Spectras, WavRng=None, Step=0.001, PLOT=False, Normalize=True, ZerEdg=True, Method='sum', FolderTargName=None, Base='_RVcorr', plot_resampled=True, ErrEstPoint=10, plot_save_path=None):
    #FolderTargName = list containing [Folder_name, Target_name]. If given assumed you want to save file. Otherwise not saving
    #Method = 'sum' or 'average'
    #To average all spectra of a given dataset. Provide standard deviation amongst combined spectra, which will be used as error bars
    #first resample spectra so on the same wavelength grid
    #ZerEdg = If False, the first and last value in the edge will be propagated if the xaxis is bigger than spectrum['waveobs']. If True, then zero values will be assigned instead.
    #Base = additional string naming, right before the .txt name #ErrEstPoint=number of points in grid to estimate error. If None then not estimating errors
    #To read in the data
    wav_diffs = np.array([]) #if Step is not defined, keep tract of the total difference of wavelength steps
    len_Spec = len(Spectras)
    for S in range(len_Spec):
        if type(Spectras[S]) == str: #if Spectras are strings, then assuming it's a string of the file name. Will read out the actual spectrum
            Spec = ispec.read_spectrum(Spectras[S])
            if np.min(Spec['waveobs']) > 1000: #Since assuming working with near UV to near IR data. Then a min wavelength greater than 1k must be in Ang, convert to nm
                Spec['waveobs'] = Spec['waveobs']/10
            if PLOT and not plot_save_path: #to get the path name of figures to be saved from the origianl files
                plot_save_path = ''
                splitfile = Spectras[S].split('/')
                for sp in range(len(splitfile)-1):
                    plot_save_path +=  splitfile[sp]+'/'                
            Spectras[S] = Spec #save read out spectra so Spectras is now only extracted spectra
        else:
            pass #otherwise, passed data is already extracted
            if PLOT and not plot_save_path:
                sys.exit("No path name to save figures specified!!!!")
        if not Step:
            wav_diffs = np.append(wav_diffs, np.diff(Spectras[S]['waveobs']))
    if not Step:
        Step = np.mean(wav_diffs)

    #To to determine the min and max wavelength of the resampled spectra
    if WavRng: #already defined parameters of combined spectra
        pass
    else: #otherwise, use the global max and min wavelength as the range and the global mean step size as the step
        WavMin, WavMax = 1e12, -1e8
        for s in Spectras:
            min_s, max_s = np.min(Spec['waveobs']), np.max(Spec['waveobs'])
            if min_s < WavMin:
                WavMin = min_s
            if max_s > WavMax:
                WavMax = max_s
        WavRng = [WavMin, WavMax]
    print ("resampled spectra wavelength step:", np.round(Step,6), "from", str(np.round(WavRng[0],2))+'-'+str(np.round(WavRng[1],2))) 
    wavelengths = np.arange(WavRng[0], WavRng[1], Step)
    resampled_spectra, cnts = np.zeros((len(wavelengths), len_Spec)), 0
    for Spect in Spectras:
        print ("resampling spectra "+str(cnts+1)+'/'+str(len_Spec))
        spec = np.copy(Spect)
        min_spec = abs(np.min(spec['flux']))
        spec['flux']=spec['flux']+min_spec #need to add offset to ensure no negatives get truncated, preserving structure of spectrum
        resampled_spectrum = ispec.resample_spectrum(spec, wavelengths, zero_edges=ZerEdg)
        resampled_spectrum['flux'] = resampled_spectrum['flux']-min_spec #subtract previously added offset
        resampled_spectra[:,cnts] = resampled_spectrum['flux']
        cnts += 1
    if Method.lower() == 'sum' or Method.lower() == 'add':
        spec_med = np.nansum(resampled_spectra, axis=1)
    if Method.lower() == 'mean' or Method.lower() == 'average':
        spec_med = np.nanmean(resampled_spectra, axis=1)
    CombinedSpec = resampled_spectrum #use the last resampled_spectrum file just to get the same formate for the combined spectra.
    CombinedSpec['flux'], CombinedSpec['err'] = spec_med, np.zeros(len(spec_med)) #The 'waveobs' grid shouldn't change, and the errors should be set to 0. Erros can be estimated below
    if ErrEstPoint: #This is the best way to properly estimate error. Using the std grossly underestimates the snr
        estimated_snr, CombinedSpec = Estimate_SNR(CombinedSpec, NumPoints=ErrEstPoint, Est_Err=True)
    factor = 1
    if Normalize: #if normalize, do it to the combined spectra. This is just a simple normalization (i.e. divide by a constant term). NOT continumm normalization
        factor = np.nanmedian(spec_med[np.where(spec_med!=0)])
        CombinedSpec['flux'] = spec_med/factor
    if PLOT:
        if plot_resampled: #if plotting all of the uncombined spectra, need to make this an interactive plot
            if Method.lower() == 'sum' or Method.lower() == 'add':
                if factor != 1:
                    factor = factor/len(resampled_spectra[0,:]) #if summing all spectra, the normalization term should also be reduced
            fig,ax = plt.subplots()
            for i in range(len(resampled_spectra[0,:])):
                plt.plot(CombinedSpec['waveobs'], resampled_spectra[:,i]/factor, '--',alpha=0.6)
        else:
            plt.figure("combined_spectra")
        plt.plot(CombinedSpec['waveobs'], CombinedSpec['flux'], 'r-', label='mean spectra')
        if ErrEstPoint:
            plt.fill_between(CombinedSpec['waveobs'], CombinedSpec['flux']-(CombinedSpec['err']/factor), CombinedSpec['flux']+(CombinedSpec['err']/factor), color='red', alpha=0.3, label='STD')
        plt.legend()
        if plot_resampled:
            pickle.dump(fig, open(plot_save_path+'combined_spectra.fig.pkl', 'wb'))
            print ("Saved interactive plot as '"+plot_save_path+'combined_spectra.fig.pkl'+"'")
        else:
            plt.savefig(plot_save_path+'combined_spectra.png')
            print ("Saved plot as '"+plot_save_path+'combined_spectra.png'+"'")
    if FolderTargName:
        SaveWavnFlux(FolderTargName[0]+'/'+FolderTargName[1]+'_'+Method+Base+'.txt', CombinedSpec['waveobs'], CombinedSpec['flux'], Errors=CombinedSpec['err']/factor)
    return CombinedSpec, resampled_spectra

def filter_cosmic_rays(Spec_Files, PLOT = False, ErrThreshlds = [5,2], CosWidth = .08, MeanWidth = 5, Save=None, IgnoreRegion = []):
    #TODO!!! cosmic ray removal not very reliable. Ask Munnazza about better cosmic ray removal
    #NEEDS TO BE RV CORRECTED BEFOREHAND!!!!
    #IgnoreRegion = Wave length regions to ignore because they are emission lines. list of list where each element =[front_wave_reg, end_wave_reg] 
    #Spec_File _ can be either string name of file to be exported or the already exported iSpec np array
    #ErrThreshld = the amount a spectra std should be multiplied by to flag potential cosmic rays 
    #Save = base file name of the cosmic ray corrected data. If Save is None, won't save data
    if Save:
        if len(Spec_Files) != len(Save):
            sys.exit("len(Spec_Files)="+str(len(Spec_Files))+' but len(Save)='+len(Save)+'. They need to be the same length!!')
    wavelengths, spec_med, spec_std, resampled_spectra, StarSpecs = AverageSpectra(Spec_Files, WavRng=None, PLOT=False)
    STD, cnt, NoCosSpectra = np.mean(spec_std), 0, [] #mean std of data from one another
    for s in range(len(resampled_spectra[0,:])):
        CorrSpec, star_spec,ResampSpec = np.copy(StarSpecs[s]), StarSpecs[s], resampled_spectra[:,s]
        normalized_spectrum, Continuum, star_spec = find_continuum_regions(star_spec, Model="Polynomy", Deg=8, PLOT =False)
        std, wavs, Ispecs = np.std(normalized_spectrum['flux']), normalized_spectrum['waveobs'], normalized_spectrum['flux'] #here std is std of each datapoint from the overall spectra
        for IgRg in IgnoreRegion: #This effectively ignores the highlighted emission regions by setting em equal to the continuum
            EmiLins = np.where((IgRg[0]<wavs) & (wavs<IgRg[1]))[0]
            Ispecs[EmiLins] = np.ones(len(EmiLins))      
        cosmics = np.where(normalized_spectrum['flux'] > 1+(std*ErrThreshlds[1]))[0]
        Diff = np.mean(np.diff(wavs)) + (np.mean(np.diff(wavs))*.25) #add extra 25% cause of numerical errors
        spec_i, wav_i, Spacing = normalized_spectrum['flux'][cosmics], normalized_spectrum['waveobs'][cosmics], np.diff(normalized_spectrum['waveobs'][cosmics])#cause we pulling spectra from 'AverageSpectra', we know the spacin is the same
        idx_CosCuts, i = np.where(Spacing>Diff)[0], 0 #there4 when cosmics spacin is greater than wave grid spacin, it's a different cos ray 
        CosPeakIndecies, CosPeaks, = [], [] #AllIndecies, AllPeaks = [], [], [], []
        PeakWavs, PeakFlux = [], []
        if len(idx_CosCuts) > 0: # if don't find any outlier counts skip
            for c in idx_CosCuts: #To find peak value of each cosmic ray
                c += 1
                if i == 0: #1st cosmic ray
                    miniPeakIdx = np.argmax(spec_i[:c]) #find peak index relative to cosmic ray data subset
                    FullPeakIdx = np.where(wavs == wav_i[:c][miniPeakIdx])[0][0] #find peak index relative whole dataset
                    PeakWidx = find_nearest(wavelengths, wavs[FullPeakIdx]) #to find wavelength location of peak in terms of the resampleed wavelength space
                    if ResampSpec[PeakWidx] > (STD*ErrThreshlds[0])+spec_med[PeakWidx]: #another filter to make sure flag object is a cosmic ray and not emission lines
                        # print ('\033[1m wavelengths[PeakWidx]:', wavelengths[PeakWidx], 'ResampSpec[PeakWidx]',ResampSpec[PeakWidx], '(STD*ErrThreshlds[0])+spec_med[PeakWidx]', (STD*ErrThreshlds[0])+spec_med[PeakWidx], '\033[0m')
                        #Remove comsic ray and replace with local mean 
                        #Use regions found by normalized spectra to correct the original data
                        Bck_Indeces = np.where((wavs[FullPeakIdx]-(MeanWidth/2.0)<wavs) & (wavs<wavs[FullPeakIdx]-(CosWidth/2.0)))[0]
                        Fnt_Indeces = np.where((wavs[FullPeakIdx]+(MeanWidth/2.0)>wavs) & (wavs>wavs[FullPeakIdx]+(CosWidth/2.0)))[0]
                        i_CosIndxs = np.where((wavs[FullPeakIdx]-(CosWidth/2.0)<wavs) & (wavs<wavs[FullPeakIdx]+(CosWidth/2.0)))[0]
                        CorrSpec['flux'][i_CosIndxs] = np.ones(len(i_CosIndxs))*np.mean(np.append(CorrSpec['flux'][Bck_Indeces],CorrSpec['flux'][Fnt_Indeces]))
                        PeakWavs.append(normalized_spectrum['waveobs'][FullPeakIdx]), PeakFlux.append(normalized_spectrum['flux'][FullPeakIdx])
                    else:
                        # print ('wavelengths[PeakWidx]:', wavelengths[PeakWidx], 'ResampSpec[PeakWidx]',ResampSpec[PeakWidx], '(STD*ErrThreshlds[0])+spec_med[PeakWidx]', (STD*ErrThreshlds[0])+spec_med[PeakWidx])
                        pass
                else: #all other cosmic rays
                    miniPeakIdx = np.argmax(spec_i[preC:c]) #find peak index relative to cosmic ray data subset
                    FullPeakIdx = np.where(wavs == wav_i[preC:c][miniPeakIdx])[0][0]#find peak index relative whole dataset
                    PeakWidx = find_nearest(wavelengths, wavs[FullPeakIdx]) #to find wavelength location of peak in terms of the resampleed wavelength space
                    if ResampSpec[PeakWidx] > (STD*ErrThreshlds[0])+spec_med[PeakWidx]: #another filter to make sure flag object is a cosmic ray and not emission lines    
                        # print ('wavelengths[PeakWidx]:', wavelengths[PeakWidx], 'ResampSpec[PeakWidx]',ResampSpec[PeakWidx], '(STD*ErrThreshlds[0])+spec_med[PeakWidx]', (STD*ErrThreshlds[0])+spec_med[PeakWidx])
                        #Remove comsic ray and replace with local mean
                        #Use regions found by normalized spectra to correct the original data
                        Bck_Indeces = np.where((wavs[FullPeakIdx]-(MeanWidth/2.0)<wavs) & (wavs<wavs[FullPeakIdx]-(CosWidth/2.0)))[0]
                        Fnt_Indeces = np.where((wavs[FullPeakIdx]+(MeanWidth/2.0)>wavs) & (wavs>wavs[FullPeakIdx]+(CosWidth/2.0)))[0]
                        i_CosIndxs = np.where((wavs[FullPeakIdx]-(CosWidth/2.0)<wavs) & (wavs<wavs[FullPeakIdx]+(CosWidth/2.0)))[0]
                        CorrSpec['flux'][i_CosIndxs] = np.ones(len(i_CosIndxs))*np.mean(np.append(CorrSpec['flux'][Bck_Indeces],CorrSpec['flux'][Fnt_Indeces]))
                        PeakWavs.append(normalized_spectrum['waveobs'][FullPeakIdx]), PeakFlux.append(normalized_spectrum['flux'][FullPeakIdx])   
                    else:
                        # print ('\033[1m wavelengths[PeakWidx]:', wavelengths[PeakWidx], 'ResampSpec[PeakWidx]',ResampSpec[PeakWidx], '(STD*ErrThreshlds[0])+spec_med[PeakWidx]', (STD*ErrThreshlds[0])+spec_med[PeakWidx], '\033[0m')
                        pass
                preC, i = c, i+1
            miniPeakIdx = np.argmax(spec_i[preC:]) #for last cosmic ray #find peak index relative to cosmic ray data subset
            FullPeakIdx = np.where(wavs == wav_i[preC:][miniPeakIdx])[0][0] #find peak index relative whole dataset
            PeakWidx = find_nearest(wavelengths, wavs[FullPeakIdx]) #to find wavelength location of peak in terms of the resampleed wavelength space
            if ResampSpec[PeakWidx] > (STD*ErrThreshlds[0])+spec_med[PeakWidx]: #another filter to make sure flag object is a cosmic ray and not emission lines
                # print ('wavelengths[PeakWidx]:', wavelengths[PeakWidx], 'ResampSpec[PeakWidx]',ResampSpec[PeakWidx], '(STD*ErrThreshlds[0])+spec_med[PeakWidx]', (STD*ErrThreshlds[0])+spec_med[PeakWidx])
                #Remove comsic ray and replace with local mean
                #Use regions found by normalized spectra to correct the original data
                Bck_Indeces = np.where((wavs[FullPeakIdx]-(MeanWidth/2.0)<wavs) & (wavs<wavs[FullPeakIdx]-(CosWidth/2.0)))[0]
                Fnt_Indeces = np.where((wavs[FullPeakIdx]+(MeanWidth/2.0)>wavs) & (wavs>wavs[FullPeakIdx]+(CosWidth/2.0)))[0]
                i_CosIndxs = np.where((wavs[FullPeakIdx]-(CosWidth/2.0)<wavs) & (wavs<wavs[FullPeakIdx]+(CosWidth/2.0)))[0]
                CorrSpec['flux'][i_CosIndxs] = np.ones(len(i_CosIndxs))*np.mean(np.append(CorrSpec['flux'][Bck_Indeces],CorrSpec['flux'][Fnt_Indeces]))   
                PeakWavs.append(normalized_spectrum['waveobs'][FullPeakIdx]), PeakFlux.append(normalized_spectrum['flux'][FullPeakIdx])
            else:
                # print ('\033[1m wavelengths[PeakWidx]:', wavelengths[PeakWidx], 'ResampSpec[PeakWidx]',ResampSpec[PeakWidx], '(STD*ErrThreshlds[0])+spec_med[PeakWidx]', (STD*ErrThreshlds[0])+spec_med[PeakWidx], '\033[0m')
                pass

        if PLOT:
            plt.figure('Normalized_'+str(cnt))
            plt.plot(normalized_spectrum['waveobs'], normalized_spectrum['flux'])
            plt.plot(normalized_spectrum['waveobs'][cosmics], normalized_spectrum['flux'][cosmics], 'r.', markersize = 8)
            SecondFilter = np.where(normalized_spectrum['flux'] > (STD*ErrThreshlds[0])+spec_med[PeakWidx])[0]
            plt.plot(normalized_spectrum['waveobs'][SecondFilter], normalized_spectrum['flux'][SecondFilter], 'm.', markersize = 5)
            plt.plot(normalized_spectrum['waveobs'], np.ones(len(normalized_spectrum['waveobs']))+(std*ErrThreshlds[1]), 'k--')
            plt.plot(PeakWavs, PeakFlux, 'g^', markersize = 1)
            plt.figure('Spectra_'+str(cnt))
            plt.plot(star_spec['waveobs'], star_spec['flux'], alpha=.5, label='OGspectrum'+str(cnt+1))
            plt.plot(CorrSpec['waveobs'], CorrSpec['flux'], label='CosmicRMspectrum')
            plt.legend()
        #     plt.show()
        #     plt.close()
        # sys.exit()
        # Input = raw_input("Do you want to save the cosmic cleaned spectra [yes(y)/no(n)]? > ")
        # if Input.lower() == 'y' or Input.lower() == 'yes':
        if Save is not None:
            f = open(Save[cnt]+'_RMcosmics.txt', 'w')
            f.write('# waveobs   flux   err\n')
            for w in range(len(CorrSpec['waveobs'])):
                f.write(str(CorrSpec['waveobs'][w])+'   '+str(CorrSpec['flux'][w])+'   '+str(CorrSpec['err'][w])+'\n') 
            f.close()
            logging.info('\033[1mSaved cosmic ray cleaned spectra as "'+Save[s]+'_RMcosmics.txt\033[0m"')
        cnt +=1
        NoCosSpectra.append(CorrSpec)
    logging.info('\033[1mRemoved identified cosmic rays. Exiting...\033[0m')
    return NoCosSpectra 

#!!!!!! Think make obsolet by SaveWavnFlux()!!!!!!
def SaveFitsAsTxt(Files): #To save .fits data as .txt files cause it's easier to read and manipulate
    for File in Files:
        spectrum = ispec.read_spectrum(File)
        if np.min(spectrum['waveobs']) > 1000: #Since assuming working with near UV to near IR data. Then a min wavelength greater than 1k must be in Ang, convert to nm
            spectrum['waveobs'] = spectrum['waveobs']/10.0
        # MinFlux = np.min(spectrum['flux']) #if min flux is less than 0, add the min flux to all elements. The contiuumm fitting messes up when there's negative flux.   
        # if MinFlux < 0:  # Don't want to offset by min because that messes up the inherent SNR.
        #     spectrum['flux'] = spectrum['flux']+abs(MinFlux)
        #To get base of file, have to remove <.type> from file. Assuming can only be .fits or .txt file
        if File.find('.txt') == -1: #spectral file is .fits file
            base = File[:File.find('.fits')]
        if File.find('.fits') == -1: #spectral file is .txt file
            base = File[:File.find('.txt')]
        f = open(base+'.txt', 'w')
        f.write('# waveobs   flux   err\n')
        NaNs = np.where(np.isnan(spectrum['err']))
        spectrum['err'][NaNs] = np.zeros(len(NaNs)) #To replace all NaNs with 0s cause code iSpec can't handle that
        for w in range(len(spectrum['waveobs'])):
            f.write(str(spectrum['waveobs'][w])+'   '+str(spectrum['flux'][w])+'   '+str(spectrum['err'][w])+'\n') 
        f.close()
        print ('Saved edited spectra as "'+base+'.txt'+'"')
    return None

#The rotines that are used for Equivalent width method. The portion that is the same for the parameter and abundance estimations
def EW_func(ContinuumModels, RV, Resolution = 115000, code="moog", AtomicLineList = "GESv5_atom_hfs_iso.420_920nm/atomic_lines.tsv",\
    use_ares=False, ModelAtmo='MARCS.GES/', SolarAbundances='Grevesse.2007/stdatom.dat'):
    #Needs to be RV corrected and continuum normailzed already!!!
    #AtomicLineList options: "VALD.300_1100nm/atomic_lines.tsv", "VALD.1100_2400nm/atomic_lines.tsv", "GESv5_atom_hfs_iso.420_920nm/atomic_lines.tsv", "GESv5_atom_nohfs_noiso.420_920nm/atomic_lines.tsv"
    #ModelAtmo options: "MARCS.GES/", "MARCS.APOGEE/", "ATLAS9.APOGEE/", "ATLAS9.Castelli/", "/ATLAS9.Kurucz/", "ATLAS9.Kirby/"
    #ContinuumModels = list of two spectra: normalized spectra (0) and origianl, unnormalized spectra (1)
    #SolarAbundances options: "Grevesse.2007/stdatom.dat", "Asplund.2005/stdatom.dat", "Asplund.2009/stdatom.dat", "Grevesse.1998/stdatom.dat", "Anders.1989/stdatom.dat"
    #RV = [rv,rv_err], determined by 'determine_radial_velocity_with_mask()'
    #RVcorrected = True/False. If RV correction already don't correct again. Still need there RV info for telluric velocities
    ###----------------------------------------------------###----------------------------------------------------###----------------------------------------------------###


    #--- Telluric velocity shift determination from spectrum --------------------------
    #telluric_linelist are use to ignore wavelength regions that might be affected by the tellurics
    logging.info("Telluric velocity shift = -RV correction = %.2f +/- %.2f" % (-RV[0], RV[1]))
    vel_telluric, vel_telluric_err = -RV[0], RV[1] #telluric V is the negative of the radial velocity correction
    telluric_linelist_file = ispec_dir + "/input/linelists/CCF/Synth.Tellurics.500_1100nm/mask.lst"
    telluric_linelist = ispec.read_telluric_linelist(telluric_linelist_file, minimum_depth=0.0)
    # models, ccf = ispec.cross_correlate_with_mask(star_spectrum, telluric_linelist, \
    #                         lower_velocity_limit=-100, upper_velocity_limit=100, \
    #                         velocity_step=0.5, mask_depth=0.01, \
    #                         fourier = False,
    #                         only_one_peak = True)
    # vel_telluric = np.round(models[0].mu(), 2) # km/s
    # vel_telluric_err = np.round(models[0].emu(), 2) # km/s

    #--- Resolution degradation ---------------(to convert to resolution of solar spectrum models)-------------
    # NOTE: The line selection was built based on a solar spectrum with R ~ 47,000 and GES/VALD atomic linelist.
    normalized_star_spectrum, star_spectrum = ContinuumModels[0], ContinuumModels[1]
    # Use a fixed value because the spectrum is already normalized
    star_continuum_model = ispec.fit_continuum(normalized_star_spectrum, fixed_value=1.0, model="Fixed value")
    # from_resolution, to_resolution = Resolution, Resolution
    # normalized_star_spectrum = ispec.convolve_spectrum(normalized_star_spectrum, to_resolution, from_resolution) #Think I still need to do this if not changing res, so have same spacing of all spec????

    #--- Fit lines -----------------------------------------------------------------
    if AtomicLineList is None: #if no AtomicLineList provided, don't use it in the line fitting, but extra fitted atomic lines from previously made models
        #--- Read lines and adjust them ------------------------------------------------
        if code in ['width', 'moog']:
            line_regions_with_atomic_data = ispec.read_line_regions(ispec_dir + "/input/regions/47000_GES/{}_ew_ispec_good_for_params_all_extended.txt".format(code))
            #line_regions_with_atomic_data = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/{}_ew_ispec_good_for_params_all_extended.txt".format(code))
        else:
            line_regions_with_atomic_data = ispec.read_line_regions(ispec_dir + "/input/regions/47000_GES/{}_synth_good_for_params_all_extended.txt".format(code))
            #line_regions_with_atomic_data = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/{}_synth_good_for_params_all_extended.txt".format(code))

        # Select only iron lines
        line_regions_with_atomic_data = line_regions_with_atomic_data[np.logical_or(line_regions_with_atomic_data['note'] == "Fe 1", line_regions_with_atomic_data['note'] == "Fe 2")]

        smoothed_star_spectrum = ispec.convolve_spectrum(star_spectrum, 2*to_resolution)
        line_regions_with_atomic_data = ispec.adjust_linemasks(smoothed_star_spectrum, line_regions_with_atomic_data, max_margin=0.5)

        #--- Fit the lines but do NOT cross-match with any atomic linelist since they already have that information
        linemasks = ispec.fit_lines(line_regions_with_atomic_data, normalized_star_spectrum, star_continuum_model,\
                                    atomic_linelist = None, max_atomic_wave_diff = 0.005, telluric_linelist = telluric_linelist,\
                                    check_derivatives = False, vel_telluric = vel_telluric, discard_gaussian=False,\
                                    smoothed_spectrum=None, discard_voigt=True, free_mu=True, crossmatch_with_mu=False, closest_match=False)
    else:
        #--- Fit lines -----------------------------------------------------------------
        logging.info("Fitting lines...") #Using cross match atomic lines provided with the spectra to find atomic lines
        atomic_linelist_file = ispec_dir + "/input/linelists/transitions/"+AtomicLineList
        atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file, wave_base=np.min(normalized_star_spectrum['waveobs']), wave_top=np.max(normalized_star_spectrum['waveobs']))

        if code in ['width', 'moog']:
            line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_GES/{}_ew_ispec_good_for_params_all.txt".format(code))
            #line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/{}_ew_ispec_good_for_params_all.txt".format(code))
        else:
            line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_GES/{}_synth_good_for_params_all.txt".format(code))
            #line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/{}_synth_good_for_params_all.txt".format(code))

        line_regions = ispec.adjust_linemasks(normalized_star_spectrum, line_regions, max_margin=0.5)
        #To use gaussian/voigt profile to fit spectra against the marked line_regions. 
        #If atomic_linelist is provied, line will be cross-matched with atomic info (more accurate cause derived from data rather than model?)
        linemasks = ispec.fit_lines(line_regions, normalized_star_spectrum, star_continuum_model,\
                                    atomic_linelist = atomic_linelist, max_atomic_wave_diff = 0.00,\
                                    telluric_linelist = telluric_linelist, smoothed_spectrum = None,\
                                    check_derivatives = False, vel_telluric = vel_telluric, discard_gaussian=False,\
                                    discard_voigt=True, free_mu=True, crossmatch_with_mu=False, closest_match=False)
        # Discard lines that are not cross matched with the same original element stored in the note
        linemasks = linemasks[linemasks['element'] == line_regions['note']]

        # Exclude lines that have not been successfully cross matched with the atomic data
        # because we cannot calculate the chemical abundance (it will crash the corresponding routines)
        rejected_by_atomic_line_not_found = (linemasks['wave_nm'] == 0)
        linemasks = linemasks[~rejected_by_atomic_line_not_found]


    # Discard bad masks
    flux_peak = normalized_star_spectrum['flux'][linemasks['peak']]
    flux_base = normalized_star_spectrum['flux'][linemasks['base']]
    flux_top = normalized_star_spectrum['flux'][linemasks['top']]
    bad_mask = np.logical_or(linemasks['wave_peak'] <= linemasks['wave_base'], linemasks['wave_peak'] >= linemasks['wave_top'])
    bad_mask = np.logical_or(bad_mask, flux_peak >= flux_base)
    bad_mask = np.logical_or(bad_mask, flux_peak >= flux_top)
    linemasks = linemasks[~bad_mask]

    # Exclude lines with EW equal to zero
    rejected_by_zero_ew = (linemasks['ew'] == 0)
    linemasks = linemasks[~rejected_by_zero_ew]

    # Exclude lines that may be affected by tellurics
    rejected_by_telluric_line = (linemasks['telluric_wave_peak'] != 0)
    linemasks = linemasks[~rejected_by_telluric_line]

    if use_ares:
        # Replace the measured equivalent widths by the ones computed by ARES
        old_linemasks = linemasks.copy()
        ### Different rejection parameters (check ARES papers):
        ##   - http://adsabs.harvard.edu/abs/2007A%26A...469..783S
        ##   - http://adsabs.harvard.edu/abs/2015A%26A...577A..67S
        #linemasks = ispec.update_ew_with_ares(normalized_star_spectrum, linemasks, rejt="0.995", tmp_dir=None, verbose=0)
        #linemasks = ispec.update_ew_with_ares(normalized_star_spectrum, linemasks, rejt="3;5764,5766,6047,6052,6068,6076", tmp_dir=None, verbose=0)
        snr = 50
        linemasks = ispec.update_ew_with_ares(normalized_star_spectrum, linemasks, rejt="%s" % (snr), tmp_dir=None, verbose=0)

    # Selected model amtosphere and solar abundances
    #Load model atmospheres
    model = ispec_dir+"/input/atmospheres/"+ModelAtmo
    modeled_layers_pack = ispec.load_modeled_layers_pack(model)

    # Load SPECTRUM abundances
    solar_abundances_file = ispec_dir+"/input/abundances/"+SolarAbundances
    solar_abundances = ispec.read_solar_abundances(solar_abundances_file)

    return modeled_layers_pack, linemasks, normalized_star_spectrum, solar_abundances

def Parameters_from_EW(ContinuumModels, initial_teff, initial_logg, initial_MH, RV=None, initial_alpha=None, Resolution=115000, max_iterations=40, SaveDat=["Folder", "<Target>", "<JDdate>"], \
    code="width", AtomicLineList = "GESv5_atom_hfs_iso.420_920nm/atomic_lines.tsv", use_ares=False, ModelAtmo='MARCS.GES/', SolarAbundances='Grevesse.2007/stdatom.dat', free_params=["teff", "logg", "vmic", "vsini"]): #metallicity is always fit for! 
    #ContinuumModels = list of two spectra: normalized spectra (0) and origianl, unnormalized spectra (1)
    #AtomicLineList options: "VALD.300_1100nm/atomic_lines.tsv", "VALD.1100_2400nm/atomic_lines.tsv", "GESv5_atom_hfs_iso.420_920nm/atomic_lines.tsv", "GESv5_atom_nohfs_noiso.420_920nm/atomic_lines.tsv"
    #ModelAtmo options: "MARCS.GES/", "MARCS.APOGEE/", "ATLAS9.APOGEE/", "ATLAS9.Castelli/", "/ATLAS9.Kurucz/", "ATLAS9.Kirby/"
    #SolarAbundances options: "Grevesse.2007/stdatom.dat", "Asplund.2005/stdatom.dat", "Asplund.2009/stdatom.dat", "Grevesse.1998/stdatom.dat", "Anders.1989/stdatom.dat"
    #RV = [rv,rv_err], determined by 'determine_radial_velocity_with_mask()'
    #RVcorrected = True/False. If RV correction already don't correct again. Still need there RV info for telluric velocities
    ###----------------------------------------------------###----------------------------------------------------###----------------------------------------------------###

    #--- Model spectra from EW --------------------------------------------------
    initial_vmic = ispec.estimate_vmic(initial_teff, initial_logg, initial_MH)
    if not RV: #if no RV is given, then assuming it's already corrected i.e. RV = 0
        RV = [0,0]
    modeled_layers_pack, linemasks, normalized_star_spectrum, solar_abundances =  EW_func(ContinuumModels, RV, Resolution=Resolution,  
        code=code, AtomicLineList=AtomicLineList, use_ares=use_ares, ModelAtmo=ModelAtmo, SolarAbundances=SolarAbundances)

    if not initial_alpha:
        initial_alpha = ispec.determine_abundance_enchancements(initial_MH)
    # Validate parameters
    if not ispec.valid_atmosphere_target(modeled_layers_pack, {'teff':initial_teff, 'logg':initial_logg, 'MH':initial_MH, 'alpha':initial_alpha}):
        msg = "The specified effective temperature, gravity (log g) and metallicity [M/H] fall out of theatmospheric models."
        print (msg)


    # Reduced equivalent width
    # Filter too weak/strong lines
    # * Criteria presented in paper of GALA
    #efilter = np.logical_and(linemasks['ewr'] >= -5.8, linemasks['ewr'] <= -4.65)
    efilter = np.logical_and(linemasks['ewr'] >= -6.0, linemasks['ewr'] <= -4.3)
    # Filter high excitation potential lines
    # * Criteria from Eric J. Bubar "Equivalent Width Abundance Analysis In Moog"
    efilter = np.logical_and(efilter, linemasks['lower_state_eV'] <= 5.0)
    efilter = np.logical_and(efilter, linemasks['lower_state_eV'] >= 0.5)
    ## Filter also bad fits
    efilter = np.logical_and(efilter, linemasks['rms'] < 1.00)
    # no flux
    noflux = normalized_star_spectrum['flux'][linemasks['peak']] < 1.0e-10
    efilter = np.logical_and(efilter, np.logical_not(noflux))
    unfitted = linemasks['fwhm'] == 0
    efilter = np.logical_and(efilter, np.logical_not(unfitted))

    results = ispec.model_spectrum_from_ew(linemasks[efilter], modeled_layers_pack, \
                        solar_abundances, initial_teff, initial_logg, initial_MH, initial_alpha, initial_vmic, \
                        free_params=["teff", "logg", "vmic", "vsini"], adjust_model_metalicity=True, \
                        max_iterations=max_iterations, enhance_abundances=True, \
                        #outliers_detection = "robust", \
                        #outliers_weight_limit = 0.90, \
                        outliers_detection = "sigma_clipping", \
                        #sigma_level = 3, \
                        tmp_dir=None, code=code)
    params, errors, status, x_over_h, selected_x_over_h, fitted_lines_params, used_linemasks = results

    ##--- Save results -------------------------------------------------------------
    if SaveDat is not None:
        dump_file = SaveDat[0]+'/'+SaveDat[1]+"_results_"+SaveDat[2]+".txt"
        SaveParamsAstxt(params, errors, status, dump_file)
    return params, errors, status

def Abundances_from_EW(ContinuumModel, RV, teff, logg, MH, alpha, microturbulence_vel, Resolution = 115000,code="moog", \
    AtomicLineList = "VALD.300_1100nm/atomic_lines.tsv", use_ares=False, ModelAtmo='MARCS.GES/', SolarAbundances='Grevesse.2007/stdatom.dat'):
    #AtomicLineList options: "VALD.300_1100nm/atomic_lines.tsv", "VALD.1100_2400nm/atomic_lines.tsv", "GESv5_atom_hfs_iso.420_920nm/atomic_lines.tsv", "GESv5_atom_nohfs_noiso.420_920nm/atomic_lines.tsv"
    #ModelAtmo options: "MARCS.GES/", "MARCS.APOGEE/", "ATLAS9.APOGEE/", "ATLAS9.Castelli/", "/ATLAS9.Kurucz/", "ATLAS9.Kirby/"
    #SolarAbundances options: "Grevesse.2007/stdatom.dat", "Asplund.2005/stdatom.dat", "Asplund.2009/stdatom.dat", "Grevesse.1998/stdatom.dat", "Anders.1989/stdatom.dat"
    #RV = [rv,rv_err], determined by 'determine_radial_velocity_with_mask()'
    #RVcorrected = True/False. If RV correction already don't correct again. Still need there RV info for telluric velocities
    ###----------------------------------------------------###----------------------------------------------------###----------------------------------------------------###

    #--- Determining abundances by EW of the previously fitted lines ---------------

    modeled_layers_pack, linemasks, normalized_star_spectrum, solar_abundances =  EW_func(ContinuumModel, RV, Resolution = Resolution,  
        code=code, AtomicLineList = AtomicLineList, use_ares=use_ares, ModelAtmo=ModelAtmo, SolarAbundances=SolarAbundances)
    
    # Validate parameters
    if not ispec.valid_atmosphere_target(modeled_layers_pack, {'teff':teff, 'logg':logg, 'MH':MH, 'alpha':alpha}):
        msg = "The specified effective temperature, gravity (log g) and metallicity [M/H] \
                fall out of theatmospheric models."
        print (msg)

    # Prepare atmosphere model
    atmosphere_layers = ispec.interpolate_atmosphere_layers(modeled_layers_pack, {'teff':teff, 'logg':logg, 'MH':MH, 'alpha':alpha}, code=code)
    spec_abund, normal_abund, x_over_h, x_over_fe = ispec.determine_abundances(atmosphere_layers, \
            teff, logg, MH, alpha, linemasks, solar_abundances, microturbulence_vel = microturbulence_vel, \
            verbose=1, code=code)

    bad = np.isnan(x_over_h)
    fe1 = linemasks['element'] == "Fe 1"
    fe2 = linemasks['element'] == "Fe 2"
    print ("[Fe 1/H]: %.2f" % np.median(x_over_h[np.logical_and(fe1, ~bad)]))
    #print ("[X/Fe]: %.2f" % np.median(x_over_fe[np.logical_and(fe1, ~bad)]))
    print ("[Fe 2/H]: %.2f" % np.median(x_over_h[np.logical_and(fe2, ~bad)]))
    #print ("[X/Fe]: %.2f" % np.median(x_over_fe[np.logical_and(fe2, ~bad)]))
    return None

#To find value in array closest to given value
find_nearest = lambda array, value: (np.abs(array - value)).argmin()

def SaveParamsAstxt(params, errors, status, dump_file):
    print ("Saving results as'"+dump_file+"'...")
    f = open(dump_file, 'w') 
    f.write('#Results:\n')
    for k in list(params.keys()):
        f.write(k+"\t"+str(params[k])+"\t"+str(errors[k])+'\n')
    f.write('#Stats:\n')
    for s in list(status.keys()):
        f.write(s+": "+str(status[s])+'\n')
    # f.write(status)
    f.close()
    return None

def Parameters_using_SynthSpectra(ContinuumModel, initial_Teff, initial_logG, initial_MH, initial_vsini, initial_alpha='fixed', initial_limb_darkening_coeff=0.6, initial_vrad=0, Resolution=115000, max_iterations=15, code="spectrum", SaveDat=["Folder", "<Target>", "<JDdate>"],\
    AtomicLineList="GESv5_atom_hfs_iso.420_920nm/atomic_lines.tsv", ModelAtmo='MARCS.GES/', SolarAbundances='Grevesse.2007/stdatom.dat', free_params=["teff", "logg", "MH", "limb_darkening_coeff", 'vsini'], UseErrors=False):
    """
    This technique minimizes chi^2 between an observed spectrum and synthetic spectra computed on the fly using a specified radiative transfer code (code???) used, model atmosphere (ModelAtmo), atomic line list (AtomicLineList)
    """
    #code = ["spectrum","turbospectrum","sme", "synthe", "moog"] For some reason "moog" doesn't work
    #UseErrors = if want to use errors to weight lines. Think it skews results
    # NOTE: The line selection was built based on a solar spectrum with R ~ 47,000 and GES/VALD atomic linelist.
    normalized_star_spectrum = ContinuumModel
    # Use a fixed value because the spectrum is already normalized
    star_continuum_model = ispec.fit_continuum(normalized_star_spectrum, fixed_value=1.0, model="Fixed value")    
    # normalized_star_spectrum = ispec.convolve_spectrum(normalized_star_spectrum, to_resolution, from_resolution)
 
    #--- Model spectra ----------------------------------------------------------
    # Parameters
    EnhAbud = False
    if initial_alpha is None:
        initial_alpha = ispec.determine_abundance_enchancements(initial_MH)
        logging.info( 'Initial abund enhancement from metallicity: '+str(initial_alpha))
    if type(initial_alpha) == str:
        if initial_alpha.lower() == 'fixed' or initial_alpha.lower() == 'fix':
            EnhAbud, initial_alpha = True, ispec.determine_abundance_enchancements(initial_MH)
    initial_vmic = ispec.estimate_vmic(initial_Teff, initial_logG, initial_MH)
    initial_vmac = ispec.estimate_vmac(initial_Teff, initial_logG, initial_MH)
    logging.info("Initial Vmic and Vmac based off initial Teff, logG, and Z = "+str(initial_vmic)+', '+str(initial_vmac))
    initial_R = Resolution

    # Selected model amtosphere, linelist and solar abundances
    #Load model atmospheres
    model = ispec_dir+"/input/atmospheres/"+ModelAtmo #options: "MARCS.GES/", "MARCS.APOGEE/", "ATLAS9.APOGEE/", "ATLAS9.Castelli/", "ATLAS9.Kurucz/", "ATLAS9.Kirby/"
    modeled_layers_pack = ispec.load_modeled_layers_pack(model)

    # Load SPECTRUM abundances
    if not SolarAbundances and "ATLAS" in model:
        solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.1998/stdatom.dat" #TODO!!! need to figure out which model is the most appropriate 
    elif not SolarAbundances: # MARCS
        solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.2007/stdatom.dat"
    else: #SolarAbundances file is specified. Thus, use that one
        solar_abundances_file = ispec_dir + "/input/abundances/"+SolarAbundances # options: Grevesse.2007/stdatom.dat, Grevesse.1998/stdatom.dat, Asplund.2005/stdatom.dat, Asplund.2009/stdatom.dat, Anders.1989/stdatom.dat
    solar_abundances = ispec.read_solar_abundances(solar_abundances_file)
    
    # Load chemical information and linelist
    atomic_linelist_file = ispec_dir + "/input/linelists/transitions/"+AtomicLineList
    atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file, wave_base=np.min(normalized_star_spectrum['waveobs']), wave_top=np.max(normalized_star_spectrum['waveobs']))
    atomic_linelist = atomic_linelist[atomic_linelist['theoretical_depth'] >= 0.01] # Select lines that have some minimal contribution in the sun

    isotopes = ispec.read_isotope_data(ispec_dir + "/input/isotopes/SPECTRUM.lst")

    # Free parameters
    # free_params = ["teff", "logg", "MH", "alpha", "vmic", "vmac", "vsini", "R", "vrad", "limb_darkening_coeff"]

    # Free individual element abundance
    free_abundances, linelist_free_loggf = None, None

    segments = ispec.read_segment_regions(ispec_dir + "/input/regions/fe_lines_segments.txt") #using iron lines to determin metallicity

    emp_vmac, emp_vmic = True, True
    if "vmac" in free_params:
        emp_vmac = False
    if "vmic" in free_params:
        emp_vmic = False
    print ("free_params:", free_params)

    obs_spec, modeled_synth_spectrum, params, errors, abundances_found, loggf_found, status, stats_linemasks = \
            ispec.model_spectrum(normalized_star_spectrum, star_continuum_model, \
            modeled_layers_pack, atomic_linelist, isotopes, solar_abundances, free_abundances, linelist_free_loggf, initial_Teff, \
            initial_logG, initial_MH, initial_alpha, initial_vmic, initial_vmac, initial_vsini, \
            initial_limb_darkening_coeff, initial_R, initial_vrad, free_params, segments=segments, \
            linemasks=None, enhance_abundances=EnhAbud, use_errors=UseErrors, vmic_from_empirical_relation=emp_vmic, \
            vmac_from_empirical_relation=emp_vmac, max_iterations=max_iterations, tmp_dir=None, code=code)
    ##--- Save results -------------------------------------------------------------
    if SaveDat is not None:
        dump_file = SaveDat[0]+'/'+SaveDat[1]+"_results_"+SaveDat[2]+".txt"
        SaveParamsAstxt(params, errors, status, dump_file)
    return params, errors, status

def Parameters_using_grid(ContinuumModel, init_Teff, init_logG, init_MH, init_Vsini, init_Alpha='fixed', init_LD_coeff = 0.6, Resolution = 115000, init_Vrad = 0, UseErrors = False,\
    max_iterations = 30, FreeParms = ["teff", "logg", "MH", "limb_darkening_coeff", "vsini"], GridDir = "/input/grid/SPECTRUM_MARCS.GES_GESv6_atom_hfs_iso.480_680nm/", SaveDat = ["Folder", "<Target>", "<JDdate>"]):
    """
    This technique minimizes chi^2 between an observed spectrum and interpolated spectra from a pre-computed grid
    spectrum must already be radial velocity corrected!!!
    """
    #UseErrors = if want to use errors to weight lines. Think it skews results
    # FreeParms options: ["teff", "logg", "MH", "alpha", "vmic", "vmac", "vsini", "R", "vrad", "limb_darkening_coeff"]
    #GridDir options (as of 7/3/20): "SPECTRUM_MARCS.GES_GESv6_atom_hfs_iso.480_680nm" or "SYNTHE_ATLAS9.Castelli_SPECTRUM.380_900nm"
    #
    #--- Normalize -------------------------------------------------------------
    normalized_star_spectrum = ContinuumModel
    # Use a fixed value because the spectrum is already normalized
    star_continuum_model = ispec.fit_continuum(normalized_star_spectrum, fixed_value=1.0, model="Fixed value")
    #--- Model spectra ----------------------------------------------------------
    # Parameters
    EnhAbud = False
    if init_Alpha is None:
        init_Alpha = ispec.determine_abundance_enchancements(init_MH)
        logging.info( 'Initial abund enhancement from metallicity: '+str(init_Alpha))
    if type(init_Alpha) == str:
        if init_Alpha.lower() == 'fixed' or init_Alpha.lower() == 'fix':
            EnhAbud, init_Alpha = True, ispec.determine_abundance_enchancements(init_MH)
    init_Vmic = ispec.estimate_vmic(init_Teff, init_logG, init_MH)
    init_Vmac = ispec.estimate_vmac(init_Teff, init_logG, init_MH)
    logging.info("Initial Vmic and Vmac based off initial Teff, logG, and Z = "+str(init_Vmic)+', '+str(init_Vmac))
    init_R, code = Resolution, 'grid'

    precomputed_grid_dir = ispec_dir + GridDir

    atomic_linelist, isotopes, modeled_layers_pack = None, None, None
    solar_abundances, free_abundances, linelist_free_loggf = None, None, None

    # Read segments if we have them or...
    segments = ispec.read_segment_regions(ispec_dir + "/input/regions/fe_lines_segments.txt")
    # ... or we can create the segments on the fly:
    # line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_GES/{}_synth_good_for_params_all.txt".format(code))
    # segments = ispec.create_segments_around_lines(line_regions, margin=0.25)

    print ("precomputed_grid_dir:", precomputed_grid_dir)
    print ("FreeParms:", FreeParms)
    emp_vmac, emp_vmic = True, True
    if "vmac" in FreeParms:
        emp_vmac = False
    if "vmic" in FreeParms:
        emp_vmic = False
    obs_spec, modeled_synth_spectrum, params, errors, abundances_found, loggf_found, status, stats_linemasks = \
            ispec.model_spectrum(normalized_star_spectrum, star_continuum_model, modeled_layers_pack, atomic_linelist,\
            isotopes, solar_abundances, free_abundances, linelist_free_loggf, init_Teff, init_logG, init_MH, init_Alpha,\
            init_Vmic, init_Vmac, init_Vsini, init_LD_coeff, init_R, init_Vrad, FreeParms, segments=segments, \
            enhance_abundances=EnhAbud, use_errors=UseErrors, vmic_from_empirical_relation=emp_vmic, \
            vmac_from_empirical_relation=emp_vmac, max_iterations=max_iterations, tmp_dir=None, code=code, 
            precomputed_grid_dir=precomputed_grid_dir)
    ##--- Save results -------------------------------------------------------------
    if SaveDat is not None:
        dump_file = SaveDat[0]+'/'+SaveDat[1]+"_results_"+SaveDat[2]+".txt"
        SaveParamsAstxt(params, errors, status, dump_file)
    return params, errors, status, stats_linemasks

def Abundances_using_grid(ContinuumModel, init_Teff, init_logG, init_MH, init_Vsini, init_Alpha = 'fixed', init_LD_coeff = 0.6, Resolution = 115000, init_Vrad = 0, element_name= ["Ca", "C", "O", "Mg", "Si", "Na", "K", "Ti", "V"],\
    max_iterations = 30, FreeParms = ["vrad"], GridDir = "/input/grid/SPECTRUM_MARCS.GES_GESv6_atom_hfs_iso.480_680nm/", SaveDat = ["Folder", "<Target>", "<JDdate>"]):
    #CODE IS BROKEN!!! CAN'T GET IT TO WORK WITH A GRID, BUT IT DOES WORK WITH SYNTHETIC DATA. WORKING WITH THAT METHOD FROM NOW ON
    # !!!!!!!!Follow this and the determine_abundances_using_synth_spectra() vs. determine_astrophysical_parameters_using_synth_spectra() fuctions in example.py for template on how to write this properly!!!!!!!!!!!
    #!!!!!!!TODO: 'GridDir' needs to include wavelengths where all of these species are. Aka 380-935
    """
    This technique minimizes chi^2 between an observed spectrum and interpolated spectra from a pre-computed grid
    spectrum must already be radial velocity corrected!!!
    """
    # FreeParms options: ["teff", "logg", "MH", "alpha", "vmic", "vmac", "vsini", "R", "vrad", "limb_darkening_coeff"]
    #GridDir options (as of 7/3/20): "SPECTRUM_MARCS.GES_GESv5_atom_hfs_iso.480_680nm" or "SYNTHE_ATLAS9.Castelli_SPECTRUM.380_900nm"
    #
    #--- Normalize -------------------------------------------------------------
    normalized_star_spectrum = ContinuumModel
    # Use a fixed value because the spectrum is already normalized
    star_continuum_model = ispec.fit_continuum(normalized_star_spectrum, fixed_value=1.0, model="Fixed value")
    #--- Model spectra ----------------------------------------------------------
    # Parameters
    EnhAbud = False
    if init_Alpha is None:
        init_Alpha = ispec.determine_abundance_enchancements(init_MH)
        logging.info( 'Initial abund enhancement from metallicity: '+str(init_Alpha))
    if type(init_Alpha) == str:
        if init_Alpha.lower() == 'fixed' or init_Alpha.lower() == 'fix':
            EnhAbud, init_Alpha = True, ispec.determine_abundance_enchancements(init_MH)
    init_Vmic = ispec.estimate_vmic(init_Teff, init_logG, init_MH)
    init_Vmac = ispec.estimate_vmac(init_Teff, init_logG, init_MH)
    logging.info("Initial Vmic and Vmac based off initial Teff, logG, and Z = "+str(init_Vmic)+', '+str(init_Vmac))
    init_R, code = Resolution, 'grid'

    precomputed_grid_dir = ispec_dir + GridDir

    atomic_linelist, isotopes, modeled_layers_pack, linelist_free_loggf = None, None, None, None #don't think I need these because fitting against model/grid spectra???
    solar_abundances, free_abundances = None, None
    # solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.2007/stdatom.dat" # !!!TODO: need to figure out which model is best, with the full wavelength range of my spectrum 
    # solar_abundances = ispec.read_solar_abundances(solar_abundances_file) # Load SPECTRUM abundances

    # # Free individual element abundance (WARNING: it should be coherent with the selected line regions!)
    # chemical_elements_file = ispec_dir + "/input/abundances/chemical_elements_symbols.dat"
    # chemical_elements = ispec.read_chemical_elements(chemical_elements_file)

    # free_abundances = ispec.create_free_abundances_structure(element_name, chemical_elements, solar_abundances)
    # free_abundances['Abund'] += init_MH # Scale to metallicity

    # element_name = "Si"
    # free_abundances = ispec.create_free_abundances_structure([element_name], chemical_elements, solar_abundances)
    # free_abundances['Abund'] += init_MH # Scale to metallicity

    # Line regions
    line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_GES/{}_synth_good_for_params_all.txt".format(code)) #!!!!TODO: Need to find better linelist, particularly that encompass a wider wavelength range, for more species
    # line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_GES/{}_synth_good_for_params_all_extended.txt".format(code))
    #line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/{}_synth_good_for_params_all.txt".format(code))
    #line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/{}_synth_good_for_params_all_extended.txt".format(code))
    
    # Select only the lines to get abundances from
    print ("elements:", element_name)
    LineRegions = []
    for en in range(len(element_name)):
        el_nm = element_name[en]
        line_regions1 = line_regions[np.logical_or(line_regions['note'] == el_nm+' 1', line_regions['note'] == el_nm+' 2')]
        for lr_i in line_regions1:
            LineRegions.append(lr_i)
    LineRegions = np.array(LineRegions)
    # print ("line_regions:", line_regions)
    # print ("type(line_regions)", type(line_regions))
    # print ("LineRegions:", LineRegions)
    # print ("type(LineRegions):", type(LineRegions))
    # sys.exit()
    segments = ispec.adjust_linemasks(normalized_star_spectrum, LineRegions, max_margin=0.5)

    obs_spec, modeled_synth_spectrum, params, errors, abundances_found, loggf_found, status, stats_linemasks = \
            ispec.model_spectrum(normalized_star_spectrum, star_continuum_model, modeled_layers_pack, atomic_linelist,\
            isotopes, solar_abundances, free_abundances, linelist_free_loggf, init_Teff, init_logG, init_MH, init_Alpha,\
            init_Vmic, init_Vmac, init_Vsini, init_LD_coeff, init_R, init_Vrad, FreeParms, segments=segments, \
            enhance_abundances=EnhAbud, use_errors = True, vmic_from_empirical_relation = True, \
            vmac_from_empirical_relation = True, max_iterations=max_iterations, tmp_dir = None, code=code, 
            precomputed_grid_dir=precomputed_grid_dir)
    ##--- Save results -------------------------------------------------------------
    print ("abundances_found:", abundances_found)
    if SaveDat is not None:
        dump_file = SaveDat[0]+'/'+SaveDat[1]+"_results_"+SaveDat[2]+".dump"
        logging.info("Saving results as'"+dump_file+"'...")
        ispec.save_results(dump_file, (params, errors, status))
#     # If we need to restore the results from another script:
#     params, errors, abundances_found, loggf_found, status, stats_linemasks = ispec.restore_results(dump_file)

#     logging.info("Saving synthetic spectrum...")
#     synth_filename = "example_modeled_synth_%s.fits" % (code)
#     ispec.write_spectrum(modeled_synth_spectrum, synth_filename)
    return params, errors, status

def Abundances_using_SynthSpectra(ContinuumModel, init_Teff, init_logG, init_MH, init_Vsini, init_Alpha = 'fixed', init_LD_coeff = 0.6, Resolution = 115000, init_Vrad = 0, element_names= ["Ca", "C", "O", "Mg", "Si", "Na", "K", "Ti", "V"],\
    max_iterations = 30, FreeParms = [], code="spectrum", ModelAtmo='MARCS.GES/', AtomicLineList = "/VALD.300_1100nm/atomic_lines.tsv", SolarAbundances=None):
    #--- Normalize -------------------------------------------------------------
    normalized_star_spectrum = ContinuumModel
    # Use a fixed value because the spectrum is already normalized
    star_continuum_model = ispec.fit_continuum(normalized_star_spectrum, fixed_value=1.0, model="Fixed value")
    #--- Model spectra ----------------------------------------------------------
    # Parameters
    EnhAbud = False
    if init_Alpha is None:
        init_Alpha = ispec.determine_abundance_enchancements(init_MH)
        logging.info( 'Initial abund enhancement from metallicity: '+str(init_Alpha))
    if type(init_Alpha) == str:
        if init_Alpha.lower() == 'fixed' or init_Alpha.lower() == 'fix':
            EnhAbud, init_Alpha = True, ispec.determine_abundance_enchancements(init_MH)
    init_Vmic = ispec.estimate_vmic(init_Teff, init_logG, init_MH)
    init_Vmac = ispec.estimate_vmac(init_Teff, init_logG, init_MH)
    logging.info("Initial Vmic and Vmac based off initial Teff, logG, and Z = "+str(init_Vmic)+', '+str(init_Vmac))

    # Selected model amtosphere, linelist and solar abundances
    # Load model atmospheres #TODO!!! need to figure out which model is the most appropriate 
    model = ispec_dir + "/input/atmospheres/"+ModelAtmo  #options: "MARCS.GES/", "MARCS.APOGEE/", "ATLAS9.APOGEE/", "ATLAS9.Castelli/", "ATLAS9.Kurucz/", "ATLAS9.Kirby/" 
    modeled_layers_pack = ispec.load_modeled_layers_pack(model)

    # Load SPECTRUM abundances
    if not SolarAbundances and "ATLAS" in model:
        solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.1998/stdatom.dat" #TODO!!! need to figure out which model is the most appropriate 
    elif not SolarAbundances: # MARCS
        solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.2007/stdatom.dat"
    else: #SolarAbundances file is specified. Thus, use that one
        solar_abundances_file = ispec_dir + "/input/abundances/"+SolarAbundances # options: Grevesse.2007/stdatom.dat, Grevesse.1998/stdatom.dat, Asplund.2005/stdatom.dat, Asplund.2009/stdatom.dat, Anders.1989/stdatom.dat
    solar_abundances = ispec.read_solar_abundances(solar_abundances_file)

    isotopes = ispec.read_isotope_data(ispec_dir + "/input/isotopes/SPECTRUM.lst") #No idea what this is. need to figure out which model is the most appropriate 

    # Load chemical information and linelist.      #TODO!!! need to figure out which linelist are the most appropriate 
    atomic_linelist_file = ispec_dir + "/input/linelists/transitions/"+AtomicLineList # options: "VALD.300_1100nm/atomic_lines.tsv", "VALD.1100_2400nm/atomic_lines.tsv", "GESv6_atom_hfs_iso.420_920nm/atomic_lines.tsv", "GESv6_atom_nohfs_noiso.420_920nm/atomic_lines.tsv"
    atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file, wave_base=np.min(normalized_star_spectrum['waveobs']), wave_top=np.max(normalized_star_spectrum['waveobs']))
    atomic_linelist = atomic_linelist[atomic_linelist['theoretical_depth'] >= 0.01] # Select lines that have some minimal contribution in the sun

    # Free individual element abundance (WARNING: it should be coherent with the selected line regions!)
    chemical_elements_file = ispec_dir + "/input/abundances/chemical_elements_symbols.dat" #think this is fine. just list of all elements and their solar abundances rank
    chemical_elements = ispec.read_chemical_elements(chemical_elements_file)

    free_abundances = ispec.create_free_abundances_structure(element_names, chemical_elements, solar_abundances)
    free_abundances['Abund'] += init_MH # Scale to metallicity
    linelist_free_loggf = None

    # Line regions
    line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_GES/{}_synth_good_for_params_all.txt".format(code)) #!!!!TODO: Need to find better linelist, particularly, one that encompass a wider wavelength range, for more species
    #line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_GES/{}_synth_good_for_params_all_extended.txt".format(code))
    #line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/{}_synth_good_for_params_all.txt".format(code))
    #line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/{}_synth_good_for_params_all_extended.txt".format(code))
    
    # Select only the lines to get abundances from
    LineRegions = []
    for en in range(len(element_names)):
        el_nm = element_names[en]
        line_regions1 = line_regions[np.logical_or(line_regions['note'] == el_nm+' 1', line_regions['note'] == el_nm+' 2')]
        for lr_i in line_regions1:
            LineRegions.append(lr_i)
    LineRegions = np.array(LineRegions)
    # line_regions = line_regions[np.logical_or(line_regions['note'] == element_names+' 1', line_regions['note'] == element_names+' 2')]
    line_regions = ispec.adjust_linemasks(normalized_star_spectrum, LineRegions, max_margin=0.5)

    # Read segments if we have them or...
    #segments = ispec.read_segment_regions(ispec_dir + "/input/regions/fe_lines_segments.txt")
    # ... or we can create the segments on the fly:
    segments = ispec.create_segments_around_lines(line_regions, margin=0.25)

    obs_spec, modeled_synth_spectrum, params, errors, abundances_found, loggf_found, status, stats_linemasks = \
            ispec.model_spectrum(normalized_star_spectrum, star_continuum_model, \
            modeled_layers_pack, atomic_linelist, isotopes, solar_abundances, free_abundances, linelist_free_loggf, init_Teff, \
            init_logG, init_MH, init_Alpha, init_Vmic, init_Vmac, init_Vsini, \
            init_LD_coeff, Resolution, init_Vrad, FreeParms, segments=segments, \
            linemasks=line_regions, \
            enhance_abundances=True, \
            use_errors = True, \
            vmic_from_empirical_relation = False, \
            vmac_from_empirical_relation = False, \
            max_iterations=max_iterations, \
            tmp_dir = None, \
            code=code)
    # ##--- Save results -------------------------------------------------------------
    # dump_file = "example_results_synth_abundances_%s.dump" % (code)
    # logging.info("Saving results...")
    # ispec.save_results(dump_file, (params, errors, abundances_found, loggf_found, status, stats_linemasks))
    # # If we need to restore the results from another script:
    # params, errors, abundances_found, loggf_found, status, stats_linemasks = ispec.restore_results(dump_file)

    # logging.info("Saving synthetic spectrum...")
    # synth_filename = "example_modeled_synth_abundances_%s.fits" % (code)
    # ispec.write_spectrum(modeled_synth_spectrum, synth_filename)
    return None

def ReduceData(Target, DataFiles, RA_Dec=None, DataType='HARPS', Folder=None, RV_Corr=True, RVlims=[-400,400], RVstep=1.0, RV_Mask = "input/linelists/CCF/HARPS_SOPHIE.G2.375_679nm/mask.lst",\
    SaveRVdat=True, RVextraSave = '', FilterCosmics=True, CosRayErrorThresh = [5,2], IgnorEmissions=[], SaveCosmicFiltered=False, CosmicFilterWidth=.08, FitContinuum=True,\
    Lines=None, Res = 115000, Deg=2, SaveContiuum=True, EstErr=False, NumP=5, Sigs=2, toPlot=False, ContFit_Model="Splines", CutSpec=None, ContFitErr=False, Suffix=None):
    #DataFiles = list of file names of data to be reduced #RA_Dec = [[RAhours, RAmins, RAsec],[Decdeg, Decmin, Decsec]]. If None assuming no Barycentric correction 
    #RVlims = [RV_lower_limit, RV_upper_limi]. RV fit limits. Constrain it based on previous RV calculations to help decrease run time
    #DataType= string of insturment where the data came from i.e. 'HARPS', 'CORALIE', etc. #NumP = the number of flux points in a block used for estimating errors
    #Sigs = number of significant figures to save data in (particularly rv data), #Suffix= to remove any excess naming of base file name in  README.txt
    #Lines=lines where continuum spectrum would be ignored #RVextraSave = additional string part of name of RV files to be saved
    #ContFit_Model= method to fit continuum. "Splines" or "Polynomy", CutSpec=wavelength ranges that will be removed from the output continuum, because they can't be properly normalized. List of 2D list
    ###================To load the data================###
    if not Folder:    
        Folder = Target #If folder is not specified, assuming that the folder is the same name as the target
    if type(DataFiles[0]) != str:
        sys.exit("Type of DataFiles[0] is "+str(type(DataFiles[0]))+". Needs to be a string of directory file names.")
    OGSpectra, ReadSpectra, BASE, Adds, i = [], [], [], np.zeros(len(DataFiles)), 0
    for File in DataFiles:
        if File.find('.txt') == -1: #spectral file is .fits file
            BASE.append(File[:File.find('.fits')])
        if File.find('.fits') == -1: #spectral file is .txt file
            BASE.append(File[:File.find('.txt')])
        mu_cas_spectrum = ispec.read_spectrum(File)
        if np.min(mu_cas_spectrum['waveobs']) > 1000: #Since assuming working with near UV to near IR data. Then a min wavelength greater than 1k must be in Ang, convert to nm
            mu_cas_spectrum['waveobs'] = mu_cas_spectrum['waveobs']/10 #HARPS and FEROS data are originally in angstroms
        min_flux = np.min(mu_cas_spectrum['flux'])
        if min_flux < 0: # to keep track of the offset needed to make the spectrum non-negative
            Adds[i] = abs(min_flux)
        i+=1
        OGSpectra.append(np.copy(mu_cas_spectrum)), ReadSpectra.append(mu_cas_spectrum)
    logging.info("\033[1mRead "+str(len(ReadSpectra))+" files.\033[0m\n")   #Also returns BASE and OGSectra
    # for i in range(len(BASE)):
    #   plt.plot(ReadSpectra[i]['waveobs'], ReadSpectra[i]['flux'], label = BASE[i])
    # plt.legend()
    # plt.show()
    # plt.close()
    # sys.exit()
    ###================================================###

    ###================Radial Velocity (Earth and Star V) correction================###
    path_name, SubdirS = '', DataFiles[0].split('/')[:-1] #to get the path name from the .fits files
    for S in SubdirS:
        path_name += S+'/'
    BaryVs, starRVs, RVerrs = np.zeros(len(DataFiles)), np.zeros(len(DataFiles)), np.zeros(len(DataFiles)) #won't be used if not calculateing barycentric or stellar radial velocities
    if not RV_Corr:
        SaveRVdat = None
    if RA_Dec is not None or RV_Corr:
        Dates, Times = GetDateTime(DataFiles, DataType=DataType, extras=Suffix)
        for s in range(len(DataFiles)):
            if RA_Dec is not None:
                ReadSpectra[s]['flux'] = ReadSpectra[s]['flux'] + Adds[s] #have to add offset, so no negative flux, to run with iSpec
                BaryVs[s], ReadSpectra[s] = calculate_barycentric_velocity(ReadSpectra[s], Dates[s], Times[s], RA_Dec[0], RA_Dec[1], Correct=True, log=False)
                ReadSpectra[s]['flux'] = ReadSpectra[s]['flux'] - Adds[s]
            if RV_Corr:
                ReadSpectra[s]['flux'] = ReadSpectra[s]['flux'] + Adds[s] #have to add offset, so no negative flux, to run with iSpec
                starRVs_i, RVerrs_i, ReadSpectra[s] = determine_radial_velocity_with_mask(ReadSpectra[s], RV_Mask, Vstp=RVstep, LwrVlim=RVlims[0], UpVlim=RVlims[1], fileName=path_name)
                starRVs[s], RVerrs[s] = FindSigFigs(starRVs_i, RVerrs_i, sifdig=Sigs)
                ReadSpectra[s]['flux'] = ReadSpectra[s]['flux'] - Adds[s]
            if SaveRVdat:
                if EstErr: 
                    estimated_snr, ReadSpectra[s] = Estimate_SNR(ReadSpectra[s], NumPoints=NumP, Est_Err=True)
                    SaveWavnFlux(BASE[s]+RVextraSave+'_RVcorr.fits', ReadSpectra[s]['waveobs'], ReadSpectra[s]['flux'], Errors=ReadSpectra[s]['err'])
                else:
                    SaveWavnFlux(BASE[s]+RVextraSave+'_RVcorr.fits', ReadSpectra[s]['waveobs'], ReadSpectra[s]['flux'])
                logging.info("\033[1mCalculated Barycentric Velocities of "+str(BaryVs[s])+"[km/s]\033[0m")
                logging.info("\033[1mand Radial velocities of "+str(starRVs[s])+"[km/s]\033[0m")
                logging.info("\033[1mwith errors of "+str(RVerrs[s])+"[km/s]\033[0m\n")
        if SaveRVdat:
            if SaveRVdat == True: #if SaveRVdat is just turned on don't add extra string to file name
                SaveRVdat = ''
            else: #else, add that extra string to the name of the saved RVdata.txt file
                pass #This is different from 'RVextraSave' because it's only written on the 'RVdata.txt' file and not also the RVcorrected spectrum
            f = open(Folder+'/'+Target+'_'+DataType+RVextraSave+SaveRVdat+'_RVdata.txt', 'w') 
            f.write('#JD     Barycentric V [km/s]     RV [km/s]     err [km/s]\n')
            #spectrum['waveobs'] * np.sqrt((1.-(velocity*1000.)/c)/(1.+(velocity*1000.)/c)) #How wavelength shift is calculated
            times = []
            for d in range(len(Dates)):
                t = Time(Dates[d]+'T'+Times[d], format='isot', scale='utc')
                times.append(t.jd) 
                f.write(str(times[-1])+'     '+str(BaryVs[d])+'     '+str(starRVs[d])+'     '+str(RVerrs[d])+'\n') 
            f.write('#TimeSpan[days]     Mean_Bary_V     Mean_RV     Mean_Err\n')
            f.write('#'+str(np.max(times)-np.min(times))+'     '+str(np.mean(BaryVs))+'     '+str(np.mean(starRVs))+'     '+str(np.mean(RVerrs)))
            f.close()
            logging.info('\033[1m RV info saved as "'+Target+'_'+DataType+RVextraSave+'_RVdata.txt"'+'\033[0m\n') 
        if toPlot: #this needs to be saved as an interactive plot
            fig,ax = plt.subplots()
            plt.title('RV_correction')
            for i in range(len(BASE)):
                plt.plot(OGSpectra[i]['waveobs'], OGSpectra[i]['flux'], '-', alpha = .5, label = BASE[i]+'_OG')
                plt.plot(ReadSpectra[i]['waveobs'], ReadSpectra[i]['flux'], '--', label = BASE[i])
            plt.legend()
            pickle.dump(fig, open(path_name+'RV_correction.fig.pkl', 'wb'))
            print ("Saved interactive plot as '"+path_name+"RV_correction.fig.pkl'")
        if not FitContinuum: #if not fitting continuum then this is needed only to get the velocity corrections, so return those values as well
            return BaryVs, starRVs, RVerrs, ReadSpectra
    ###================================================###

    ###================Filter Cosmic Rays ================###
    #TODO: REWRITE COS RM CODE. This doesn't work right
    # if SaveCosmicFiltered:
    #   save, RMedCosNm = BASE, '_RMcosmics'
    # else:
    #   save, RMedCosNm = None, ''
    # ReadSpectra = filter_cosmic_rays(ReadSpectra, ErrThreshlds=CosRayErrorThresh, Save=save, IgnoreRegion = IgnorEmissions, CosWidth=CosmicFilterWidth)
    # logging.info("\n")
    ###================================================###

    ###================Continuum Fit================###
    if FitContinuum:
        FullNormSpectra = []
        for B in range(len(BASE)):
            if CutSpec:
                CutName = 'Cut' #to save that the normalized data has been cut in the name
                if type(CutSpec[0]) != list: #in the case when only one cut region is given, convert the 2d list to a list of 2d list, so it's in the same formate as if there were more than one cut region
                    CutSpec = [CutSpec]
                Spectra_cut = np.copy(ReadSpectra[B])
                Screen = np.array([])
                for ex in CutSpec:
                    exclude = np.where((ex[0]<Spectra_cut['waveobs']) & (Spectra_cut['waveobs']<ex[1]))[0]
                    Screen = np.concatenate((Screen, exclude))
                Spectra_cut = np.delete(Spectra_cut, Screen.astype(int), axis=0) #omit specific regions when fitting continuum
            else:
                Spectra_cut, CutName = ReadSpectra[B], '' #the spectrum is not cut
            # Spectra_cut['err'] = np.zeros(len(Spectra_cut['err'])) #the errors messes up the continuum fit. Thus, set it to 0.
            NormSpec, ContinM, StarSpec = find_continuum_regions(Spectra_cut, Model=ContFit_Model, IgnRegion=Lines, Res=Res, Deg=Deg, Normalize=True, PLOT=toPlot, UseErr=ContFitErr, plot_save_path=path_name)
            # if ContFit_Model == "Polynomy":
            #     X = np.loadtxt(ispec_dir+'/input/regions/'+Lines)
            #     waveD, waveU = X[:,0], X[:,1]
            #     UsedWavs, NormSpec = np.copy(ReadSpectra[B]), np.copy(ReadSpectra[B])
            #     Screen = np.array([])
            #     for w in range(len(waveD)):
            #         exclude = np.where((waveD[w]<UsedWavs['waveobs']) & (UsedWavs['waveobs']<waveU[w]))[0]
            #         Screen = np.concatenate((Screen, exclude))
            #     UsedWavs = np.delete(UsedWavs, Screen.astype(int), axis=0) #omit specific regions when fitting continuum
            #     coeffs = np.polyfit(UsedWavs['waveobs'], UsedWavs['flux'], Deg) #do a quick Nth order poly to fit data
            #     ContinM = np.polyval(coeffs, ReadSpectra[B]['waveobs']) 
            #     NormSpec['flux'], StarSpec = ReadSpectra[B]['flux']/ContinM, ReadSpectra[B]
            #     if toPlot: #to plot the simple poly correction on top of the iSpec poly function
            #         plt.figure(1)
            #         plt.plot(NormSpec['waveobs'], NormSpec['flux'], 'r.')
            #         plt.plot(NormSpec['waveobs'], ContinM, 'y--')
            #         plt.figure(2)
            #         plt.plot(NormSpec['waveobs'], NormSpec['flux'], 'r.')
            if SaveContiuum:
                # f = open(BASE[B]+RMedCosNm+'_ContinuumNorm.txt', 'w')
                f = open(BASE[B]+'_ContinuumNorm'+CutName+'.txt', 'w')
                if CutSpec:
                    f2 = open(BASE[B]+'_'+CutName+'.txt', 'w') #to save the origianl (non-normalized) data, but cut the same way the normalized data is, in case need to use both
                    f2.write('# waveobs   flux   err\n')
                f.write('# waveobs   flux   err\n')
                for w in range(len(NormSpec['waveobs'])):
                    f.write(str(NormSpec['waveobs'][w])+'   '+str(NormSpec['flux'][w])+'   '+str(NormSpec['err'][w])+'\n')
                    if CutSpec:
                        f2.write(str(StarSpec['waveobs'][w])+'   '+str(StarSpec['flux'][w])+'   '+str(StarSpec['err'][w])+'\n')
                f.close()
                if CutSpec:
                    f2.close()
                # logging.info('\033[1mSaved continuum normalized spectra as "'+BASE[B]+RMedCosNm+'_ContinuumNorm.txt\033[0m"\n')
                logging.info('\033[1mSaved continuum normalized spectra as "'+BASE[B]+'_ContinuumNorm.txt\033[0m"\n')
            FullNormSpectra.append(NormSpec)
        return FullNormSpectra, ReadSpectra
    ###================================================###
    return ReadSpectra

def SaveWavnFlux(FileName, wavelengths, Flux, Errors=None): #To save an array of wavelength and array of flux as a file in the form iSpec will accept (either .txt or .fits)
    spectrum = np.recarray((len(Flux), ), dtype=[('waveobs', float),('flux', float),('err', float)])
    if Errors is not None:
        pass
    else:
        Errors = np.zeros(len(wavelengths))
    spectrum['waveobs'], spectrum['flux'], spectrum['err'] = wavelengths, Flux, Errors
    ispec.write_spectrum(spectrum, FileName)
    print ('Saved spectra in iSpec format as "'+FileName+'"\n')
    return None

def SampleData(BaseFile, Type, choose=4, CosmicCorr=True, ContinNorm=True):
    #BaseFile = String containing the directory, and consistent folder name of file
    #ex: BaseFile = 'HATS29/ADP*.fits'
    # Files1, Files2, Files3 = glob.glob(BaseFile+'*'+Type),  glob.glob(BaseFile+'*_RMcosmics'+Type), glob.glob(BaseFile+'*_RMcosmics_ContinuumNorm'+Type) 
    Files1, Files2 = glob.glob(BaseFile+'*'+Type),  glob.glob(BaseFile+'*_ContinuumNorm'+Type) 
    Files = []
    for f in Files1: #To get the base name of 'choose' random selected spectra
        if f in Files2:
            pass
        # elif f in Files3:
        #     pass
        else:
            Files.append(str(f))
    FITs = np.random.choice(Files, choose, replace=False) 
    Cosmic, Contin, OGFile= [], [], []
    for F in FITs: #To get the base name of 'choose' random selected spectra
        OGFile.append(str(F))
        # if CosmicCorr:
        #    Cosmic.append(str(F[:F.find(Type)]+'_RMcosmics.txt'))
        if ContinNorm:
           Contin.append(str(F[:F.find(Type)]+'_ContinuumNorm.txt'))
           # Contin.append(str(F[:F.find(Type)]+'_RMcosmics_ContinuumNorm.txt'))
    File_Names = {}
    File_Names['OG_Files'] = OGFile
    if CosmicCorr:
        File_Names['CosmicCorr_Files'] = Cosmic   
    if ContinNorm:
        File_Names['ContinNorm_Files'] = Contin                  
    return File_Names

def RM_SNR_estimates(Files): # code to fixed the last two lines of some of the RVcorrected.txt files that I added the SNR estimates at the end of the files
    for F in Files:
        (print) ("Working with file: \033[1m"+F+"'\033[0m" )
        file=open(F,'r')
        lines = file.readlines()
        if '# estimated SNR:' in lines[len(lines)-2]:
            print ('Removing the SNR estimates from end of this file!')
            lines = lines[:-2]
            fd=open(F,"w")
            for l in lines:
                fd.write(l)
            fd.close()          
    return None

InitialGuess = lambda TrueVale, PercErrStd : np.random.normal(TrueVale, abs(TrueVale*PercErrStd), 1000)

#NOT TOO USEFUL!!! because for some reason can plot it from my mac machine after being created in a linux. Works on a linux though?? 
def ReplotInteractive(Fig): #to replot interactive plots
    figx = pickle.load(open(Fig, 'rb')) 
    plt.show() #strange glitch: in order to interact with figure, have to 1st do figure manipulation THEN make the figure larger or smaller
    plt.close()
    return None

def ParameterizeUncertainty(): #to run parameter estimation with both approaches (grid vs syntethic), holding different parameters fix (combinatorics of "teff", "logg",'vsini' held fixed). This will help characterize the level of uncertainty
    test = []# list of different parameters to change for each different test
    for t in tests:
        params, errors, status, stats_linemasks = Parameters_using_grid()
    return None

def Cut(file, CutRange, outfile=None, Replace=False): #To cut the spectra so only using specifed wavelengths. This is generally needed because some of the edge cases of the FEROS data goes wonky
    #CutRange=2 element list wavelength range where data will be used, in nm
    spec = ispec.read_spectrum(file)
    if np.min(spec['waveobs']) > 1000: #Since assuming working with near UV to near IR data. Then a min wavelength greater than 1k must be in Ang, convert to nm
        spec['waveobs'] = spec['waveobs']/10 #might as well convert to nm here
    used = np.where((CutRange[0]<spec['waveobs'])&(spec['waveobs']<CutRange[1]))[0]
    if Replace: #delete old file and replace it with the new cut file
        os.remove(file)
        outfile = file
    else:
        if not outfile:
            outfile = file[:file.rfind('.')]+'_CUT'+file[file.rfind('.'):] #if no outfile is specified, just use the same outfile, but with the added '_CUT' label
    ispec.write_spectrum(spec[used], outfile)
    print ("saved cut spectrum as '"+outfile+"'")
    return None    

from scipy.special import wofz
from tqdm import tqdm
from scipy import optimize
def ClearFEROS_outliers(spec, Sig_clips=5, Save=True, UseWavelengths=[388,402], Wav_window=5): 
    """
    For some reason the low resolution FEROS data has way too many outliners (too many to be cosmic rays - 
    maybe its similar to the issue with the bias subtraction with the HARPS data). This code is desiged to clear those out. This is done by 
    1) grouping the spectra to sections of 'Wav_window' [nm] (try 5nm = 50Ang), 2) Take the mean counts of that region 
    (might have to try fitting a continuum to the region), 3) Find all data that's is more than 'Sig_clips'-sigma larger than the mean, 
    4) Take those outliers as max points to fit voigt profiles, and remove all point on the voigt fit
    Input:
        spec == ispec.read_spectrum() data
        Sig_clips == how many sigmas away from the mean are we going to tolerate, more than this and we model it out
        UseWavelengths == The wavelength range (in nm) that we are using, default the region where Cal H & K lines are calculated
        Save == if True, save the clipped data as clipped_spectra.fits in cwd, if 'Name' then save in the path and as file name specified in 'Name'
        Wav_window = wavelength size of bites that will be used to determine outliers
    Return:
        A list of the bad indeces that should be ignored. Indeces are given relative to the spectra array in the 'UseWavelengths' range
    """
    ########################### SUB-FUNCTIONS #############################
    def chisq_for_optimize(x0, data, mean_wave):
        """Generate the voigt model and evaluate the chi2 of profile vs model. Needed for scipy.optimize
        Inputs:
           x == len_4 list (or array) of voigt profile parameters: [alpha, gamma, center, scale]
           data == len_3 array (or list) of the data: np.array([waveobs, flux, err])
           mean_wave == where the voigt profile will be centered, based on the peak of the spectra
        Returns:
        chi squared of voigt profile against spectra
        """
        model = Voigt(data[0], x0[0], x0[1], mean_wave, x0[2])
        resids = (model - data[1])/np.mean(data[2]) #resids = (model-data)/err. !!!NOTE: for some reason I get wonky results if I use the full error array, so just take the mean err 

        return np.sum(resids*resids)

    def Voigt(x, alpha, gamma, center, scale):
        """Return the Voigt (convolution of a gaussian and Lorentzian profile) line shape at x, where 
        #gamma = HWHM Lorentzian component, #alpha = HWHM Gaussian component, center = the mean of the 
        #profile (the loaction of the peak) #scale = a term to multiple the voigt profile by, 
        #correlated to how high you want the peak of the profile to be. Without this the integral of the profile is normalized to one"""
        alpha, gamma = alpha, gamma
        sigma = alpha / np.sqrt(2 * np.log(2))
        x_prime = x-center
        General_Voit = np.real(wofz((x_prime + 1j*gamma)/sigma/np.sqrt(2))) / sigma /np.sqrt(2*np.pi)
        return General_Voit*scale

    #steps 2) Take the mean counts of that region, 3) excluding out all data that's is more than Sig_clips-sigma than the mean,
    # and 4) Take those outliers as max points to fit voigt profiles, and remove all point on the voigt fit
    def sigclip(data_seg): #the reused function to exclude the outlier data
        #data_seg = the dataset that needs to be sigma clipped
        #if this is the 1st iteration, pass and empty list
        WV, FLX, ERR = data_seg['waveobs'], data_seg['flux'], data_seg['err'] 
        Mean, Std = np.mean(FLX), np.std(FLX)
        Peaks = np.where(FLX > (Sig_clips*Std)+Mean)[0] #if it's more than 3*sigma than the mean, flag
        if len(Peaks) == 0: #if there are no peak found then return an empty array of elements to ignore
            return np.array([])
        #To find the peak of the peaks, which will be used as the central point for the voigt profiles
        diff_peaks = np.diff(Peaks)
        dividers = np.where(diff_peaks > 1)[0]
        unUsed_idx, min_voigt_peak = np.array([]), 0.01 #the min frational value where to consider the fitted voigt profile
        
        def Get_unUsed_idx(unUsed_idx, P_idxs): 
            """a function to figure out the unused indeces via fitting voigt profiles centered on the peak of points
            in a sub function because have to call this twice """
            idx_PoP = np.argmax(FLX[P_idxs])+P_idxs[0] #to find the Peak of the group of peaks. +P_idxs[0] to convert to window's index
            #So to not mess up the fit of a given peak fit by having the other peaks nearby, only do the fit from a narrow wavelength range
            centWav, bnds = WV[idx_PoP], .1 #ASSUMING THAT THE PEAKS AREN'T CLOSER THAN .1nm APARAT!!
            reg_inst = np.where((centWav-bnds < WV) & (WV < centWav+bnds))[0] 
            data = np.array([WV[reg_inst], FLX[reg_inst], ERR[reg_inst]])
            #get best fit of data
            x0_sys = [.001, .001, 5e-4] #no idea what the best initial values are
            sys_fit = optimize.minimize(chisq_for_optimize,x0_sys,args=(data, centWav),method='Nelder-Mead') #optimized systematic parameters
            fit = Voigt(data[0], sys_fit.x[0], sys_fit.x[1], centWav, sys_fit.x[2])
            used = np.where(fit>np.max(fit)*min_voigt_peak)[0]
            StartPoint = np.where(WV == data[0][used[0]])[0] #to find the start point of the excluded data. The wavelength is the exact same in the full spectra as in the voigt profile
            shift = StartPoint-used[0] #used to find the true indeces of the data that will be ignore, indeces relative to whole spectra
            unUsed_idx = np.append(unUsed_idx, used+shift) 
            return unUsed_idx

        if len(dividers) == 0: #then there is only one peak, so no need to further divide data
            unUsed_idx = Get_unUsed_idx(unUsed_idx, Peaks)
        else:
            for d in range(len(dividers)+1): #+1, because n-1 dividers for the number of Peak groups
                if d == 0: #for the 1st iteration, the group of peaks starts from the beginning of the peak array 
                    P_idxs_i = Peaks[:dividers[d]+1] #and ends at the 1st divider (INCULDING the divider)
                elif d == len(dividers): #for the last iteration, the group of peaks starts from the previous divider 
                    P_idxs_i = Peaks[dividers[d-1]+1:] #and ends at the end of the Peaks array
                else: #for any other iteration, the group of peaks starts from the previous divider
                    P_idxs_i = Peaks[dividers[d-1]+1:dividers[d]+1] #and ends at the 1st divider (INCULDING the divider)
                unUsed_idx = Get_unUsed_idx(unUsed_idx, P_idxs_i)
        unUsed_idx = unUsed_idx.astype(int)
        return unUsed_idx #This is unused indeces in terms of the data segment, so will have to add another offset term
    ##############

    if min(spec['waveobs']) > 1000: #if in Angstroms
        spec['waveobs'] = spec['waveobs']/10.0 #switch to nm
    if UseWavelengths:
        window = np.where((UseWavelengths[0]<spec['waveobs']) & (spec['waveobs']<UseWavelengths[1]))
    else:
        window = np.arange(spec['waveobs'].shape[0]) #if no wave range, then use all the data
    spec_wav, spec_flux, spec_err, SPECwindow = spec['waveobs'][window], spec['flux'][window], spec['err'][window], spec[window]
    ALLclipIDX = np.array([])
    #step 1) group the spectra to sections of 'Wav_window' [nm]
    NumWindow = np.ceil((spec_wav[-1]-spec_wav[0])/Wav_window)
    start_w, end_w = min(spec_wav), min(spec_wav)+Wav_window
    StartIdx = 0
    for W in tqdm(range(int(NumWindow-1))): #for every case but the 1st and 2nd
        spec_range_i = np.where((start_w<=spec_wav) & (spec_wav<end_w))[0]
        clip_idx = sigclip(SPECwindow[spec_range_i]).astype(int)
        # plt.figure(W)
        # plt.plot(SPECwindow[spec_range_i]['waveobs'], SPECwindow[spec_range_i]['flux'], '-')
        # plt.plot(SPECwindow[spec_range_i]['waveobs'][clip_idx], SPECwindow[spec_range_i]['flux'][clip_idx], 'r.')
        # print ("StartIdx:", StartIdx)
        # print ("clip_idx:", clip_idx)
        ALLclipIDX = np.append(ALLclipIDX, clip_idx+StartIdx) 
        start_w = end_w
        end_w += Wav_window
        StartIdx += len(spec_wav[spec_range_i]) #for the next iteration add the length of the previously used spectra, to keep track of indicies that must be add to get global indecie positions
    #for the last case. Because last bin might be short than the rest, have the last bin be len('Wav_window')
    last_bnd = spec_wav[-1]-Wav_window #regardless of if it might overlap with the previous bin
    spec_range_i = np.where(spec_wav>=last_bnd)[0]
    StartIdx = len(spec_wav) - len(spec_wav[spec_range_i])
    clip_idx = sigclip(SPECwindow[spec_range_i]) #for the last case
    ALLclipIDX = np.append(ALLclipIDX, clip_idx+StartIdx).astype(int) 
    # print ("StartIdx:", StartIdx)
    # print ("clip_idx:", clip_idx) 
    # print ("ALLclipIDX:", ALLclipIDX)
    print ("Cut", str(len(ALLclipIDX))+'/'+str(len(spec_wav)), "outlier points")
    CUTwav, CUTflux, CUTerr = np.delete(spec_wav, ALLclipIDX), np.delete(spec_flux, ALLclipIDX), np.delete(spec_err, ALLclipIDX)
    if Save: #if on, then save data 
        if type(Save) == str: #if path and file name is specified save the data with that info
            SaveWavnFlux(Save, CUTwav, CUTflux, Errors=CUTerr)
        else:
            SaveWavnFlux('clipped_spectra.fits', CUTwav, CUTflux, Errors=CUTerr)
    return ALLclipIDX


def CorrRV_nCutoutliers(targ, rv_lims=None,insturment='FEROS', SigClips1=3, SigClips2=5, Initialprefix='.fits', RVprefix='_clippedFULL.fits', CUT=False, outfolderBase='', Print=True, Plt_RV=True): 
    """To get the FEROS or HARPS data from its raw form in the ESO archive to something that can be used to calculate Ca II H & K indeces
    This is done by 1) sigma clipping the original data before calc RV shift (do a 3 sig clip there because there will be negatives that make it harder to clip when finding the mean) - FEROS ONLY
    2a) Get an estimate of the RV shift with a broad rv shift scan from -400 to 400km. 2b) Calculate fine tuned RV shift based on 2a or from rv_lims given
    and lastly, 3) cut the data down to just wavelengths needed for the CalH&K regions and
    clipping that data again (do a 5 sigma clip then because already done heavy clipping so want to make sure it's outlier data that's being clipped then) - FEROS ONLY
    INPUT:
        outfolderBase = any additional path information to get to all of the spectra data. DON'T FORGET BACKSLASH IF NEEDED!!!
        rv_lims=upper and lower bounds (in km) for fine tune RV scan, skip step 2a if given. if rv_lims == None, then assuming skipping this step entirely. if rv_lims == 'unknown', then do step 2a)
    """
    notprefix = ['_RVcorr.fits', '_clippedFULL.fits', '_clippedFULL_RVcorr_clippedWavRng385_406.fits', 'Mean']#, '_clippedFULL'] #list of values Initialprefix names that should not be included in viable data
    outfolder = outfolderBase+targ+'/'+insturment

    ##### Pull out the datafiles
    def getDataFiles(prefix, notprefix, cnt=''):
        AllDataFiles, DataFiles = glob.glob(outfolder+'/ADP*'+prefix), []
        for D in AllDataFiles: #to only include files that DON'T have the specific string in notprefi 
            Keep = True
            for notP in notprefix:
                if notP in D:
                    Keep = False
            if Keep:
                DataFiles.append(D)
        if Print:
            print ("outfolder+'/ADP*'+Initialprefix:", outfolder+'/ADP*'+prefix)
            print ("DataFiles"+cnt+":", DataFiles)
        return DataFiles
    DataFiles1 = getDataFiles(Initialprefix, notprefix, cnt='1')
    
    if insturment == 'FEROS':
    ##### ONLY NEED TO DO THIS ONCE EVER PER DATASET!!! If on cut data to only include wavelengths from 377 to 920, because everything else doesn't make sense
        if CUT: 
            for D in DataFiles1: #To get rid of the start wavelength of the FEROS data because often that data isn't reduced properly. Also not needed for calculating the velocity
                Cut(D, [377,920], Replace=True)
        
        def ClipSpec(DataFiles, X_range, SigClips): #a subfunction to clip the FEROS data since I run this twice
            for d in range(len(DataFiles)):
                spec = ispec.read_spectrum(DataFiles[d])
                if min(spec['waveobs']) > 1000: #if in Angstroms
                    spec['waveobs'] = spec['waveobs']/10.0 #switch to nm
                file = DataFiles[d].split('/')[-1]
                file_name_with_path = DataFiles[d][:-len('.fits')]
                if not X_range:
                    spec_range = np.arange(len(spec['waveobs']))
                    WavRngStr = 'FULL'
                else:
                    spec_range = np.where((X_range[0]<spec['waveobs']) & (spec['waveobs']<X_range[1]))[0]
                    WavRngStr = 'WavRng'+str(X_range[0])+'_'+str(X_range[1])
                clipped = ClearFEROS_outliers(spec, Save=file_name_with_path+'_clipped'+WavRngStr+'.fits', UseWavelengths=X_range, Sig_clips=SigClips)
            return clipped
        
        ##### The initial clip
        if SigClips1:
            Xrange1 = None #wavelength range to use in nm and associated name
            clipped1 = ClipSpec(DataFiles1, Xrange1, SigClips1)

    ##### Calculate the RV shift of the data
    if rv_lims:
        DataFiles2 = getDataFiles(RVprefix, ['_RVcorr.fits', '_clippedFULL_RVcorr_clippedWavRng385_406.fits'], cnt='2')
        print ("DataFiles2:", DataFiles2)
        #a) To get an estimate of the RV shift of the data
        extrass = None
        if RVprefix != '.fits' and RVprefix != '.txt': #only have extras to be removed if the prefix is more than the file type
            extrass =  RVprefix[:-len('.fits')]
        if type(rv_lims) == str:
            if rv_lims.lower() == 'unknown':
                BaryVs, starRVs, RVerrs, ReadSpectra = ReduceData(targ, DataFiles2, RA_Dec=None, DataType=insturment, RV_Corr=True, RVlims=[-400,400], RVstep=.5, RV_Mask="input/linelists/CCF/HARPS_SOPHIE.G2.375_679nm/mask.lst",\
                    SaveRVdat=False, FitContinuum=False, EstErr=False, Folder=outfolder, toPlot=False, Suffix=extrass)#RA_Dec=None, because HARPS and FEROS reduced data is already corrected for barycentric velocity
            else:
                sys.exit("ERROR!!! The only string value rv_lims accepts is 'unknown'!!!")
            rv_lims = [np.mean(starRVs)-20, np.mean(starRVs)+20]
        #b) To do the fine tune RV search of the data
        if Print:
            print ("rv_lims:", rv_lims)
        ReduceData(targ, DataFiles2, RA_Dec=None, DataType=insturment, RV_Corr=True, RVlims=rv_lims, RVstep=0.1, RV_Mask="input/linelists/CCF/HARPS_SOPHIE.G2.375_679nm/mask.lst",\
            SaveRVdat=True, FitContinuum=False, EstErr=True, Folder=outfolder, toPlot=Plt_RV, Suffix=extrass)#RA_Dec=None, because HARPS reduced data is already corrected for barycentric velocity
    
    ##### Calculate final sigma clip
    if insturment == 'FEROS' and SigClips2:
        Xrange2, DataFiles3 = (385,406), glob.glob(outfolder+"/ADP*_clippedFULL_RVcorr.fits")
        if Print:
            print ("DataFiles3:", DataFiles3)
        clipped2 = ClipSpec(DataFiles3, Xrange2, SigClips2)
    return None

if __name__ == '__main__':
    ######### To run the WHOLE analysis steps to get the data ready to calc R'_{Hk} indeces
    # Target = 'HATS29'
    # CorrRV_nCutoutliers(Target, rv_lims=[-40,0], insturment='HARPS', RVprefix='.fits')
    # print ("\n\n")
    # CorrRV_nCutoutliers(Target, rv_lims=[-40,0], insturment='FEROS')
    # print ("\n\n\n\n")

    # Target = 'WASP6'
    # CorrRV_nCutoutliers(Target, rv_lims=[11.49-20, 11.49+20], insturment='HARPS', RVprefix='.fits')
    # print ("\n\n\n\n")

    Target = 'WASP25'
    CorrRV_nCutoutliers(Target, rv_lims=[-2.62-20, -2.62+20], insturment='HARPS', RVprefix='.fits')
    print ("\n\n\n\n")

    Target = 'WASP55'
    CorrRV_nCutoutliers(Target, rv_lims=[-4.31-20, -4.31+20], insturment='FEROS')
    print ("\n\n")
    CorrRV_nCutoutliers(Target, rv_lims=[-4.31-20, -4.31+20], insturment='HARPS', RVprefix='.fits')
    print ("\n\n\n\n")

    Target = 'WASP96'
    CorrRV_nCutoutliers(Target, rv_lims=[-21.19, 18.81], insturment='FEROS')
    print ("\n\n\n")

    Target = 'WASP110'
    CorrRV_nCutoutliers('WASP110', rv_lims=[-34.66-20, -34.66+20], insturment='FEROS')
    print ("\n\n\n\n")

    Target = 'WASP124'
    CorrRV_nCutoutliers('WASP124', rv_lims=[-5.94-20, -5.94+20], insturment='FEROS')
    print ("\n\n\n\n")
    # CorrRV_nCutoutliers('HD23249', rv_lims='unknown', insturment='HARPS', RVprefix='.fits')
    #########

    # ########## To plot spectra (preCut and/or post cut):
    #Info about target
    #HATS29: RVlims=[-40,0], WASP6:RVlims=[-9,31], WASP25:RVlims=[-22,18], WASP55:RVlims=[-24,16]
    # targ, insturment, prefix = 'HD16160', 'FEROS', '.fits' #"_clippedFULL_RVcorr.fits"
    # BASE = '/Users/chimamcgruder/SSHFS/CM_GradWrk/TwinPlanets/HarpsCal_Lovis2011/'
    # TarGet, insturment, prefix = 'HD23249', 'FEROS', '_clippedWavRng385_406.fits'
    # notprefix = ['_clippedFULL.fits']#, '_clippedFULL'] #list of values prefix names that should not be included in viable data
    # outfolder = BASE+TarGet+'/'+insturment
    # Colors = ['r', 'b', 'orange', 'green', 'm', 'purple', 'c', 'grey']
    # AllDataFiles, DataFiles1 = glob.glob(outfolder+"/ADP*"+prefix), []
    # for D in AllDataFiles: #to only include files that DON'T have the specific string in notprefi 
    #     Keep = True
    #     for notP in notprefix:
    #         if notP in D:
    #             Keep = False
    #     if Keep:
    #         DataFiles1.append(D)
    # print ("DataFiles1:", DataFiles1)

    # # # #####
    # # # #To get rid of the start wavelength of the FEROS data because often that data isn't reduced properly. Also not needed for calculating the velocity, but let's do it beforehand just in case (since the RV wavelength shift is minimal)
    # # # for D in DataFiles1: #Only have to do this once
    # # #     Cut(D, [377,920], Replace=True) 
    # # # #####

    # ##### Also to cut the spec data. Do this twice, once for the original data before calc RV shift (do a 3 sig clip there because there will be negatives that make it harder to clip when finding the mean) 
    # #And again after RV shift is calculateded, just for the CalH&K regions (do a 5 sigma clip then because already done heavy clipping so want to make sure it's outlier data that's being clipped then)
    # CLIP, MaxY = False, -1 
    # Xrange = None #(385, 406) #wavelength range to plot in nm
    # for d in range(len(DataFiles1)):
    #     spec = ispec.read_spectrum(DataFiles1[d])
    #     if min(spec['waveobs']) > 1000: #if in Angstroms
    #         spec['waveobs'] = spec['waveobs']/10.0 #switch to nm
    #     file = DataFiles1[d].split('/')[-1]
    #     file_name = file[:-len(prefix)]
    #     file_name_with_path = DataFiles1[d][:-len('.fits')]
    #     if not Xrange:
    #         spec_range = np.arange(len(spec['waveobs']))
    #         WavRngStr = 'FULL'
    #     else:
    #         spec_range = np.where((Xrange[0]<spec['waveobs']) & (spec['waveobs']<Xrange[1]))[0]
    #         WavRngStr = 'WavRng'+str(Xrange[0])+'_'+str(Xrange[1])
    #     if np.max(spec['flux'][spec_range]) > MaxY: #to get the plotting range
    #         MaxY = np.max(spec['flux'][spec_range])
    #     spec_plot = spec[spec_range]
    #     plt.figure(TarGet)
    #     plt.legend()
    #     plt.plot(spec_plot['waveobs'], spec_plot['flux'], '.', color=Colors[d], label=file_name, markersize=1, zorder=0)
    #     if CLIP:
    #         cut = ClearFEROS_outliers(spec, Save=file_name_with_path+'_clipped'+WavRngStr+'.fits', UseWavelengths=Xrange, Sig_clips=5)
    #         spec_cut = np.delete(spec_plot, cut)
    #         plt.plot(spec_cut['waveobs'], spec_cut['flux'], '.', color=Colors[d], markersize=4, zorder=2)
    #         plt.figure(TarGet+'_onlyCLIPed')
    #         plt.plot(spec_cut['waveobs'], spec_cut['flux'], '.', color=Colors[d], markersize=4, zorder=2, label=file_name)
    # plt.legend()
    # plt.show()
    # plt.close()
    # # ################

    # #A) To get an estimate of the RV shift of the data
    # prefix, notprefix = '_clippedFULL.fits', []#"_clippedFULL_RVcorr.fits"#'_RVcorr.fits' #prefix needs to also inlude file type #notprefix = list of values prefix names that should not be included in viable data
    # outfolder = TarGet+'/'+insturment
    # AllDataFiles, DataFiles1 = glob.glob(outfolder+"/ADP*"+prefix), []
    # for D in AllDataFiles: #to only include files that DON'T have the specific string in notprefi 
    #     Keep = True
    #     for notP in notprefix:
    #         if notP in D:
    #             Keep = False
    #     if Keep:
    #         DataFiles1.append(D)
    # print ("DataFiles1:", DataFiles1)
    # prefix = "_clippedFULL"
    # # ReduceData(TarGet, DataFiles1, RA_Dec=None, DataType=insturment, RV_Corr=True, RVlims=[-400,400], RVstep=.5, RV_Mask="input/linelists/CCF/HARPS_SOPHIE.G2.375_679nm/mask.lst",\
    # #     SaveRVdat=False, FitContinuum=False, EstErr=False, Folder=outfolder, toPlot=False, Suffix=prefix)#RA_Dec=None, because HARPS and FEROS reduced data is already corrected for barycentric velocity

    # #1) Calculate the RV shift of the data
    # ReduceData(TarGet, DataFiles1, RA_Dec=None, DataType=insturment, RV_Corr=True, RVlims=[-26,14], RVstep=0.1, RV_Mask="input/linelists/CCF/HARPS_SOPHIE.G2.375_679nm/mask.lst",\
    #     SaveRVdat=True, FitContinuum=False, EstErr=True, Folder=outfolder, toPlot=True, Suffix=prefix)#RA_Dec=None, because HARPS and FEROS reduced data is already corrected for barycentric velocity
    
    # # 1b) Degrade resolution. I don't think this is necessary
    # DataFiles1b = glob.glob(outfolder+"/ADP*_RVcorr.txt")
    # CurrtRes, NewRes = 115000, 65000
    # for D in DataFiles1b:
    #     degrade_resolution(D, from_res=115000, to_res=65000, Save=True)

    """                                               !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!                                      """
    """!!! after step 1 and 1b I calculate the R'_hk values from 'Research/GitHub/HARPSN_activity/ACT_indexesCHIMA.py' !!!"""
    """                                               !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!                                      """

    # #2)  Combine all RV corrected spectra into one, for better stellar parameter determination
    # DataFiles2 = glob.glob(outfolder+"/ADP*_RVcorr.txt")
    # DataFiles2 = glob.glob("/Users/chimamcgruder/SSHFS/CM_GradWrk/TwinPlanets/HATS29/HARPS/ADP.2015-04-15T08_03_56.403_RVcorr.txt")
    # DataFiles2 = glob.glob(outfolder+"/ADP*_RVcorrRes"+str(NewRes)+".txt") #if want to use lower res data
    # FolderTargName = [outfolder, targ]
    # CombineSpectra(DataFiles2, PLOT=True, Normalize=False, ZerEdg=True, Method='sum', FolderTargName=FolderTargName, Base='_RVcorr', plot_resampled=True, ErrEstPoint=10)

    # #3) Get the continuum Normalized combined spectrum
    # DataFiles3 = [outfolder+'/HATS29_sum_RVcorr.txt'] #needs to be in list form
    # CutNormSpectra, ReadSpectra = ReduceData(targ, DataFiles3, Folder=outfolder, DataType=insturment, RV_Corr=False, FitContinuum=True, ContFit_Model="Polynomy", toPlot=True, Lines=targ+'_StrongSegments.txt', CutSpec=[300,430], Deg=2) #ContFit_Model= "Splines" or "Polynomy"
    
    # #4) Run Corrected/contiuum normalized master spectrum through parameter estimated code
    # Add  = '_Res65000'  
    # CutNormSpectra = ispec.read_spectrum(outfolder+"/"+targ+"_sum_RVcorr_ContinuumNormCut.txt")
    # T, G, M, V = 5617, 4.385, 0.15, 2.54
    # Parameters_using_grid(CutNormSpectra, T, G, M, V, init_Alpha='fixed', init_LD_coeff=0.6, Resolution=115000, init_Vrad=0,max_iterations=50, FreeParms = ["teff", "logg", "MH", "limb_darkening_coeff", "vsini"],\
    #     GridDir = "/input/grid/SPECTRUM_MARCS.GES_GESv6_atom_hfs_iso.480_680nm/", SaveDat=[outfolder, targ+'_grid', "summed"])#, SaveDat = ['WASP6/MIKE_preReduced/', "WASP6", "28.8.2018"]) #SaveDat = [Folder, Target, '_'+X+Add]
    
    # Parameters_using_SynthSpectra(CutNormSpectra, T, G, M, V, initial_alpha='fixed', initial_limb_darkening_coeff=0.6, initial_vrad=0, Resolution=115000, max_iterations=50, code="spectrum", SaveDat=[outfolder, targ+'_synth', "summed"],\
    #     AtomicLineList = "GESv5_atom_hfs_iso.420_920nm/atomic_lines.tsv", ModelAtmo='MARCS.GES/', SolarAbundances='Grevesse.2007/stdatom.dat', free_params = ["teff", "logg", "MH", "limb_darkening_coeff", 'vsini'])

    # CutSpectra = ispec.read_spectrum(outfolder+"/HATS29_sum_RVcorr_Cut.txt")
    # Parameters_from_EW([CutNormSpectra,CutSpectra], T, G, M, Resolution = 115000, max_iterations=40, code="moog", AtomicLineList = "GESv5_atom_hfs_iso.420_920nm/atomic_lines.tsv", \
    #     use_ares=False, ModelAtmo='MARCS.GES/', SolarAbundances='Grevesse.2007/stdatom.dat', SaveDat=[outfolder, targ+'_ew', "summed"])

    ################
    # DataFiles1 = outfolder+'/ADP.2017-01-05T09_06_13.286.fits'
    # spec = ispec.read_spectrum(DataFiles1) 
    # plt.plot(spec['waveobs']/10, spec['flux'], 'r.-') #to plot data, before the 'cosmic rays' are removed
    # cos_rmed = ClearFEROS_outliers(spec, Sig_clips=3,Wav_window=5)
    # plt.plot(cos_rmed['waveobs'], cos_rmed['flux'], 'b.-') #To plot data after the removal

    # plt.figure('HD10700')
    # file = 'ADP.2021-11-09T07_36_48.872.fits'#'HD152391/ADP.2016-09-20T08_08_13.022_CUT.fits'
    # spec = ispec.read_spectrum(file)
    # plt.plot(spec['waveobs'], spec['flux'], 'g-.')
    # plt.ylabel('FLUX (ADU?)')
    # plt.xlabel('wavelength (nm)')
    # plt.savefig('HD10700.png')

    # plt.figure('WASP55')
    # file = 'WASP55/FEROS/ADP.2016-09-27T09_50_42.605.fits'
    # spec = ispec.read_spectrum(file)
    # plt.plot(spec['waveobs']/10, spec['flux'], 'r-.')
    # plt.ylabel('FLUX (ADU?)')
    # plt.xlabel('wavelength (nm)')
    # plt.savefig('WASP55.png')

    # plt.ylim((-.005,MaxY+.005))
    # plt.xlim(Xrange)
    # plt.ylabel('FLUX (ADU?)')
    # plt.xlabel('wavelength (nm)')
    # plt.legend()
    # plt.show()
    # plt.close()
