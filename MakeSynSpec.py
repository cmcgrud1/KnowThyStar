# Code to produce synthetic spectra. Convolve iSpec synthetic spectra (takes into account line list and stellar parameters), blackbody curve, and HST/STIS throughput
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['PYSYN_CDBS']
from scipy.interpolate import interp1d
from scipy import signal
import sys
import pysynphot as S
c = 299792458 #speed of light [m/s]

# #--- iSpec directory -------------------------------------------------------------
# ispec_dir = '/Users/chimamcgruder/Research/TwinPlanets/iSpec_v20190302'
# sys.path.insert(0, ispec_dir)
# import ispec

def CalSurfG(Mass, Radius): #quoted values are in cgs units
    G, M_sun, R_sun = 6.6743e-8,1.989e33, 6.9634e10 #cm3 g-1 s-2, g, cm
    M, R = Mass*M_sun, Radius*R_sun #to convert from solar units to cgs units
    return np.log10((G*M)/(R**2))

def PlotHARPSThroughput():
  import pandas
  read_csv()
  
# To calculate black body curve of star of given temperature
def PlankSpecIrrad(wavs, T, ext_fct = 0): #T in Kelvin, wavs in nm
    h,k = 6.6260701e-34, 1.38064852e-23 # UNITS = SI: h=J*s, k=J/K
    wavs = wavs*1e-9 # to convert from nm to meters
    Ext = S.Extinction(ext_fct, 'lmcavg')
    Ext_wav = (1/Ext.wave)*1e-6 # units of ext are 1/um for some reason. Converts to meters
    f = interp1d(Ext_wav, Ext.throughput) #1e-10 to convert from ang to m
    throughput = f(wavs)
    A, expC = (2*h*(c**2))/(wavs**5), (h*c)/(wavs*k*T)
    B = 1.0/(np.exp(expC)-1)
    return A*B*throughput #units: spectral radiance (W sr-1 m-3)

#To take data with a given wavelength range and interpolate it to the same wavelength range of data
def Iterpolate(final_wave, init_wave, init_through): #assuming init_wave given in nm
  f = interp1d(init_wave, init_through)
  return f(final_wave)

#To shift wavelength and spectra based on radial/barycentric velocity. if v < 0 spectra will shift to left. if v > 0 spectra will shift to right. Assuming v in km/s
def RadVeShift(Wave, spect, vel):
  Wav_shift = Wave*np.sqrt((1+(vel/(c*1e-3)))/(1-(vel/(c*1e-3)))) #c*1e-3 to convert m/s to km/s
  LowOvrflw, HigOvrflw = np.where(Wav_shift < Wave[0])[0], np.where(Wav_shift > Wave[-1])[0] #spectral elements that run out of the orignal wavelength range of Wave[0]-Wave[-1]
  #To cut off data that runs out of the spectra range after the wavelength shift
  if len(LowOvrflw) > 0:
    finWav, finSpec = Wav_shift[LowOvrflw[-1]+1:], spect[LowOvrflw[-1]+1:]
  if len(HigOvrflw) > 0:
    finWav, finSpec = Wav_shift[:HigOvrflw[0]+1], spect[:HigOvrflw[0]+1]
  return finWav, finSpec

#To bin spectrum 
def BinSpec(flux, wavs, WavRes=None, R = None):
  #WavRes = Wavelength step size of data
  #R = spectral resolution = lambda/DeltaLambda
  print('Current mean WavRes =', np.mean(np.diff(wavs)), 'and ~R =', np.mean(wavs)/np.mean(np.diff(wavs))) # R = lambda/DeltaLambda
  if R: # if given in sepctral res, need to convert to WavRes
    WavRes = np.mean(wavs)/R 
  bin_num = int(np.round((wavs[-1]-wavs[0])/WavRes))
  bin_width = int(np.round(len(wavs)/bin_num)) #number of original wav elements in new binning scheme
  prev_binstrt = 0
  bin_mean, bin_wavs = np.zeros(bin_num), np.zeros(bin_num)
  for i in range(bin_num):
    if i == bin_num-1: #on last bin
      bin_mean[i], bin_wavs[i] = np.mean(flux[prev_binstrt:]), np.mean(wavs[prev_binstrt:])
    else:
      bin_mean[i], bin_wavs[i] = np.mean(flux[prev_binstrt:bin_width+prev_binstrt]), np.mean(wavs[prev_binstrt:bin_width+prev_binstrt])
    prev_binstrt = bin_width+prev_binstrt
  if math.isnan(np.mean(bin_wavs)): # because the initial data is made of finate bins, some change in bins can't be done properly and must be rounded. Those rounded bins could cause nans
    NaNs = np.argwhere(np.isnan(bin_wavs))
    bin_mean, bin_wavs = np.delete(bin_mean, NaNs), np.delete(bin_wavs, NaNs)
  print ('Binned wavelength Res =', np.mean(np.diff(bin_wavs)), 'and ~R =', np.mean(wavs)/np.mean(np.diff(bin_wavs)))
  return bin_mean, bin_wavs

#To find value in array closest to given value
find_nearest = lambda array, value: (np.abs(array - value)).argmin()

def SynthCosmicRays(Wave, Spec, Rays, meanH=1, sigH=.5, meanW=.12, sigW=.01, Print=True):
#Ray = number of rays you want to afflict spectrum 
#meanH, sigH = percentage higher you want the cosmic rays to be over the mean spectrum height and the spread of the height of each ray
# To produce the final spectra (save as .txt file) and plot the steps in creating final spectra, if wanted
  MeanSpec, Num = np.mean(Spec), 9 #anything past 7 just produced datapoints at the bottom of the gaussian. Have num of 9 for a little cushion
  # print ('MeanSpec', MeanSpec)
  Gauss = signal.gaussian(Num, std=1)
  CenWav = np.random.choice(Wave, size=Rays, replace=False) #to pick random locations for cosmic rays to hit
  if Print:
    print ('\033[1mIncluding cosmic rays.\033[0m')
  PRINT0, PRINT1 = 'Cetenterd at:', 'With peaks of:'
  for c in CenWav:
    height = np.random.normal(meanH, sigH)  #to determine hight of cosmic ray signal
    height = np.abs(height*MeanSpec)
    width = np.random.normal(meanW, sigW) #to determine width of cosmic ray signal
    gauss = Gauss*height #to scale based on spectra
    CosWav = np.linspace(c-width/2.0, c+width/2.0, num=Num-2)
    CosWav = np.append(c-width, CosWav) #Add these points for a little extra cushion for the interpolations. extra points are base of guassian
    CosWav = np.append(CosWav, c+width)
    NearBck, NearFnt = find_nearest(Wave, c-width/2.0), find_nearest(Wave, c+width/2.0)
    WavGrp = Wave[NearBck:NearFnt+1]
    PropGauss = Iterpolate(WavGrp, CosWav, gauss)
    Spec[NearBck:NearFnt+1] = PropGauss+Spec[NearBck:NearFnt+1]
    PRINT0, PRINT1 = PRINT0+str(c)+', ', PRINT1+str(height)+', '
  # plt.plot(Wave, Spec)
  if Print:
    print (PRINT0+'[nm]')
    print (PRINT1+'[counts]')
  return Spec, PRINT0, PRINT1

def SynthEmissionLines(Wave, Spec, Emissions, Height=1, Width=.12):
#Emissions = array of wavelength positions where the emission lines should peak
#meanH, sigH = percentage higher you want the cosmic rays to be over the mean spectrum height and the spread of the height of each ray
# To produce the final spectra (save as .txt file) and plot the steps in creating final spectra, if wanted
  MeanSpec, Num = np.mean(Spec), 9 #anything past 7 just produced datapoints at the bottom of the gaussian. Have num of 9 for a little cushion
  # print ('MeanSpec', MeanSpec)
  Gauss = signal.gaussian(Num, std=1)
  for center in Emissions:
    c = Wave[find_nearest(Wave, center)] #to find wavelength location of desired emission
    height = np.abs(Height*MeanSpec)
    gauss = Gauss*height #to scale based on spectra
    CosWav = np.linspace(c-Width/2.0, c+Width/2.0, num=Num-2)
    CosWav = np.append(c-Width, CosWav) #Add these points for a little extra cushion for the interpolations. extra points are base of guassian
    CosWav = np.append(CosWav, c+Width)
    NearBck, NearFnt = find_nearest(Wave, c-Width/2.0), find_nearest(Wave, c+Width/2.0)
    PropGauss = Iterpolate(Wave[NearBck:NearFnt+1], CosWav, gauss)
    Spec[NearBck:NearFnt+1] = PropGauss+Spec[NearBck:NearFnt+1]
  # plt.plot(Wave, Spec)
  return Spec

def Noise(Synth_spec, PercentNoise=5): #To make noise at a specific percentage level of signal. Def not a physical way to make noise. Just a quick & dirty method
  Noise = np.random.normal(1, (PercentNoise/100.0), len(Synth_spec))
  return Synth_spec*Noise

def SynSpec(LinesFile, Temp, ext_fct=0, bary_V=0, Rad_V=0, ThroughHead=None, TelFil=None, CosRays=0, Plot=False, OutFile='SynSpec.txt', EmissLines=[], SNR=0, FigStyle=None, Print=True): 
  #NoiseLvl = gaussian fit on percentage of noise level relative to each signal
  #Have to have the lines list and temperature for a blackbody curve. Everything else is optional
  ########### Required Code ########### 
  waveobs,flux_lines,err = np.loadtxt(LinesFile, unpack=True) #stellar lines with no errors
  B_blackBod = PlankSpecIrrad(waveobs, Temp, ext_fct = ext_fct) #black body light curve, including extinction
  Synth_spec = B_blackBod*flux_lines #unshifted spectra BEFORE reaching telescope
  FinalSpecStr = 'blackbody*iSpecLines'
  if Plot:
    if FigStyle is not None:
      style1, style2, style3, style4, style5, style6 = FigStyle[0], FigStyle[1], FigStyle[2], FigStyle[3], FigStyle[4], FigStyle[5]
    else:
      style1, style2, style3, style4, style5, style6 = 'm', 'r', 'c', 'g', 'k', 'b--'
    plt_cnts = 2 #To count the number of subplots I'll need. Default at least 3 (blackbody, lineslist, and convolved data)
    rw_cnt = 0
    if ThroughHead: #the convolved data is not reflected in 'plt_cnts' because it has it's own row
        plt_cnts += 1
    if TelFil: #Plotting the effect of RV shift and cosmic rays is too much
        plt_cnts += 1
    plt.subplot2grid((2, plt_cnts), (0, rw_cnt), rowspan=1, colspan = 1) #for blackbody spectra
    plt.title(str(Temp)+'K_blackbody')
    plt.plot(waveobs,B_blackBod, style1)
    y = [] #to exclude ticks on the y axis
    plt.yticks(y, " ")
    plt.ylabel('spectral radiance (W sr-1 m-3)')
    plt.xlabel('Wavelength [nm]')
    rw_cnt += 1  
    plt.subplot2grid((2, plt_cnts), (0, rw_cnt), rowspan=1, colspan = 1) #for continuum normalized line list
    if '/' in LinesFile:
        SPLIT = LinesFile.split('/')[-1]
    else:
    	SPLIT = LinesFile
    SPLIT = SPLIT.split('.txt')[0] #to get the base name of spectral lines file. #Assuming the linelist is saved as .txt file
    plt.title('iSpeclines\n'+SPLIT)
    plt.plot(waveobs,flux_lines, style2)
    y = [] #to exclude ticks on the y axis
    plt.yticks(y, " ")
    plt.ylabel('Continuum Normalized counts')
    plt.xlabel('Wavelength [nm]')
    rw_cnt += 1
  ########### Optional Routines ########### 
  if bary_V != 0 or Rad_V != 0: #rad velocity due to earth and rad velocity of star [km/s]
    waveobs, Synth_spec = RadVeShift(waveobs, Synth_spec, bary_V+Rad_V) #apply barycentric and radial velocity shifts
    Synth_spec_err = np.zeros(len(Synth_spec)) #Need to add errors after doing the radial velocity correction
    FinalSpecStr += '(+RVshift)'
  else:
    Synth_spec_err = np.zeros(len(Synth_spec)) #If no RV correction still need to add errors
    
  if CosRays > 0:
    Synth_spec, CRlocal, CRcount = SynthCosmicRays(waveobs, Synth_spec, CosRays, Print=Print)
    FinalSpecStr += '(+'+str(CosRays)+'CosRays)'
  
  if len(EmissLines) > 0: #Just a test to make cosmic ray like event, to ensure my cosmic ray detector doesn't erase em
    Synth_spec = SynthEmissionLines(waveobs, Synth_spec, EmissLines)

  if ThroughHead: 
    wave, throughput = np.loadtxt(ThroughHead, unpack=True) #Assuming data is in wavelength [nm]x throughput [efficency]
    Throughput = Iterpolate(waveobs, wave, throughput) #to convert to same wavelength scale as above data
    Synth_spec = Synth_spec*Throughput #convole spectra with HST throughput
    if Plot:
        FinalSpecStr += '*Throughput'
        if '/' in ThroughHead:
        	SPLIT = ThroughHead.split('/')[-1]
        else:
        	SPLIT = ThroughHead
        SPLIT = SPLIT.split('.')[0] #to get the base name of spectral lines file
        plt.subplot2grid((2, plt_cnts), (0, rw_cnt), rowspan=1, colspan = 1) #normalized throughput
        plt.title('Throughput\n'+SPLIT)
        plt.plot(waveobs,Throughput, style3)
        y = [] #to exclude ticks on the y axis
        plt.yticks(y, " ")
        plt.ylabel('Efficency')
        plt.xlabel('Wavelength [nm]')
        rw_cnt += 1

  if TelFil:
    FinalSpecStr += '*Tellurics'
    tell_wave, tell_flux, tell_err = np.loadtxt(TelFil, unpack=True) #tellurics
    extend_wave, ext_tel_flux = np.concatenate((np.linspace(300,350,100), tell_wave)), np.concatenate((np.ones(100), tell_flux)) #Assuming no tellurics from 300-350nm. Might not be true, but just a test so who cares
    TelFlux = Iterpolate(waveobs, extend_wave, ext_tel_flux)
    Synth_spec = Synth_spec*TelFlux #Know it doens't make sense to have tellurics with HST data, but just want synthetic data as test case 
    if Plot:
        plt.subplot2grid((2, plt_cnts), (0, rw_cnt), rowspan=1, colspan=1) #telluric lines
        SPLIT = TelFil.split('=')[-1] #if I named file right, there will always be '=' for defining resolution
        SPLIT = SPLIT.split('.')[0] #to get Resolution of telluric file
        plt.title('iSpecTellurics_R='+SPLIT)
        plt.plot(waveobs,TelFlux, style4)
        plt.ylabel('attinuation')
        plt.xlabel('Wavelength [nm]')
        rw_cnt += 1
  
  if SNR > 0: #Add noise to the spectrum fluxes to simulate a given SNR (Signal to noise Ratio).
     CleanSynthSpec = np.copy(Synth_spec)     
     sigma = Synth_spec/SNR #N/sqrt(N) = sigma = snr = sqrt(N), there4 to include snr: counts/snr = sigma
     sigma[sigma <= 0.0] = 1.0e-10 #filter out the zeros and replace em with small numbers
     Synth_spec += np.random.normal(0, sigma, len(Synth_spec))
     Synth_spec_err += sigma 

  if Plot:
      if bary_V != 0 or Rad_V != 0:
        plt.title('Final Spectra: barycentric V='+str(bary_V)+', Radial V='+str(Rad_V))
      plt.subplot2grid((2, 4), (1, 0), rowspan=1, colspan = 4) #for data convolving all spectral components
      plt.title('Final Spectra')
      plt.plot(waveobs,CleanSynthSpec, style5, label = 'Clean') #/np.mean(Synth_spec)
      if Noise:
        plt.plot(waveobs,Synth_spec, style6, label = 'Noisy_SNR='+str(SNR)) #/np.mean(Synth_spec)
      y = [] #to exclude ticks on the y axis
      plt.yticks(y, " ")
      plt.xlim([290,585]) #to cut off points with 0 throughput
      plt.ylabel('mean Normalized flux')
      plt.xlabel('Wavelength [nm]')
      plt.legend()
 
  if type(OutFile) == str: #only save file if OutFile is specified
    OutFileBase = OutFile.split('.txt')[0] #Assuming the only file type is '.txt'
    f0 = open(OutFileBase+'_Robust.txt', 'w') #For version of file that includes 0s and locations & amplitudes of cosmic rays
    f1 = open(OutFile, 'w')
    f1.write('# waveobs   flux   err\n'), f0.write('# waveobs   flux   err\n')
    f0.write('#'+CRlocal+'\n')
    f0.write('#'+CRcount+'\n')
    for it in range(len(waveobs)):
        f0.write(str(waveobs[it])+'   '+str(Synth_spec[it])+'   '+str(Synth_spec_err[it])+'\n')
        if Synth_spec_err[it] > 1e-10: #only save non-0 datapoints
            f1.write(str(waveobs[it])+'   '+str(Synth_spec[it])+'   '+str(Synth_spec_err[it])+'\n')
    f0.close(), f1.close()
    if '/' in OutFileBase:# assuming each slash is a dir path
        SPLITs = OutFileBase.split('/')
        OutFileName, Dir = SPLITs[-1], ''
        for S in range(len(SPLITs)-1):
            Dir += SPLITs[S]+'/'
    else:
    	Dir, OutFileName = os.getcwd(), OutFileBase
    if Print:
      print ('Saved final synthetic spectra in "'+Dir+'" as "'+OutFileName+'_Robust.txt" for the Robust file and "'+OutFileName+'.txt" for the base data')
    return waveobs, Synth_spec

def SaveHST_throughput(ThroughHead, FileName = None): 
  bp = S.ObsBandpass(ThroughHead) #source on using this https://pysynphot.readthedocs.io/en/latest/appendixb.html
  Throughput, waveobs = bp.throughput, bp.wave/10.0 #1/10 to convert from angstroms to nm.
  #bp.throughput is the fractional amount of incident light that the detector reads 
  if FileName is not None:
    pass
  else:
    SPLIT = ThroughHead.split(',')
    FileName = SPLIT[0]
    for s in range(len(SPLIT)-1):
      FileName += '_'+SPLIT[s+1]
    FileName += '.txt'
  f = open(FileName, 'w')
  f.write('# waveobs   throughput[fraction]\n')
  for w in range(len(waveobs)):
      f.write(str(waveobs[w])+'   '+str(Throughput[w])+'\n') 
  f.close()
  # print ('Throughput', Throughput)
  # print ('os.getcwd()', os.getcwd())
  # print ('FileName', FileName)
  print ('Saved HST throughput of file in "'+os.getcwd()+'" as "'+FileName+'"') 

if __name__ == "__main__":
  # #To import the continumm normalized spectra produced by iSpec
  # SaveHST_throughput('stis,ccd,g430l', FileName = None)
  HARPSbase= '../TwinPlanets/SynSpec_moog_5670.0_4.39_0.16_0.0_1.02_BaryV-18.26RV205_CosRay6_SNR15_EmiLines2_ExtFct.1'
  SynLines, HARPSthro, TelTemp = HARPSbase+'/moog_5670.0_4.39_0.16_0.0_1.02.txt','../TwinPlanets/HARPS_efficiency.txt', '../TwinPlanets/Tellurics_R=115k.txt'
  STIS_Thro = 'stis_ccd_g430l.txt'
  # # SynSpec(SynLines, 4875, ext_fct=.1, ThroughHead=STISthro, TelFil=TelTemp, Plot=True, OutFile=None, NoiseLvl = 10)
  # waveobs1, Synth_spec1 = SynSpec(SynLines, 4875, ext_fct=.1, ThroughHead=STISthro, TelFil=TelTemp, CosRays = 8, Plot=False, NoiseLvl = 10, OutFile = '../TwinPlanets/NosySynSpec1noRV.txt', EmissLines=[320, 500])
  # plt.plot(waveobs1, Synth_spec1, '-')
  # waveobs2, Synth_spec2 = SynSpec(SynLines, 4875, ext_fct=.1, ThroughHead=STISthro, TelFil=TelTemp, CosRays = 8, Plot=False, NoiseLvl = 10, OutFile = '../TwinPlanets/NosySynSpec2noRV.txt', EmissLines=[320, 500])
  # plt.plot(waveobs2, Synth_spec2, '--')
  # waveobs3, Synth_spec3 = SynSpec(SynLines, 5670, ext_fct=.1, bary_V=-18.26, Rad_V=205, ThroughHead=HARPSthro, TelFil=TelTemp, CosRays = 6, Plot=True, SNR = 15, EmissLines=[320, 500], OutFile = HARPSbase+'/SynSpec.txt')
  SynSpec(SynLines, 5670, ext_fct=.1, bary_V=0, Rad_V=0, ThroughHead=STIS_Thro, CosRays = 6, Plot=True, SNR = 15, EmissLines=[320, 500], OutFile = None)
  # plt.plot(waveobs3, Synth_spec3, '.')
  plt.show()
  plt.close()
  # print (CalSurfG(Mass = 1.032, Radius = 1.073))
  
  ####====================================To make X amount of synthetic spectra for testing different parameter estimation techniques========(takes about ~15hrs. Run on Hydra!)============================####
  #500 for Synth will take ~5days if each run time is ~15min
  #5000 for grid will take ~5days if each run time is ~1.5min
  #5000 for ew will take ~.6days if each run time is ~.16min (10secs)
  # X = 5000
  # import time
  # StartT = time.time()
  # Updates = np.round(np.linspace(0,X, 11), decimals=0) #To print update after ever 10% completeness
  # for x in range(X):
  #   waveobs3, Synth_spec3 = SynSpec(SynLines, 5670, ext_fct=.1, bary_V=-18.26, Rad_V=205, ThroughHead=HARPSthro, TelFil=TelTemp, CosRays = 6, Plot=False, SNR = 15, EmissLines=[320, 500], OutFile = HARPSbase+'/SynSpec'+str(x)+'.txt', Print=False)
  #   if x in Updates: #to get updates after every 10% increments
  #       print ("Created "+str(x)+"/"+str(X)+"Synthetic spectra. Run time = "+str((time.time()-StartT)/3600.0)+"hours")
  ####=============================(takes about ~15hrs. Run on Hydra!)===================================================================================================================####
