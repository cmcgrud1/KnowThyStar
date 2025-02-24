import ftplib
import sys
import wget
import os
import glob
#To download the PHONEIX models
######===== INPUT =====######
Ftp_link = 'phoenix.astro.physik.uni-goettingen.de'
Main_dir = '/HiResFITS/PHOENIX-ACES-AGSS-COND-2011'
outpath = '/Users/chimamcgruder/Research/KnowThyStar/PHOENIX/' #path of where want downloaded files to go RYAN MUST CHANGE!!!!!!
######==========######

def LogIn(ftp_link=Ftp_link, main_dir=Main_dir): #to log in to the online ftp server storing all the phenix models
	ftp = ftplib.FTP(ftp_link)
	ftp.login()
	ftp.cwd(main_dir)
	print ("In path:", ftp.pwd())
	filelist=ftp.nlst()
	Models_Alpha0 = []
	for f in filelist:
		if 'Alpha' not in f: #only want to download models where Alpha = 0
			Models_Alpha0.append(f)
	print ("Directories:", Models_Alpha0, '\n')
	return ftp, Models_Alpha0

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

def DownloadMods(FtpLink=Ftp_link, MainDir=Main_dir, threads=None): #to download all the PHEONIX model data
	def download(Models, LINK=FtpLink, Dir=MainDir, thred_count=0): #sub function to do downloading loop (so can be parallelized)
		for z in Models: #for each metallicity parent directory
			try: #should still be logged-in to ftp forever (keeps quitting for some reason)
				models=ftp.nlst(z)
			except: #But if not, re-login here
				ftp = LogIn(ftp_link=LINK, main_dir=Dir)[0] #use the same Models from what was initially passed in fuction, so every recall has the subdirectories in the same order
				models=ftp.nlst(z)
			L_mods, cnt = len(models), 0
			print ("Downlading all ", L_mods, "files in subdirectory '"+z+"'...")
			for m in models: #for each model in each parent directory
				file = m.split('/') #1st element is the path defining the metallicity. 2nd element is the name of the file
				if os.path.isfile(file[1]):
					cnt +=1
					continue #skip if already downloaded file
				cnt +=1
				print ("downloading \033[1m"+m+"\033[0m. File "+str(cnt)+"/"+str(L_mods)+" in path : ")
				wget.download('ftp://'+FtpLink+MainDir+'/'+m, out=outpath)
				print ("\n")
			if cnt != L_mods:
				sys.exit("Something wrong! Counted "+str(cnt)+" out of "+str(L_mods)+" files")
			print ("Downloaded all "+str(cnt)+"/"+str(L_mods)+" files\n")
		try: #if didn't already quit out of ftp, quit here so not still logged in to online database 
			ftp.quit()
		except:
			pass # For some reason, after reading info in every subdirectory it automatically quits out of the ftp
		print ("Finished Thread-"+str(thred_count)+"!")
		print ('\n')
		return None

	ftp, Models_Alpha0 = LogIn(ftp_link=FtpLink, main_dir=MainDir) #only have to save the list of metallicity subdirectories once
	if threads:
		MulitModels = chunks(Models_Alpha0, threads) #to break the list of models into sublist, which will be downloaded in parallel
		import threading
		class download_parallel (threading.Thread): #to make a multi-threading subclass
			def __init__(self, LIST, counter):
				threading.Thread.__init__(self)
				self.counter = counter
				self.name = "Thread-"+str(counter)
				self.LIST = LIST
			def run(self):
				print ("Starting " + self.name)
				download(self.LIST, thred_count=self.counter) 
		conter=1
		THREADs = [] #list of threads, needed to keep track of all the threads so can join them all at the end
		for multi in MulitModels:
			THREADs.append(download_parallel(multi, conter))
			THREADs[-1].start()
			conter+=1
		for thread in THREADs: # now join them all so code doesn't finish until all threads are done
			thread.join()
	else:
		download(Models_Alpha0)
	return None

def check_file_count(Directory, Total=None):
	files = glob.glob('lte*.fits')
	if Total: #if we know how many files needed to be downloaded
		print ("Downloaded\033[1m", str(len(files))+'/'+str(Total), "\033[0mfiles.")
	else: #Otherwise, just print number of files downloaded
		print ("Downloaded\033[1m", str(len(files)), "\033[0mfiles.")
	return files	
#### write here to compress the fits files to gzips
####Code
#### is 
##### supposed 
#### to 
##### be 
##### here

######===== Run Functions =====######
if __name__ == '__main__':
	# DownloadMods(threads = 9)
	# print ("FINISHED!")
	check_file_count('PHOENIX', Total=7559)

