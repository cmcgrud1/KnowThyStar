# KnowThyStar
A package that tests the effectiveness of using the standard spectroscopic data taken before and after the planet passes in front of the host star, out-of-transit (OOT) data. This project is intended to help us determine the best observing, data-detrending, and atmospheric modeling strategies. More discussion of this package can be found on my website: https://www.chima-mcgruder.com/software-products

SCRIPTS: 
1) 0DownloadModels_Alpha0.py - The script used to download all of the PHOENIX models used as the basis to create my synthetic spectrum 

2) AnalyzeDat.py - The majority of this script is used for other analyses. However, ‘degrade_resolution()’ function is used here

3) ModelAtmosphere.py - How I added the effects of stellar variability, which is caused by passing hot and cold spots over the stellar surface. This was modeled by replacing a small piece of the star with a different stellar spectrum. For passing cold spots, I would add a stellar spectrum that has the same composition but a cooler temperature. Conversely, I added a hotter stellar spectrum covering a fraction of the stellar surface to simulate passing hot spots.

4) MakeSynSpec.py - This is where I created the final synthetic spectrum which included various SNR levels, specific instrumental throughputs, tellurics (for ground-based observations), and injected systematics such as shot noise and cosmic rays.

5) ModAtmo_X.py (where X is ‘ChiSqrd’, ‘DualAnneal’, ‘Dynesty’, and ‘Pymultinest’) - These are the different data analysis routines that I used to test how well we could retrieve parameters from the OOT data. 

6) RunGrids.py - Here is where I combined all of the components above to run the large grid of tests in order to identify how effective using the OOT data is depending on observing conditions and the specifics of the star.

7) Read_Results.py - This script was used to interpret the results of these tests, including creating the triangular plots that are included in this repository (the .png files).

8) PNG images - The result of the test, showing how accurately we can identify the injected stellar parameters
