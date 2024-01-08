#This file is used to conduct ellipse fitting for polarized intensity observations of low inclined debris disks using a computer cluster such as Compute Canada. Triple hashtags (###) indicate that something needs to be entered or possibly altered by the user based on their needs.

import numpy as np
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
import emcee
import astropy.units as units
import astropy.constants as const
import os
import glob
import time
import sys
import multiprocessing as mp
from schwimmbad import MPIPool
from scipy.ndimage import gaussian_filter
from scipy import ndimage


def conv_Jy(image, t_exp, conv): #Function to convert GPI polarized data from ADU/coadd to Jy/arcsec^2. Values taken from Esposito et al. (2020). This is not necessary for other data sets.
    adus = image / (t_exp)
    to_Jy = adus*(conv)
    Jyarc = to_Jy / (0.014166**2)
    return Jyarc


data = "/path/to/data" ###Path to your data file.
get_file = get_pkg_data_filename(data)
image = fits.getdata(get_file) #Opening fits file in Python.
image[np.where(np.isnan(image))] = 0 #Convert nan values to 0.

Qphi = image[1] ###Define your radial Stokes Q_phi image from your data set. Change indexing if necessary.
Uphi = image[2] ###Define your radial Stokes U_phi image. This will be used to estimate your uncertainty. Change indexing if necessary


#The following two functions are used to make a noise map using your U_phi image. For more information about these functions see the github page for Tom Esposito (tmesposito).
def make_radii(arr, cen):
    
    grid = np.indices(arr.shape, dtype=float)
    grid[0] -= cen[0]
    grid[1] -= cen[1]
    return np.sqrt((grid[0])**2 + (grid[1])**2)


#Creates standard deviation map.
def get_ann_stdmap(im, cen, radii, r_max=None, mask_edges=False):
    
    if r_max==None:
        r_max = radii.max()
    
    if mask_edges:
        cen = np.array(im.shape)/2
        mask = np.ma.masked_invalid(gaussian_filter(im, mask_edges)).mask
        mask[cen[0]-mask_edges*5:cen[0]+mask_edges*5, cen[1]-mask_edges*5:cen[1]+mask_edges*5] = False
        im = np.ma.masked_array(im, mask=mask).filled(np.nan)
    
    stdmap = np.zeros(im.shape, dtype=float)
    for rr in np.arange(0, r_max, 1):
        if rr==0:
            wr = np.nonzero((radii >= 0) & (radii < 2))
            stdmap[cen[0], cen[1]] = np.nanstd(im[wr])
        else:
            wr = np.nonzero((radii >= rr-0.5) & (radii < rr+0.5))
            stdmap[wr] = np.nanstd(im[wr])
    return stdmap

star = np.array([140, 140]) ###Enter the star x and y coordinates in your image. For GPI the star is located at (140 pix, 140 pix).

#Create noise map.
radii = make_radii(Qphi, star)
Uphi_noise = get_ann_stdmap(Uphi, star, radii, r_max=135) ###Define your max radius with r_max. For GPI I create a noise map out to 135 pixels.
Uphi_noise[Uphi_noise==0.0] = np.nan #Convert 0 values to nans for noise map.


disk_PA = ###Define estimated disk PA here. The PA should be defined as East of North.
disk_PA = disk_PA - 90 #We subtract 90 from the disk PA as we want to rotate the disk so that the major-axis is horizontal in the image when applying the gaussian and ellipse fitting.
disk_inc = ###Define disk inclination here.

PA1 = disk_PA - 90
PA2 = disk_PA + 90
PA_range = np.linspace(PA1, PA2, 91) ###This defines the range of PA values we will cover for the gaussian and ellipse fitting. Essentially we are rotating the disk to measure the radial slice at one disk ansae and then continually rotating the disk and measuring along radial slices until we reach the other ansae (assuming we leave it at -90 and +90 degrees). If the back side of the disk is visable, the user may wish to go beyond -90 and +90 degrees to cover the back side of the disk as well. For example, in the case of TWA 7 (also known as CE ANT), the entire disk is visible given it's low inclination, therefore we can go all the way from -180 degrees to +180 degrees to measure the entire disk. Additionally, the user could do asymmetrical values (e.g. -95 and +100 degrees) in the case where one side of the disk is lower SNR than the other.

y1, y2 = ###define the min and max y values that should be used for the gaussian fitting process in pixels. This is assuming that the disk is rotated by its PA so that major-axis is horizontal in the image. For example, if the disk emission (when rotated) lies between 140 and 160 pixels, I can define y1 and y2 to be 130 and 170 pixels. I choose slightly lower and higher values because I want to ensure that the entire disk emission is covered. If only part of the disk is covered, the gaussian fit will be skewed. You may need to play around with this separately to find the most ideal values.


#Gaussian function.
def gaus(ys,a,b,x0,sigma):
    return a*np.exp(-(ys-x0)**2/(2*sigma**2)) + b #I added an extra parameter "b", as the Guassian may not flatten out at 0.


#This function fetches the surface brightness of the disk along each radial slice.
def prof(xl,masked_Qphi,Uphi_noise,y1,y2):
    br = []
    errs = []
    dy = y2 - y1 + 1
    ys = np.linspace(y1,y2,dy)
    for y in ys:
        int_ys = int(y)
        intens = masked_Qphi[(int_ys,xl)]
        err = Uphi_noise[(int_ys,xl)]
        br.append(intens)
        errs.append(err)

    return br, errs

#This function fits a gaussian profile to the surface brightness along radial slices of the disk.
def gauss_fit(masked_Qphi,Uphi_noise,PA_range,disk_PA,y1,y2,sigma_s=1):
    from scipy.ndimage import zoom
    from scipy.optimize import curve_fit
    from scipy import asarray as ar,exp
    from scipy import ndimage
    import copy
    from scipy.ndimage import gaussian_filter
    
    binned_im = gaussian_filter(masked_Qphi, sigma=sigma_s)  ###Use gaussian filter to smooth out data. Default is sigma=1 pixel. This is often necessary, especially for noisy data. It may be difficult to get a good gaussian fit to the data if it is not smoothed to some degree.
    
    dy = y2 - y1 + 1
    ys = np.linspace(y1,y2,dy) #Creating the array of y-values for each slice. Should be big enough to encompass the entire gaussian profile.
    pix_scale = 0.014166 ###Enter the pixel scale of your instrument here. For GPI, the pixel scale is 0.014166 pixels/arcsec.
    center = 140 ###Change to the center of your image in pixels. For GPI the center of the image along either axis is 140 pixels.
    
    fwhm = [] #full-width-half-max of gaussian.
    error = [] #uncertainty in FWHM.
    means = [] #mean of gaussian.
    mean_err = [] #uncertainty in mean.
    x_arc = [] #x-value of mean.
    for i in PA_range:
        try:
            rotate_image = ndimage.interpolation.rotate(binned_im, i, reshape=False) #Rotating the image.
            x = center  ###Make sure each x-value is an integer or else getting the intensity won't work. x should be the center of your image along the x-axis in pixels.
            br, br_errs = prof(x,rotate_image,Uphi_noise,y1,y2)  #Find intensity across disk at x.
            n = len(ys)
            int_max_br = np.argmax(br)
            mean = ys[int_max_br]
            sigma_g = np.sqrt(sum((ys-mean)**2)/n) #sigma_g is the sigma for the gaussian profile. Not to be confused with sigma_s.
            popt,pcov = curve_fit(gaus,ys,br,p0=[1,0,mean,sigma_g],sigma=br_errs) #Fit gaussian to intensity profile.
        except:
            pass  #If a gaussian cannot be fit, ignore error and continue
        fwhms = np.abs(popt[3]*2.35*pix_scale)  #Calculate fwhm in arcseconds. fwhm is 2.35*sigma.
        perr = np.sqrt(np.diag(pcov)) #Calculate gaussian error in arcseconds
        fwhm.append(fwhms)
        error.append(perr[3]*2.35*pix_scale) #Caluclate the fwhm error in arcseconds.
        #Given the way we are fitting the gaussian profile (rotating the disk and taking vertical slices) the mean value and x-position of the mean are currently in polar coordinates. We need to convert them to cartesian coordinates, which is what happens below:
        rad = popt[2] - center
        PA_rad = np.deg2rad(disk_PA - i)
        mean_cart = rad - (rad - rad*np.cos(PA_rad)) #Caclculating the mean in cartesian coordinates.
        mean_arc = mean_cart*pix_scale #Calculate the mean in arcseconds.
        means.append(mean_arc)
        mean_err.append(perr[2]*pix_scale) #Calculate the mean error in arcseconds.
        
        xs = center + rad*np.sin(PA_rad)
        x_arcs = xs*pix_scale - 2
        x_arc.append(x_arcs)
    
    return np.array(fwhm), np.array(error), np.array(means), np.array(mean_err), np.array(x_arc)


#Function that masks the center of your image. Useful for making sure the noisy regions around the star don't skew the gaussian fitting.
#This function was modified from stack overflow.
def create_circular_mask(h, w, center=None, radius=None):
    
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center >= radius
    return mask

center = (140,140) ###Define the center of your image in pixels.
img = Qphi
h, w = img.shape[:2]
radius = 0.12/0.014166 ###Define the radius of the mask in pixels.
mask = create_circular_mask(h, w,center=center,radius=radius)
masked_Qphi = img.copy()
masked_Qphi[~mask] = 0 #Creates circular mask with value 0.


#This function creates the ring model. This model is a circular ring that is inclined creating an ellipse. Only half the ellipse is generated, while outside the ellipse is a single value determined by the y_offset.
def ring_model(rad_ring,ring_offset,y_offset,inc_ring,rads):
    #rad_ring = the ring radius
    #ring_offset = disk offset along the x-axis.
    #y_offset = disk offset along the y-axis.
    #inc_ring = ring inclination.
    #rads = array of x-values the ring will be generated at.
    
    theta=np.linspace(0,900,901)/5.
    
    xx=rad_ring*np.cos(theta*np.pi/180.)+ring_offset
    yy=rad_ring*np.sin(theta*np.pi/180.)*np.cos(inc_ring*np.pi/180.) + y_offset
    
    spine_models=[]
    
    for i in range(len(rads)):
        if (rads[i] < np.min(xx)) or (rads[i] > np.max(xx)):
            spine_model=0.0+y_offset
            spine_models.append(spine_model)
        else:
            spine_model=np.interp(rads[i],xx,yy,period=360)
            spine_models.append(spine_model)

    return spine_models


#Chi squared function used fror the MCMC fitting.
def chi2(params,rads,masked_Qphi,Uphi_noise,PA_range,y1,y2):
    rad_ring, ring_offset, y_offset, inc_ring, PA = params
    
    fwhm, fwhm_err, img_spine, img_spine_err, x_arc = gauss_fit(masked_Qphi,Uphi_noise,PA_range,PA,y1,y2,1) ###Make sure to change sigma_s.
    
    #For this first foreloop, we are getting rid of any data points that are noisy using a threshold of 3sigma.
    img_spine2 = []
    img_spine_err2 = []
    x_arc2 = []
    for i in range(len(fwhm)):
        if np.abs(fwhm[i]) > 3*np.abs(fwhm_err[i]):
            x_arc2.append(x_arc[i])
            img_spine2.append(img_spine[i])
            img_spine_err2.append(img_spine_err[i])

    #This second foreloop is only necessary if you want to mask out data point close to the star in the case of moderately inclined disks
#    img_spine3 = []
#    img_spine_err3 = []
#    x_arc3 = []
#    for i in range(len(x_arc2)):
#        if x_arc2[i]>=0.22 or x_arc2[i]<=-0.27:
#            x_arc3.append(x_arc2[i])
#            img_spine3.append(img_spine2[i])
#            img_spine_err3.append(img_spine_err2[i])

    spine_front_err = []
    spine_front = []
    x_arc_front = []
    spine_back_err = []
    spine_back = []
    x_arc_back = []
    #Here we are separating the data points that are located in the front of the disk versus the back of the disk. That way we can fit them with the ellipse separately.
    for i in range(len(img_spine2)):
        if img_spine3[i]<=y_offset: ###The "<" sign means that the front side of the disk is below the star. This needs to be changed to ">" if the front side of the disk is above the star otherwise the ellipse fitting won't work.
            spine_front_err.append(img_spine_err2[i])
            spine_front.append(img_spine2[i])
            x_arc_front.append(x_arc2[i])
        if img_spine3[i]>y_offset: ###Similar to above, change the sign to represent the location of the backside of the disk. It should be the opposite of the front side of the disk.
            spine_back_err.append(img_spine_err2[i])
            spine_back.append(img_spine2[i])
            x_arc_back.append(x_arc2[i])
            
    model_spine_front = ring_model(rad_ring,ring_offset,y_offset,inc_ring,rads)
    model_spine_front_vals = np.interp(x_arc_front,rads,model_spine_front)
    
    model_spine_back = -np.array(model_spine_front) - 2*y_offset
    model_spine_back_vals = np.interp(x_arc_back,rads,model_spine_back)
    
    chi_front = np.sum(((np.array(spine_front)-model_spine_front_vals)**2.0)/np.array(spine_front_err)**2)
    chi_back = np.sum(((np.array(spine_back)-model_spine_back_vals)**2.0)/np.array(spine_back_err)**2)

    #chi is evaulated using the fit of the frontside and backside together.
    chi = -0.5*((chi_front+chi_back)/2)
    return chi


#In this function we define our log priors for the MCMC fitting. MAKE SURE TO CHANGE!!!
def get_ln_prior(params,inc_init,PA_init):
    #inc_init and PA_init are the originally defined disk inclination and PA defined near the top of the script.
    rad_ring, ring_offset, y_offset, inc_ring, PA = params

    min_rad = 0.4 ###min and max priors for the disk radius in arcseconds.
    max_rad = 0.5
    
    if rad_ring < min_rad or rad_ring > max_rad:
        ln_prior_rad = -np.inf
    else:
        ln_prior_rad = 1
    
    min_off = -0.1 ###min and max priors for the disk offset along the x-axis in arcseconds.
    max_off = 0.1
    
    if ring_offset < min_off or ring_offset > max_off:
        ln_prior_off = -np.inf
    else:
        ln_prior_off = 1
    
    min_y = -0.1 ###min and max priors for the disk offset along the y-axis in arcseconds.
    max_y = 0.1
    
    if y_offset < min_y or y_offset > max_y:
        ln_prior_y = -np.inf
    else:
        ln_prior_y = 1
    
    min_inc = inc_init - 2 ###min and max priors for the inclination.
    max_inc = inc_init + 2
    
    if inc_ring < min_inc or inc_ring > max_inc:
        ln_prior_inc = -np.inf
    else:
        ln_prior_inc = 1
    
    min_PA = PA_init - 2 ###min and max priors for the disk PA.
    max_PA = PA_init + 2
    
    if PA < min_PA or PA > max_PA:
        ln_prior_PA = -np.inf
    else:
        ln_prior_PA = 1
    
    ln_prior = ln_prior_rad + ln_prior_off + ln_prior_y + ln_prior_inc + ln_prior_PA
    return ln_prior


#Final function calculates the log likelihood for the MCMC fitting.
def get_ln_post(params,rads,masked_Qphi,Uphi_noise,PA_range,y1,y2,inc_init,PA_init):
    ln_prior = get_ln_prior(params,inc_init,PA_init)
    if not np.isfinite(ln_prior):
        return -np.inf
    ln_like = chi2(params,rads,masked_Qphi,Uphi_noise,PA_range,y1,y2)
    if not np.isfinite(ln_like):
        return -np.inf
    else:
        ln_post = ln_like + ln_prior
    return ln_post


rads = np.linspace(-2,2,201) ###Define the rads parameter used in creating the ellipse. Currently the array goes from -2 to 2 arcseconds, but change this for your data.
inc_init = disk_inc
PA_init = disk_PA

#Define MCMC parameters
ndim=5 #number of parameters fitted, in this case five.
nwalkers=200 #number of walkers, currently defined as 200.
nsteps=2000 #number of steps taken for each walker. This isn't a complex model, so a few thousand steps is probably plenty.


namefile = ###Define what you want to name the resulting .dat file which will contain the results of the MCMC.
#Below is the code for the MCMC fitting procedure, which we are using the Python code emcee. We are also using MPIPool so that we can run this script on a computer cluster and run the code in a more efficient time.
with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    
    filename = "/path/to/save/file/{}.dat".format(namefile)
backend = emcee.backends.HDFBackend(filename) #Save results to your .dat file
    backend.reset(nwalkers,ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, get_ln_post, args=[rads,masked_Qphi,Uphi_noise,PA_range,y1,y2,inc_init,PA_init], backend=backend, pool=pool)
    p0 = emcee.utils.sample_ball((0.45,0.0,0.0,inc_init,PA_init),(.02,.01,.05,.05,.05),nwalkers) ###Initial positions for walkers which are initialized over a gaussian distribution. First set of values represents the mean initial positions, the second set of values represent sigma. Make sure sigma is not too big, we don't want walkers to start outside the range of your log priors.
    sampler.run_mcmc(p0, nsteps)


#We are ending the code by having the percentiles for each parameter of the best fitting model printed in the slurm file. That way if something happens to your .dat file you can still see the final results.
flat_samples = sampler.get_chain(flat=True)

for j in range(ndim):
    mcmc = np.percentile(flat_samples[:, j], [16, 50, 84])
    print(mcmc)
