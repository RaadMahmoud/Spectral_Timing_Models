import matplotlib
matplotlib.use('Agg')
from pylab import *
import numpy as np
import numpy.fft as fourier
import matplotlib.pyplot as plt
font = {'family' : 'normal',
        'style'  : 'italic',
        'size'   : 11}
matplotlib.rc('font', **font)
from matplotlib import gridspec
import scipy.interpolate as interp

#######################################################################################
#++++++++++++++++++++Global variables+++++++++++++++++++++++++++++++++++++++++++++++++#
#######################################################################################


'''Physical constants'''
pi = np.pi
c = 29979245800.0               #cm / s
M_sol =  1.989e33               #g Solar mass
planck = 6.63*10**-27           #cm2 g s-1
hbar = planck/(2*pi)            #cm2 g s-1
G = 6.6743e-8                   #cm3 g-1 s-2

'''Simulation parameters'''
dt_sim = 2**-10                  #s
dt_dat = 0.012                  #s
NfDec = 20                      #Number of bins per frequency decade when computing PSD/lag-frequency.
NfBand = 40                     #Number of bins in each of the three lag-energy frequency bands.
M_len = 50                      #Length (in seconds) of segments on which lightcurves are ensemble averaged.
N_r = 50                        #Number of radial bins in the simulation.


'''Object/data-specific parameters. Derived from either the xspec spectral model fit or literature.'''
Elow_min = 0.3                  #Lower bound of the low energy band (keV)
Elow_max = 0.7                  #Upper bound of the low energy band (keV)
Eint_min = 0.7                  #Lower bound of the intermediate energy band (keV)
Eint_max = 1.5                  #Upper bound of the intermediate energy band (keV)
Ehi_min = 3.0                   #Lower bound of the high energy band (keV)
Ehi_max = 5.0                   #Upper bound of the high energy band (keV)

freq_slo_min = 0.02             #Lower bound of the low-frequency band of the lag-energy spectra (Hz).
freq_slo_max = 0.3              #Upper bound of the low-frequency band of the lag-energy spectra (Hz).
freq_mid_min = 0.3              #Lower bound of the middle-frequency band of the lag-energy spectra (Hz).
freq_mid_max = 1                #Upper bound of the middle-frequency band of the lag-energy spectra (Hz).
freq_fas_min = 1                #Lower bound of the high-frequency band of the lag-energy spectra (Hz).
freq_fas_max = 30               #Upper bound of the high-frequency band of the lag-energy spectra (Hz).

M_bh = 7. * M_sol               #g --------  Black hole mass.
a = 0.93                        #------------Dimensionless spin parameter
R_g = G*M_bh /c**2              #cm
incl = 50./180 * pi             #radians
Omega_on2pi_soft = 0.296789
Omega_on2pi_hard = Omega_on2pi_soft


'''Physical model parameters'''
Amp1 =6e-5
r_o  =19.5
r_i =4.02
F_vard1= 1.64
F_vardC =1.1 
r_vard1 =19.2
wid1 =0.246
D_DS =1.55
D_SH =4.91
f_disc_V =0.21
S_m = 0.

gamma = 4.92
B_disc =  0.03
m_disc = 0.5
B_flow =0.175
m_flow =1.20
r_disc = 400.

########################################################################################################################
#+Set up the frequency grid on which the power spectra and lag-frequency spectra of the data and model will be binned.+#
########################################################################################################################

di_dat = 2 ** int(log2(M_len/dt_dat))
freqs_dat = abs(fourier.fftfreq(di_dat, dt_dat))[1:di_dat/2+1]
data_fbins = np.logspace(log10(freqs_dat[0]), log10(freqs_dat[-1]), int(log10(freqs_dat[-1]) - log10(freqs_dat[0])) * NfDec)
inds_dat = np.digitize(freqs_dat, data_fbins)

N_freq = 2 ** int(log2(M_len/dt_sim))
freqs_sim = abs(fourier.fftfreq(N_freq, dt_sim))[1:N_freq/2+1]
inds_sim = np.digitize(freqs_sim, data_fbins)

to_be_deleted = []
for i_d in range(inds_dat[-1]):
    if i_d not in inds_sim:
        to_be_deleted.append(i_d)
data_fbins = np.delete(data_fbins, to_be_deleted)

freqs_binned = np.array(())
inds = np.digitize(freqs_dat, data_fbins)
for i in range(1,inds[-1]):
    if i in inds:
        i_min, i_max = min(np.argwhere(inds == i))[0], max(np.argwhere(inds==i))[0]
        freqs_binned = np.append(freqs_binned, (freqs_dat[i_min] + freqs_dat[i_max])/2)

########################################################################################################################
#+Tools for manipulating, and producing statistics from data light curves here+++++++++++++++++++++++++++++++++++++++++#
########################################################################################################################
        
def poisson_power(s, sbkg, dt):
    '''In the absence of dead-time (e.g. for XMM-Newton where dead time is negligible but not for e.g. RXTE) compute the rms-normalized power spectrum of the poisson noise.'''
    P_noise = 2 * (np.average(s-sbkg) + np.average(sbkg))/np.average(s-sbkg)**2 
    return P_noise

def ensemble_averaging(s, dt,sbkg, mode):
    '''Given a lightcurve, s, timebinning dt and background sbkg, compute the ensemble-averaged rms-normalized power spectrum.'''
    '''If mode == 'clean', account for the poisson noise. If 'dirty', do not account for poisson noise.'''
    di = 2 ** int(log2(M_len/dt))
    M =  int(len(s) /di)
    s = s[0: M*di]
    s.shape = (M, di)
    
    sbkg = sbkg[0:M*di]
    sbkg.shape = (M, di)

    power_spectra = np.zeros((M, di/2), dtype = 'complex')
    power_errors = np.array((di/2), dtype = 'complex')
    
    if mode == 'clean':
        for j in range(M):
            S = fourier.fft(s[j]) / di
            power_0 = (np.conj(S) * S) * di *2* dt / (np.average(s[j]- sbkg[j]))**2
            power_spectra[j] = power_0[1:di/2+1] - poisson_power(s[j],sbkg[j], dt)
            freqs = abs(fourier.fftfreq(len(S), dt))[1:di/2+1]
            
    if mode == 'dirty':
        for j in range(M):
            S = fourier.fft(s[j]) / di
            power_0 = (np.conj(S) * S) * di *2* dt / (np.average(s[j]- sbkg[j]))**2
            power_spectra[j] = power_0[1:di/2+1]
            freqs = abs(fourier.fftfreq(len(S), dt))[1:di/2+1]
    
    powers_ave = power_spectra.sum(axis = 0) / (M)
    
    power_errors = sqrt(np.var(power_spectra, axis=0, ddof = 1)) / sqrt(M)
    
    return freqs, powers_ave, power_errors
    
def binned_spectrum(s, dt, sbkg, mode, fbins=data_fbins):
    '''Given a lightcurve, s, timebinning dt and background sbkg, compute the ensemble-averaged AND rebinned rms-normalized power spectrum.'''
    '''Binned in frequency on grid fbins.'''
    A = ensemble_averaging(s, dt, sbkg, mode)
    freqs, P, geoerrs  = A[0], A[1], A[2]

    P_binned = np.array((), dtype = 'complex')
    dP = np.array((), dtype = 'complex')
    
    inds = np.digitize(freqs, fbins)
    K = np.array(())
    
    for i in range(1, inds[-1]):
        if i in inds:
            i_min, i_max = min(np.argwhere(inds == i))[0], max(np.argwhere(inds==i))[0]
            P_binned = np.append(P_binned, np.average(P[i_min:i_max+1]))
            K = np.append(K, i_max-i_min)
            if i_max != i_min:
                dP = np.append(dP, sqrt(sum(geoerrs[i_min:i_max+1]**2))/(i_max-i_min))
            else:
                dP = np.append(dP, geoerrs[i_min])
        else:
            pass
    
    return P_binned, dP, K

def Coherence(s, h, dt, sbkg, hbkg, fbins=data_fbins):
    '''Compute the coherence between two lightcurves s and h which have time bins dt. Binning consistent with binned_spectrum.'''
    '''This function is not used in this code, but useful.'''
    
    P_s = abs(binned_spectrum(s, dt, sbkg, 'clean')[0])
    P_h = abs(binned_spectrum(h, dt, hbkg, 'clean')[0])
    
    di = 2 ** int(log2(M_len/dt))
    M =  int(len(s) /di)
    
    h = h[0: M*di]
    h.shape = (M, di)
    s = s[0: M*di]
    s.shape = (M, di)
    
    sbkg = sbkg[0:M*di]
    sbkg.shape = (M, di)
    hbkg = hbkg[0:M*di]
    hbkg.shape = (M, di)

    cross_spec = np.zeros((M, di/2), dtype = 'complex')
    N_s_ens, N_h_ens = np.zeros((M, di/2), dtype = 'float'), np.zeros((M, di/2), dtype = 'float')
    
    for j in range(M):
        H = fourier.fft(h[j]) / di
        S = fourier.fft(s[j]) / di
        N_s_ens[j] = poisson_power(s[j], sbkg[j], dt)
        N_h_ens[j] = poisson_power(h[j], hbkg[j], dt)

        cross_0 = (np.conj(S) * H) * di * 2 * dt /  (np.average(h[j] - hbkg[j])*np.average(s[j]-sbkg[j]))
        cross_spec[j] = cross_0[1:di/2+1]
        freqs_ens = abs(fourier.fftfreq(len(H), dt))[1:di/2+1]
    
    cross_ens = cross_spec.sum(axis = 0) / M
    cross_re_ens = real(cross_ens)
    cross_im_ens = imag(cross_ens)
    
    N_s_ens = N_s_ens.sum(axis = 0) / M
    N_h_ens = N_h_ens.sum(axis = 0) / M 
    
    freqs = np.array(())
    cross_re = np.array((), dtype = 'float')
    cross_im = np.array((), dtype = 'float')
    
    N_s_sq, N_h_sq = np.array(()), np.array(())
    
    inds = np.digitize(freqs_ens, fbins)
    K = np.array(())
    
    for i in range(1, inds[-1]):
        if i in inds:
            i_min, i_max = min(np.argwhere(inds == i))[0], max(np.argwhere(inds==i))[0]
            freqs = np.append(freqs, (freqs_ens[i_min] + freqs_ens[i_max])/2)
            cross_re = np.append(cross_re, np.average(cross_re_ens[i_min:i_max+1]))
            cross_im = np.append(cross_im, np.average(cross_im_ens[i_min:i_max+1]))
            N_s_sq = np.append(N_s_sq, np.average(N_s_ens[i_min:i_max+1]))
            N_h_sq = np.append(N_h_sq, np.average(N_h_ens[i_min:i_max+1]))
            K = np.append(K, i_max-i_min+1)
        else:
            pass
        
    n2 = (P_h *N_s_sq + P_s * N_h_sq + N_h_sq*N_s_sq) / (K*M)
    
    gamma2 = (cross_re**2 + cross_im**2 - n2) /( (P_s)* (P_h))
    
    for l in range(len(gamma2)):
        if gamma2[l] < 0:
            gamma2[l]= 1 / (K[l]*M)
    
    dgamma = sqrt(2) * gamma2 * (1-gamma2) / ( (np.conj(gamma2) * gamma2)**0.25 * np.sqrt(K*M))
    
    err_gamma2 = gamma2 / np.sqrt(K*M) * ( 2*n2**2 *  K*M / (cross_re**2 + cross_im**2 - n2)**2 + N_s_sq**2 / P_s**2 + N_h_sq**2 / P_h**2 + K*M* dgamma / gamma2**2)**0.5
    
    return gamma2, err_gamma2

def FourierLag(s, h, dt, sbkg, hbkg, fbins=data_fbins):
    '''Compute the fourier time lag between two lightcurves s and h which have time bins dt. Binning consistent with binned_spectrum.'''
    
    P_s = abs(binned_spectrum(s, dt, sbkg, 'clean')[0])
    P_h = abs(binned_spectrum(h, dt, hbkg, 'clean')[0])
    P_s_noisy = abs(binned_spectrum(s, dt, sbkg, 'dirty')[0])
    P_h_noisy = abs(binned_spectrum(h, dt, hbkg, 'dirty')[0])
    
    di = 2 ** int(log2(M_len/dt))
    M =  int(len(s) /di)
    
    h = h[0: M*di]
    h.shape = (M, di)
    s = s[0: M*di]
    s.shape = (M, di)
    
    sbkg = sbkg[0:M*di]
    sbkg.shape = (M, di)
    hbkg = hbkg[0:M*di]
    hbkg.shape = (M, di)

    cross_spec = np.zeros((M, di/2), dtype = 'complex')
    N_s_ens, N_h_ens = np.zeros((M, di/2), dtype = 'float'), np.zeros((M, di/2), dtype = 'float')
    
    for j in range(M):
        H = fourier.fft(h[j]) / di
        S = fourier.fft(s[j]) / di
        N_s_ens[j] = poisson_power(s[j], sbkg[j], dt)
        N_h_ens[j] = poisson_power(h[j], hbkg[j], dt)

        cross_0 = (np.conj(S) * H) * di * 2 * dt /  (np.average(h[j] - hbkg[j])*np.average(s[j]-sbkg[j]))
        cross_spec[j] = cross_0[1:di/2+1]
        freqs_ens = abs(fourier.fftfreq(len(H), dt))[1:di/2+1]
    
    cross_ens = cross_spec.sum(axis = 0) / M
    cross_re_ens = real(cross_ens)
    cross_im_ens = imag(cross_ens)
    
    N_s_ens = N_s_ens.sum(axis = 0) / M
    N_h_ens = N_h_ens.sum(axis = 0) / M 
    
    freqs = np.array(())
    cross_re = np.array((), dtype = 'float')
    cross_im = np.array((), dtype = 'float')
    
    N_s_sq, N_h_sq = np.array(()), np.array(())
    
    inds = np.digitize(freqs_ens, fbins)
    K = np.array(())
    
    for i in range(1, inds[-1]):
        if i in inds:
            i_min, i_max = min(np.argwhere(inds == i))[0], max(np.argwhere(inds==i))[0]
            freqs = np.append(freqs, (freqs_ens[i_min] + freqs_ens[i_max])/2)
            cross_re = np.append(cross_re, np.average(cross_re_ens[i_min:i_max+1]))
            cross_im = np.append(cross_im, np.average(cross_im_ens[i_min:i_max+1]))
            N_s_sq = np.append(N_s_sq, np.average(N_s_ens[i_min:i_max+1]))
            N_h_sq = np.append(N_h_sq, np.average(N_h_ens[i_min:i_max+1]))
            K = np.append(K, i_max-i_min+1)
        else:
            pass

    n2 = (P_h *N_s_sq + P_s * N_h_sq + N_h_sq*N_s_sq) / (K*M)
    
    gamma2 = (cross_re**2 + cross_im**2 - n2) / (P_s_noisy* P_h_noisy)
    
    for l in range(len(gamma2)):
        if gamma2[l] < 0:
            gamma2[l]= 1 / (K[l]*M)
    
    dphi = sqrt((1-gamma2) / (2 * gamma2 * K*M))
    
    dtau = dphi / (2*pi*freqs)

    Dphi = np.arctan2(cross_im, cross_re)
    taus = Dphi / (2*pi*freqs)
    
    f_aves_hls, tau_means_hls, dtau_hls = np.array(()), np.array(()), np.array(())
    f_aves_slh, tau_means_slh, dtau_slh = np.array(()), np.array(()), np.array(())
    
    
    for i in range(0, len(taus)):
        if taus[i] < 0:
            f_aves_hls = np.append(f_aves_hls, freqs[i])
            tau_means_hls = np.append(tau_means_hls, taus[i])
            dtau_hls = np.append(dtau_hls, dtau[i])
        else:
            f_aves_slh = np.append(f_aves_slh, freqs[i])
            tau_means_slh = np.append(tau_means_slh, taus[i])
            dtau_slh = np.append(dtau_slh, dtau[i]) 
    
    return f_aves_hls,f_aves_slh, -tau_means_hls, tau_means_slh, dtau_hls, dtau_slh, -taus, dtau


########################################################################################################################
#+Importing and processing the power spectra and lags for our lightcurves (from De Marco 2017) here++++++++++++++++++++#
########################################################################################################################

s_low = np.genfromtxt('lightcurves/300_700eV', skip_header = 200, skip_footer = 37000)
s_low = s_low[np.logical_not(np.isnan(s_low[:,1]))] / dt_dat
s_series_obs_low = s_low[:,1]
s_series_obs_low = s_series_obs_low[:len(s_series_obs_low) - len(s_series_obs_low)%2]
s_series_obs_low = s_series_obs_low.reshape(int(len(s_series_obs_low)/2), 2)
s_series_obs_low = s_series_obs_low.sum(axis = 1)
sbkg_low = np.zeros(len(s_series_obs_low))
pow_s_obs_low = binned_spectrum(s_series_obs_low, dt_dat, sbkg_low, 'clean')
P_s_obs_low, dP_s_obs_low = pow_s_obs_low[0], pow_s_obs_low[1]
fP_s_obs_low, fdP_s_obs_low = freqs_binned*P_s_obs_low, freqs_binned*dP_s_obs_low

#################################################

s_int = np.genfromtxt('lightcurves/700_1500eV', skip_header = 200, skip_footer =37000)
s_int = s_int[np.logical_not(np.isnan(s_int[:,1]))]/ dt_dat
s_series_obs_int = s_int[:,1]
s_series_obs_int = s_series_obs_int[:len(s_series_obs_int) - len(s_series_obs_int)%2]
s_series_obs_int = s_series_obs_int.reshape(int(len(s_series_obs_int)/2), 2)
s_series_obs_int = s_series_obs_int.sum(axis = 1)
sbkg_int = np.zeros(len(s_series_obs_int))
pow_s_obs_int = binned_spectrum(s_series_obs_int, dt_dat, sbkg_int, 'clean')
P_s_obs_int, dP_s_obs_int = pow_s_obs_int[0], pow_s_obs_int[1]
fP_s_obs_int, fdP_s_obs_int = freqs_binned*P_s_obs_int, freqs_binned*dP_s_obs_int

##########################################

s_hi = np.genfromtxt('lightcurves/3000_5000eV', skip_header = 200, skip_footer = 37000)
s_hi = s_hi[np.logical_not(np.isnan(s_hi[:,1]))]/ dt_dat
s_series_obs_hi = s_hi[:,1]
s_series_obs_hi = s_series_obs_hi[:len(s_series_obs_hi) - len(s_series_obs_hi)%2]
s_series_obs_hi = s_series_obs_hi.reshape(int(len(s_series_obs_hi)/2), 2)
s_series_obs_hi = s_series_obs_hi.sum(axis = 1)
sbkg_hi = np.zeros(len(s_series_obs_hi))
pow_s_obs_hi = binned_spectrum(s_series_obs_hi, dt_dat, sbkg_hi, 'clean')
P_s_obs_hi, dP_s_obs_hi = pow_s_obs_hi[0], pow_s_obs_hi[1]
fP_s_obs_hi, fdP_s_obs_hi = freqs_binned*P_s_obs_hi, freqs_binned*dP_s_obs_hi
 
f_aves_HL_obs, f_aves_LH_obs, tau_means_HL_obs, tau_means_LH_obs, dtau_HL_obs, dtau_LH_obs, taus_HL, dtaus_HL = FourierLag(s_series_obs_low, s_series_obs_hi, dt_dat, sbkg_low, sbkg_hi)

f_aves_IL_obs, f_aves_LI_obs, tau_means_IL_obs, tau_means_LI_obs, dtau_IL_obs, dtau_LI_obs, taus_IL, dtaus_IL = FourierLag(s_series_obs_low, s_series_obs_int, dt_dat, sbkg_low, sbkg_int)

f_aves_HI_obs, f_aves_IH_obs, tau_means_HI_obs, tau_means_IH_obs, dtau_HI_obs, dtau_IH_obs, taus_HI, dtaus_HI = FourierLag(s_series_obs_int, s_series_obs_hi, dt_dat, sbkg_int, sbkg_hi)


########################################################################################################################
#+Plotting the data alone++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
########################################################################################################################

fig = plt.figure(figsize=(3, 5))
gs = gridspec.GridSpec(5, 1)
ax3 = plt.subplot(gs[0:2, :])
ax4 = plt.subplot(gs[2, :])
ax5 = plt.subplot(gs[3, :])
ax6 = plt.subplot(gs[4, :])

ax3.fill_between(freqs_binned, fP_s_obs_low - fdP_s_obs_low, fP_s_obs_low + fdP_s_obs_low, color = 'pink', alpha = 1, label = '3-5kev data')
ax3.fill_between(freqs_binned, fP_s_obs_int - fdP_s_obs_int, fP_s_obs_int + fdP_s_obs_int, color = 'green', alpha = 0.35, label = '10-20kev data')
ax3.fill_between(freqs_binned, fP_s_obs_hi - fdP_s_obs_hi, fP_s_obs_hi + fdP_s_obs_hi, color = 'blue', alpha = 0.35, label = '20-35kev data')
ax3.tick_params(axis = 'both', bottom='on', top='on', left='on', right='on', direction= 'in')
ax3.set_ylim(0.0002, 0.035)
ax3.set_xlim(0.045,40)
ax3.set_yscale('log')
ax3.set_xscale('log')
ax3.set_ylabel(r'$fP_f$ $([rms/mean]^2)$')

ax4.set_xlim(0.045,40)
ax4.set_ylabel(r'$Int-Low$ $lag$ $(s)$')
ax4.set_yscale('log')
ax4.set_xscale('log')
ax4.tick_params(axis = 'both', bottom='on', top='on', left='on', right='on', direction= 'in')
ax4.errorbar(freqs_binned, taus_IL, dtaus_IL, fmt = 's', color = 'darkred', ecolor = 'darkred',capsize = 0)

ax5.set_xlim(0.045,40)
ax5.set_yscale('log')
ax5.set_xscale('log')
ax5.set_xscale('log')
ax5.set_ylabel(r'$High-Int$ $lag$ $(s)$')
ax5.tick_params(axis = 'both', bottom='on', top='on', left='on', right='on', direction= 'in')
ax5.errorbar(freqs_binned, taus_HI, dtaus_HI, fmt = 'D', color = 'darkgreen', ecolor = 'darkgreen',capsize = 0)

ax6.set_xlim(0.045,40)
ax6.set_ylabel(r'$High-Low$ $lag$ $(s)$')
ax6.set_yscale('log')
ax6.set_xscale('log')
ax6.set_xlabel(r'$f$ $(Hz)$')
ax6.tick_params(axis = 'both', bottom='on', top='on', left='on', right='on', direction= 'in')
ax6.errorbar(freqs_binned, taus_HL, dtaus_HL, fmt = 'o', color = 'darkblue', ecolor = 'darkblue',capsize = 0)

#######################################################################################
#+++++++++++++ The model subroutines and main function (output). +++++++++++++++++++++#
#######################################################################################

def f_kep(r):
    '''Keplerian frequency at radius r.'''
    return c / (2*pi*R_g*(r**(3./2) + a))

def f_alpha(r, B, m):
    '''Viscous frequency at radius r.'''
    return B * r**-m * c / (2*pi*R_g*(r**(3./2) + a))

def lorentzian(r, dr, i, f, i_DS, i_SH, r_o, r_i, B_disc, m_disc, B_flow, m_flow, dt, F_vard1, F_vardC, r_vard1, wid1):   
    '''A power spectrum Lorentzian at radius r given model parameters.'''
    if i < i_DS:
        f_0 = f_alpha(r, B_disc, m_disc)
    else:
        f_0 = f_alpha(r, B_flow, m_flow)        

    N_dec = N_r / log10(r_o/r_i)

    F_vard = F_vard1 * np.exp(-(r-r_vard1)**2 / (2*wid1**2)) + F_vardC

    sig = F_vard / sqrt(N_dec)
        
    P_i = f_0 / (f_0**2 + f**2) * (2*sig**2/pi)

    return P_i

def r_bounds(r_o, r_i, N_r):
    '''The radial bounds of our hot flow/viscous disc annuli.'''
    return np.logspace( log10(r_o), log10(r_i), N_r + 1 )

def emissivity(r, r_o, Amp1, gamma, r_vard1, wid1):
    '''Emissivity at radius r, given model parameters.'''
    emiss = Amp1 * np.exp(-(r-r_vard1)**2. / (2.*wid1**2.)) + r ** -gamma
    return emiss

def transferfn(r_disc, r_o, dt_sim = 2**-13):
    '''Compute the point-illumination transfer function between a source at r=0 and a disc extending from r_o to r_disc.'''
    N_disc = 2500
    N_phi = 2500
    dphi = 2*pi / N_phi
    taus = np.arange(0, M_len, dt_sim)
    
    r_bins = r_bounds(r_disc, r_o, N_disc)
    rs = np.asarray([(r_bins[i+1]+r_bins[i])/2 for i in range(len(r_bins)-1)])
    drs = [(r_bins[i] -r_bins[i+1]) for i in range(len(r_bins)-1)]

    tf = np.zeros(len(taus))
    for i in range(N_disc):
        for j in range(N_phi):
            t = rs[i] * R_g / c * (1.0 - np.sin(incl)*np.cos(-pi/2. + dphi*j))
            tindex = int(t/dt_sim)+1
            f_r = drs[i]/(r_disc-r_o)
            f_phi = dphi/(2*pi)
            tf[tindex] += f_r*f_phi
    
    N_freq = 2 ** int(log2(M_len/dt_sim))
    TF_raw = np.conj(fourier.fft(tf)[1:N_freq/2 + 1])
    
    return TF_raw
    
def TFrebin(TF_raw, fbins, dt_sim=2**-13):
    '''Rebin the transfer function and exclue those parts of the TF which fall outside the range of the data.'''
    N_freq = 2 ** int(log2(M_len/dt_sim))
    freqs = abs(fourier.fftfreq(N_freq, dt_sim))[1:N_freq/2+1]
    
    TF_selected = np.array(())
    
    inds = np.digitize(freqs, fbins)
    
    for i in range(1, inds[-1]+1):
        if i in inds:
            i_min = min(np.argwhere(inds == i))[0]
            i_max = max(np.argwhere(inds == i))[0]
            TF_selected = np.append(TF_selected, TF_raw[i_min])
            if i_min != i_max:
                TF_selected = np.append(TF_selected, TF_raw[i_max])
                
    return TF_selected
   

def PSDCalc(j, k, freqs, prop_spectra, weights, weights_refl, reTF, imTF, TF2, proplag, S_m, D):
    '''Given propagated power spectra (prop_spectra), compute the apparent power spectra in an energy band with direct weights (weights) and reflected weights (weights_refl).'''
    out = 2 * prop_spectra[j] * np.exp(-S_m*proplag*freqs) * \
                               ( np.cos(2*pi*proplag*freqs) *  (weights[j]*weights[k] + reTF * (weights[j]*weights_refl[k] + weights_refl[j] * weights[k]) +\
                                        TF2 * weights_refl[j]*weights_refl[k]) +\
                                imTF * np.sin(2*pi*proplag*freqs) * (weights[j] * weights_refl[k] - weights_refl[j] * weights[k])) / D
    return out

    
def ReCalc(j, k, freqs, prop_spectra, weights_1, weights_2, weights_refl_1, weights_refl_2, reTF, imTF, TF2, proplag, S_m, D):
    '''A subroutine for the function Recomp below.'''
    out = prop_spectra[j] * np.exp(-S_m*proplag*freqs) * \
      (np.cos(2*pi*proplag*freqs) * ( (weights_1[j] * weights_2[k] + weights_1[k] * weights_2[j]) + \
                                 reTF * (weights_1[j] * weights_refl_2[k] +  weights_refl_1[j] * weights_2[k] +\
                                     weights_1[k] * weights_refl_2[j] + weights_refl_1[k] * weights_2[j])+\
                                 TF2 * (weights_refl_1[j] * weights_refl_2[k] + weights_refl_1[k] * weights_refl_2[j])) - \
      np.sin(2*pi*proplag*freqs) * imTF * (weights_1[j] * weights_refl_2[k] - weights_refl_1[j] * weights_2[k] -\
                                     weights_1[k] * weights_refl_2[j] + weights_refl_1[k] * weights_2[j])) / D                      
    
    return out
    
def ImCalc(j, k, freqs, prop_spectra, weights_1, weights_2, weights_refl_1, weights_refl_2, reTF, imTF, TF2, proplag, S_m, D):
    '''A subroutine for the function Imcomp below.'''
    out = prop_spectra[j] * np.exp(-S_m*proplag*freqs) * \
    (np.sin(2*pi*proplag*freqs) * ( (weights_1[j] * weights_2[k] - weights_1[k] * weights_2[j]) + \
                                reTF * (weights_1[j] * weights_refl_2[k] +  weights_refl_1[j] * weights_2[k] -\
                                    weights_1[k] * weights_refl_2[j] - weights_refl_1[k] * weights_2[j])+\
                                TF2 * (weights_refl_1[j] * weights_refl_2[k] - weights_refl_1[k] * weights_refl_2[j])) + \
    np.cos(2*pi*proplag*freqs) * imTF * (weights_1[j] * weights_refl_2[k] - weights_refl_1[j] * weights_2[k] +\
                                    weights_1[k] * weights_refl_2[j] - weights_refl_1[k] * weights_2[j])) / D
                                                                 
    return out

def Recomp(freqs, prop_spectra, truelags, weights_1, weights_2, weights_refl_1, weights_refl_2, L_1_Disc_Nat,L_2_Disc_Nat, reTF, imTF, TF2, S_m,\
           D_DS, D_SH, i_DS, i_SH, weights_refl_base_1 = None, weights_refl_base_2 = None):
    '''Given propagated power spectra (prop_spectra), compute the real part of the apparent complex spectrum between two energy bands with direct weights (weights1, weights2)
        reflected weights (weights_refl_1, weights_refl_2), and constant disc flux contributions (L_1_Disc_Nat, L_2_Disc_Nat). '''
    
    Recomp = np.zeros(len(freqs), dtype ='complex')
    
    for k in range(N_r):
        Recomp += prop_spectra[k] * (weights_1[k]*weights_2[k] + reTF * (weights_1[k]*weights_refl_2[k] + weights_refl_1[k]*weights_2[k]) +  weights_refl_1[k] * weights_refl_2[k] * TF2)
        if k ==0:
            pass
                
        else:
            if k < i_DS:
                for j in range(k):
                    proplag = np.sum(truelags[j:k])
                    Recomp += ReCalc(j, k, freqs, prop_spectra, weights_1, weights_2, weights_refl_1, weights_refl_2, reTF, imTF, TF2, proplag, S_m, 1)

            elif i_DS <= k < i_SH:
                for j in range(k):
                    proplag = np.sum(truelags[j:k])
                    if j < i_DS:
                        Recomp += ReCalc(j, k, freqs, prop_spectra, weights_1, weights_2, weights_refl_1, weights_refl_2, reTF, imTF, TF2, proplag, S_m, D_DS)
                    else:
                        Recomp += ReCalc(j, k, freqs, prop_spectra, weights_1, weights_2, weights_refl_1, weights_refl_2, reTF, imTF, TF2, proplag, S_m, 1)

            else:
                for j in range(k):
                    proplag = np.sum(truelags[j:k])
                    if j < i_DS:
                        Recomp += ReCalc(j, k, freqs, prop_spectra, weights_1, weights_2, weights_refl_1, weights_refl_2, reTF, imTF, TF2, proplag, S_m, D_DS*D_SH)
                    elif i_DS <= j < i_SH:
                        Recomp += ReCalc(j, k, freqs, prop_spectra, weights_1, weights_2, weights_refl_1, weights_refl_2, reTF, imTF, TF2, proplag, S_m, D_SH)
                    else:
                        Recomp += ReCalc(j, k, freqs, prop_spectra, weights_1, weights_2, weights_refl_1, weights_refl_2, reTF, imTF, TF2, proplag, S_m, 1)

    if any(weights_refl_base_1==None):
        pass
    else:
        weights_refl_1 = weights_refl_base_1

    Recomp = Recomp / ( (np.sum(weights_1+weights_refl_1) + L_1_Disc_Nat)* (np.sum(weights_2+weights_refl_2) + L_2_Disc_Nat))

    return Recomp
    
def Imcomp(freqs, prop_spectra, truelags, weights_1, weights_2, weights_refl_1, weights_refl_2,L_1_Disc_Nat,L_2_Disc_Nat, reTF, imTF, TF2, S_m,\
           D_DS, D_SH, i_DS, i_SH, weights_refl_base_1 = None, weights_refl_base_2 = None): 
    
    '''Given propagated power spectra (prop_spectra), compute the imaginary part of the apparent complex spectrum between two energy bands with direct weights (weights1, weights2),
        reflected weights (weights_refl_1, weights_refl_2), and constant disc flux contributions (L_1_Disc_Nat, L_2_Disc_Nat).'''    
                               
    Imcomp = np.zeros((len(freqs)), dtype ='complex')
    for k in range(N_r):
        Imcomp += prop_spectra[k] * imTF * (weights_1[k]*weights_refl_2[k] - weights_refl_1[k]*weights_2[k])
            
        if k ==0:
            pass
                
        else:
            if k < i_DS:
                for j in range(k):
                    proplag = np.sum(truelags[j:k])
                    Imcomp += ImCalc(j, k, freqs, prop_spectra, weights_1, weights_2, weights_refl_1, weights_refl_2, reTF, imTF, TF2, proplag, S_m, 1)

                        
            elif i_DS <= k < i_SH:
                for j in range(k):
                    proplag = np.sum(truelags[j:k])
                    if j < i_DS:
                        Imcomp += ImCalc(j, k, freqs, prop_spectra, weights_1, weights_2, weights_refl_1, weights_refl_2, reTF, imTF, TF2, proplag, S_m, D_DS)
   
                    else:
                        Imcomp += ImCalc(j, k, freqs, prop_spectra, weights_1, weights_2, weights_refl_1, weights_refl_2, reTF, imTF, TF2, proplag, S_m, 1.)
   
            else:
                for j in range(k):
                    proplag = np.sum(truelags[j:k])
                    if j < i_DS:
                        Imcomp += ImCalc(j, k, freqs, prop_spectra, weights_1, weights_2, weights_refl_1, weights_refl_2, reTF, imTF, TF2, proplag, S_m, D_DS*D_SH)
   
                    elif i_DS <= j < i_SH:
                        Imcomp += ImCalc(j, k, freqs, prop_spectra, weights_1, weights_2, weights_refl_1, weights_refl_2, reTF, imTF, TF2, proplag, S_m, D_SH)
   
                    else:
                        Imcomp += ImCalc(j, k, freqs, prop_spectra, weights_1, weights_2, weights_refl_1, weights_refl_2, reTF, imTF, TF2, proplag, S_m, 1.)

    if any(weights_refl_base_1==None):
        pass
    else:
        weights_refl_1 = weights_refl_base_1
   
    Imcomp = Imcomp / ( (np.sum(weights_1+weights_refl_1) +L_1_Disc_Nat)* (np.sum(weights_2+weights_refl_2) + L_2_Disc_Nat))
    
    return Imcomp



def outputs(Amp1, gamma, B_disc, m_disc, B_flow, m_flow, F_vard1, F_vardC, r_vard1,\
             wid1, r_o, r_i, D_DS, D_SH, S_m, r_disc, f_disc_V):
    '''The main function where we use the above defined subroutines to compute the power spectra, lag-frequency and lag-energy spectra of our data, producing the main plots found in the paper.'''
    
    
    
    '''Set up the central radii (rs) and thicknesses (drs) of the annuli in the modelled flow.'''
    r_bins = r_bounds(r_o, r_i, N_r)
    rs = np.asarray([(r_bins[i+1]+r_bins[i])/2 for i in range(len(r_bins)-1)])
    drs = [(r_bins[i] -r_bins[i+1]) for i in range(len(r_bins)-1)]    
    
    '''Set up the raw frequency range at which we will compute our power spectra and lag-frequency spectra.'''
    '''This variable (freqs) is updated several times to save on computations, but we keep the same variable name for conciseness.'''
    N_freq = 2 ** int(log2(M_len/dt_sim))
    freqs = abs(fourier.fftfreq(N_freq, dt_sim))[:N_freq/2+1]
    
    
    '''nustar fit SED data'''
    data_raw_NS = np.genfromtxt('spectral_model_fits/NuStar_unabs_model.qdp', skip_header = 3)
    
    E_raw_NS, dE_raw_NS, F_all_raw_NS, F_disc_raw_NS, F_soft_raw_NS, F_hard_raw_NS, F_refl_raw_soft_NS, F_refl_raw_hard_NS =\
        data_raw_NS[:,0], data_raw_NS[:,1], data_raw_NS[:,2], data_raw_NS[:,3], data_raw_NS[:,4], data_raw_NS[:,5], data_raw_NS[:,7], data_raw_NS[:,6]
    
    F_all_raw_NS, F_disc_raw_NS, F_soft_raw_NS, F_hard_raw_NS, F_refl_raw_soft_NS, F_refl_raw_hard_NS = \
        F_all_raw_NS/(E_raw_NS), F_disc_raw_NS/E_raw_NS, F_soft_raw_NS/(E_raw_NS),  F_hard_raw_NS/(E_raw_NS), F_refl_raw_soft_NS/(E_raw_NS), F_refl_raw_hard_NS/(E_raw_NS)
    
    Lo = np.sum(F_soft_raw_NS * 2*dE_raw_NS)
    Hi = np.sum(F_hard_raw_NS * 2*dE_raw_NS)
    Disc = np.sum(F_disc_raw_NS * 2*dE_raw_NS)
    
    L_rep_soft_NS = Lo * Omega_on2pi_soft  - np.sum(F_refl_raw_soft_NS*2*dE_raw_NS)
    L_rep_hard_NS = Hi * Omega_on2pi_hard - np.sum(F_refl_raw_hard_NS*2*dE_raw_NS)
    
    f_soft_rep = L_rep_soft_NS / (Disc)
    f_hard_rep = L_rep_hard_NS / (Disc)
    ##################################################################################################################
    
    
    '''XMM fit SED'''
    data_raw = np.genfromtxt('spectral_model_fits/XMM_unabs_model.qdp', skip_header = 3)
    data_abs = np.genfromtxt('spectral_model_fits/XMM_abs_model.qdp', skip_header = 3)
    X1s = data_abs[:,0]
    Y1s = data_abs[:,2]/data_raw[:,2]
    
    rmf_energs = np.genfromtxt('redistribution_matrix/rmf_energs')
    rmf_chanenergs =np.genfromtxt('redistribution_matrix/rmf_chanenergs')
    rmf_raws = np.load('redistribution_matrix/rmf_raws.npy')
    xx, yy = np.meshgrid(rmf_chanenergs, rmf_energs)
    rmf_func = interp.interp2d(rmf_chanenergs, rmf_energs, rmf_raws)
    
    E_raw, dE_raw, F_all_raw, F_disc_raw, F_soft_raw, F_hard_raw, F_refl_raw_soft, F_refl_raw_hard =\
        data_raw[:,0], data_raw[:,1], data_raw[:,2], data_raw[:,3], data_raw[:,4], data_raw[:,5], data_raw[:,7], data_raw[:,6]
    
    F_all_raw, F_disc_raw, F_soft_raw, F_hard_raw, F_refl_raw_soft, F_refl_raw_hard = \
                     F_all_raw/(E_raw), F_disc_raw/E_raw, F_soft_raw/(E_raw),  F_hard_raw/(E_raw), F_refl_raw_soft/(E_raw), F_refl_raw_hard/(E_raw)
    
    absorption = np.interp(E_raw, X1s, Y1s) # Compute the absorption of the original SED for every energy bin.
    rmfs = rmf_func(E_raw, E_raw)
    rmfs = np.transpose(rmfs)
    
    for i in range(len(rmfs[:,0])): # Compute the 2-D redistribution matrix function (details can be found in Ingram+ 2019), so that our model is wrapped around the same instrument response as the data.
        if not np.sum(rmfs[i])==0:
            rmfs[i] = rmfs[i] / np.sum(rmfs[i])
        
    
    '''Compute the raw flux from each component as observed by XMM for input into our model.
       This step ensures that our model deals with the energy spectrum in exactly the same way as the data.'''
    F_disc_raw = np.dot(F_disc_raw*absorption,rmfs)
    F_hard_raw = np.dot(F_hard_raw*absorption, rmfs)
    F_soft_raw = np.dot(F_soft_raw*absorption, rmfs)
    F_refl_raw_hard = np.dot(F_refl_raw_hard*absorption, rmfs)
    F_refl_raw_soft = np.dot(F_refl_raw_soft*absorption, rmfs)
    
    bins = np.logspace(-1, 0.8, 80)
    digitized = np.digitize(E_raw, bins)
    E_range = np.asarray([E_raw[digitized == i].mean() for i in range(1, len(bins))])
    dE = np.asarray([2*dE_raw[digitized == i].sum() for i in range(1, len(bins))])
    F_all = np.asarray([F_all_raw[digitized == i].mean() for i in range(1, len(bins))])
    F_disc = np.asarray([F_disc_raw[digitized == i].mean() for i in range(1, len(bins))])
    F_hard = np.asarray([F_hard_raw[digitized == i].mean() for i in range(1, len(bins))])
    F_soft = np.asarray([F_soft_raw[digitized == i].mean() for i in range(1, len(bins))])
    F_refl_soft = np.asarray([F_refl_raw_soft[digitized == i].mean() for i in range(1, len(bins))])
    F_refl_hard = np.asarray([F_refl_raw_hard[digitized == i].mean() for i in range(1, len(bins))])
    
    '''Discard any NaN values from arrays at response edges.'''
    dE = dE[np.logical_not(np.isnan(E_range))]
    F_all = F_all[np.logical_not(np.isnan(E_range))]
    F_disc = F_disc[np.logical_not(np.isnan(E_range))]
    F_hard = F_hard[np.logical_not(np.isnan(E_range))]
    F_soft = F_soft[np.logical_not(np.isnan(E_range))]
    F_refl_soft = F_refl_soft[np.logical_not(np.isnan(E_range))]
    F_refl_hard = F_refl_hard[np.logical_not(np.isnan(E_range))]
    E_range = E_range[np.logical_not(np.isnan(E_range))]
    
    
    '''The indices of those energies (in E_range) at the max and min limits of the energy bands of interest.'''
    i_Elow_min, i_Elow_max = min(np.argwhere(E_range > Elow_min))[0], min(np.argwhere(E_range > Elow_max))[0]
    i_Eint_min, i_Eint_max = min(np.argwhere(E_range > Eint_min))[0], min(np.argwhere(E_range > Eint_max))[0]
    i_Ehi_min, i_Ehi_max = min(np.argwhere(E_range > Ehi_min))[0], min(np.argwhere(E_range > Ehi_max))[0]
    
    '''The luminosities of all of our spectral components in the low energy band.'''
    L_Lo_Hard= np.sum((F_hard*dE)[i_Elow_min:i_Elow_max])
    L_Lo_Soft = np.sum((F_soft*dE)[i_Elow_min:i_Elow_max])
    L_Lo_Soft_Refl = np.sum((F_refl_soft*dE)[i_Elow_min:i_Elow_max])
    L_Lo_Hard_Refl = np.sum((F_refl_hard*dE)[i_Elow_min:i_Elow_max])
    L_Lo_Soft_Rep = np.sum((F_disc*dE)[i_Elow_min:i_Elow_max]) * f_soft_rep
    L_Lo_Hard_Rep = np.sum((F_disc*dE)[i_Elow_min:i_Elow_max]) * f_hard_rep
    L_Lo_Disc_V = np.sum((F_disc*dE)[i_Elow_min:i_Elow_max]) * f_disc_V                                         #The disc component between r_o and r_DS which has intrinsic variability in the low energy band.
    L_Lo_Disc_C = np.sum((F_disc*dE)[i_Elow_min:i_Elow_max]) - L_Lo_Soft_Rep - L_Lo_Hard_Rep - L_Lo_Disc_V      #The disc component beyond r_o which produces constant emission in the low energy band.
        
    
    '''The same as above for the intermediate band.'''    
    L_Int_Hard= np.sum((F_hard*dE)[i_Eint_min:i_Eint_max] )
    L_Int_Soft = np.sum((F_soft*dE)[i_Eint_min:i_Eint_max] )
    L_Int_Soft_Refl = np.sum((F_refl_soft*dE)[i_Eint_min:i_Eint_max])
    L_Int_Hard_Refl = np.sum((F_refl_hard*dE)[i_Eint_min:i_Eint_max])
    L_Int_Soft_Rep = np.sum((F_disc*dE)[i_Eint_min:i_Eint_max]) * f_soft_rep
    L_Int_Hard_Rep = np.sum((F_disc*dE)[i_Eint_min:i_Eint_max]) * f_hard_rep
    L_Int_Disc_V = np.sum((F_disc*dE)[i_Eint_min:i_Eint_max]) * f_disc_V                                        #The disc component between r_o and r_DS which has intrinsic variability in the intermediate energy band.
    L_Int_Disc_C = np.sum((F_disc*dE)[i_Eint_min:i_Eint_max]) - L_Int_Soft_Rep - L_Int_Hard_Rep- L_Int_Disc_V   #The disc component beyond r_o which produces constant emission in the intermediate energy band.
        
    
    '''The same as above for the high band.'''
    L_Hi_Hard= np.sum((F_hard*dE)[i_Ehi_min:i_Ehi_max] )
    L_Hi_Soft= np.sum((F_soft*dE)[i_Ehi_min:i_Ehi_max] )
    L_Hi_Soft_Refl= np.sum((F_refl_soft*dE)[i_Ehi_min:i_Ehi_max] )
    L_Hi_Hard_Refl = np.sum((F_refl_hard*dE)[i_Ehi_min:i_Ehi_max])
    L_Hi_Soft_Rep = np.sum((F_disc*dE)[i_Ehi_min:i_Ehi_max]) * f_soft_rep
    L_Hi_Hard_Rep = np.sum((F_disc*dE)[i_Ehi_min:i_Ehi_max]) * f_hard_rep
    L_Hi_Disc_V = np.sum((F_disc*dE)[i_Ehi_min:i_Ehi_max]) * f_disc_V
    L_Hi_Disc_C = np.sum((F_disc*dE)[i_Ehi_min:i_Ehi_max]) - L_Hi_Soft_Rep - L_Hi_Hard_Rep- L_Hi_Disc_V


    '''Set up the transfer function from the flow to the disc.'''
    TF_raw = transferfn(r_disc, r_o)
    TF = TFrebin(TF_raw, fbins = data_fbins)
    TF2 = np.conj(TF) * TF
    reTF = real(TF) # Split the transfer function into real and imaginary parts to be applied to the fourier space calculations later.
    imTF = imag(TF)


    '''Set up an array of the emissivities of each annulus.'''
    emissivities = np.zeros(N_r)
    for x in range(N_r):
        r = rs[x]
        dr = drs[x]
        emissivities[x] = emissivity(r, r_o,Amp1, gamma, r_vard1, wid1)

    '''Compute the spectral transition radii (r_DS, r_SH) such that equation 3 in MDD19'''
    '''is satisfied /as well as possible/ given the radial discretization.'''
    lumdiffs = np.zeros((len(rs), len(rs) )) + 9e20
    for r_DS in rs[1:-2]:
        i_DS = min(np.argwhere(rs < r_DS))[0]
        for r_SH in rs[i_DS:-1]:
            i_SH = min(np.argwhere(rs < r_SH))[0]
            eout = np.sum(emissivities[0:i_DS] * drs[0:i_DS])
            emid = np.sum(emissivities[i_DS:i_SH] * drs[i_DS:i_SH])
            ein = np.sum(emissivities[i_SH:] * drs[i_SH:])
            lumdiffs[i_DS, i_SH] = abs((eout/emid)/(f_disc_V*Disc/Lo))  + abs((emid/ein)/(Lo/Hi))
         
    minimum = np.unravel_index(lumdiffs.argmin(), lumdiffs.shape)
    r_DS = rs[minimum[0]-1]
    i_DS = min(np.argwhere(rs < r_DS))[0]
    r_SH =rs[minimum[1]-1]
    i_SH = min(np.argwhere(rs < r_SH))[0]
    print "rDS = {} and rSH = {}".format(r_DS, r_SH)
    print "iDS = {} and iSH = {}".format(i_DS, i_SH)
    
    '''Compute the emissivity weightings of each annulus relative to other annuli, independently of the energy band.'''
    weightsout = 0
    weightsmid = 0
    weightsin = 0
    weights = np.array(())
    for x in range(N_r):
        r = rs[x]
        dr = drs[x]
        emiss = emissivities[x]
        weights = np.append(weights, emiss * r * dr)        
        if x < i_DS:
            weightsout += emiss * r * dr
        elif i_DS <= x < i_SH:
            weightsmid +=  emiss * r * dr
        else:
            weightsin += emiss * r *dr
    
    weights[:i_DS] = weights[:i_DS] / weightsout
    weights[i_DS:i_SH] = weights[i_DS:i_SH] / weightsmid
    weights[i_SH:] = weights[i_SH:] / weightsin
    
    
    '''Compute the flux weightings of each annulus in the flow for the low energy band.'''
    weights_low = np.zeros(N_r)
    weights_low[:i_DS] = weights[:i_DS] * L_Lo_Disc_V
    weights_low[i_DS:i_SH] = weights[i_DS:i_SH] * L_Lo_Soft
    weights_low[i_SH:] = weights[i_SH:] * L_Lo_Hard

    '''The same as above for the intermediate band.'''
    weights_int = np.zeros(N_r)
    weights_int[:i_DS] = weights[:i_DS] * L_Int_Disc_V
    weights_int[i_DS:i_SH] = weights[i_DS:i_SH] * L_Int_Soft
    weights_int[i_SH:] = weights[i_SH:] * L_Int_Hard

    '''The same as above for the high band.'''
    weights_hi = np.zeros(N_r)
    weights_hi[:i_DS] = weights[:i_DS] * L_Hi_Disc_V
    weights_hi[i_DS:i_SH] = weights[i_DS:i_SH] * L_Hi_Soft
    weights_hi[i_SH:] = weights[i_SH:] * L_Hi_Hard

    '''The lightcurve weightings for the reflected components in each band.'''
    weights_refl_low = np.zeros(N_r)
    weights_refl_int = np.zeros(N_r)
    weights_refl_hi = np.zeros(N_r)
            
    weights_refl_low[:i_DS] = weights[:i_DS] * 0
    weights_refl_low[i_DS:i_SH] = weights[i_DS:i_SH] * (L_Lo_Soft_Refl + L_Lo_Soft_Rep)
    weights_refl_low[i_SH:] = weights[i_SH:] * (L_Lo_Hard_Refl+L_Lo_Hard_Rep)

    '''The same as above for the intermediate band.'''  
    weights_refl_int[:i_DS] = weights[:i_DS] * 0
    weights_refl_int[i_DS:i_SH] = weights[i_DS:i_SH] * (L_Int_Soft_Refl + L_Int_Soft_Rep)
    weights_refl_int[i_SH:] = weights[i_SH:] * (L_Int_Hard_Refl+L_Int_Hard_Rep)

    '''The same as above for the high band.'''
    weights_refl_hi[:i_DS] = weights[:i_DS] * 0
    weights_refl_hi[i_DS:i_SH] = weights[i_DS:i_SH] * (L_Hi_Soft_Refl + L_Hi_Soft_Rep)
    weights_refl_hi[i_SH:] = weights[i_SH:] * (L_Hi_Hard_Refl+L_Hi_Hard_Rep)
    
    
    '''Here we compute the time delays between adjacent annuli (truelags).'''
    lag_times = np.array(())
    truelags = np.zeros(N_r-1)
    for i in range(N_r):
        r = rs[i]
        dr = drs[i]
        if i <i_DS:
            v_r = r * f_alpha(r, B_disc, m_disc) # Propagation velocity between annuli at r.
        else:
            v_r = r * f_alpha(r, B_flow, m_flow)
            
        t_lag = dr / v_r
        lag_times = np.append(lag_times, t_lag)

    for j in range(1, len(lag_times)):            
            truelags[j-1] = (lag_times[j-1]/2 + lag_times[j]/2)


    '''Here we compute the power spectra of the /unpropagated/ mass accretion rates, mdot(r) (unprop_spectra).'''
    unprop_spectra = real(lorentzian(rs[0], drs[0], 0, freqs, i_DS, i_SH, r_o, r_i, B_disc, m_disc, B_flow, m_flow, dt_sim, F_vard1, F_vardC, r_vard1, wid1))
    unprop_spectra[0] = 0.
    for i in range(1, N_r):
        input_generator = real(lorentzian(rs[i], drs[i], i, freqs, i_DS, i_SH, r_o, r_i, B_disc, m_disc, B_flow, m_flow, dt_sim, F_vard1, F_vardC, r_vard1, wid1))
        input_generator[0] = 0.
        unprop_spectra = np.vstack((unprop_spectra, input_generator))
    
    
    '''Here we compute the power spectra of the /propagated/ mass accretion rates, Mdot(r) (prop_spectra).'''
    prop_spectra = unprop_spectra[0]
    a_data = unprop_spectra[0] * np.exp(-2*S_m*truelags[0]*freqs)
    b_data = unprop_spectra[1]

    if i_DS == 1:
        a_data[:] = a_data[:] / D_DS**2
    a_data = fourier.ifft(a_data) *1/(2*dt_sim)
    b_data = fourier.ifft(b_data) *1/(2*dt_sim)
    c_data = fourier.fft(a_data * b_data)
    c_data = c_data / (N_freq**2*dt_sim)
    c_data[0] = 1.
    c_data += unprop_spectra[0] + unprop_spectra[1]
    prop_spectra = np.vstack( (prop_spectra, c_data) )
    
    for i in range(2, N_r):
        a_data = prop_spectra[-1] * np.exp(-2*S_m*truelags[i-1]*freqs)
        a_data[0] =0.
        b_data = unprop_spectra[i]
        if i == i_DS:
            a_data = a_data / D_DS**2
            a_data = fourier.ifft(a_data) *1/(2*dt_sim)
            b_data = fourier.ifft(b_data) *1/(2*dt_sim)
            c_data = fourier.fft(a_data * b_data)
            c_data = c_data / (N_freq**2*dt_sim)
            c_data += prop_spectra[-1]/D_DS**2 * np.exp(-2*S_m*truelags[i-1]*freqs) + unprop_spectra[i]
            c_data[0] = 1.
        elif i == i_SH:
            a_data = a_data / D_SH**2
            a_data = fourier.ifft(a_data) *1/(2*dt_sim)
            b_data = fourier.ifft(b_data) *1/(2*dt_sim)
            c_data = fourier.fft(a_data * b_data)
            c_data = c_data / (N_freq**2*dt_sim)
            c_data += prop_spectra[-1]/D_SH**2  * np.exp(-2*S_m*truelags[i-1]*freqs) + unprop_spectra[i]
            c_data[0] = 1.
        else:            
            a_data = fourier.ifft(a_data) *1/(2*dt_sim)
            b_data = fourier.ifft(b_data) *1/(2*dt_sim)
            c_data = fourier.fft(a_data * b_data)
            c_data = c_data / (N_freq**2*dt_sim)
            c_data += prop_spectra[-1] * np.exp(-2*S_m*truelags[i-1]*freqs) + unprop_spectra[i]
            c_data[0] = 1.
        
        prop_spectra = np.vstack((prop_spectra, c_data))

    freqs = freqs[1:]
    prop_spectra = real(prop_spectra[:, 1:N_freq/2+1])    
    prop_spectra = np.transpose(prop_spectra)
    
    
    '''Here we select the frequencies at which we will evaluate the final energy-dependent power spectra and lags (freqs_selected) before rebinning at the end.'''
    '''Computing those products (tot_low, etc., Recomps etc.) at every frequency in freqs would be computationally very expensive.'''
    '''We therefore retain only those frequencies (and associated prop_spectra values) at the limits of the frequency bins, and throw away values between the liminal frequencies.'''
    '''This yields prop_spectra_selected, which we replace the variable prop_spectra with for convenience.'''
    freqs_selected = np.array(())
    prop_spectra_selected = np.zeros(N_r)
    inds = np.digitize(freqs, data_fbins)  
    for i in range(1, inds[-1]+1):
        if i in inds:
            i_min = min(np.argwhere(inds == i))[0]
            i_max = max(np.argwhere(inds == i))[0]
            freqs_selected = np.append(freqs_selected, freqs[i_min])
            prop_spectra_selected = np.vstack((prop_spectra_selected, prop_spectra[i_min, :]))
            if i_min != i_max:
                freqs_selected = np.append(freqs_selected, freqs[i_max])
                prop_spectra_selected = np.vstack((prop_spectra_selected, prop_spectra[i_max, :]))
    
    prop_spectra = np.transpose(prop_spectra_selected)[:,1:]
    freqs = freqs_selected
    
    
    
    '''Compute the modelled rms-normalized power spectrum of the low band (tot_low).'''
    tot_low = np.zeros((len(freqs)), dtype ='complex')
    for k in range(N_r):
        tot_low += prop_spectra[k] * ( weights_low[k]**2 + 2 * weights_low[k] * weights_refl_low[k] * reTF + weights_refl_low[k]**2 * TF2)
        if k ==0:
            pass
        else:
            if k < i_DS:
                for j in range(k):
                    proplag = np.sum(truelags[j:k])
                    tot_low += PSDCalc(j, k, freqs, prop_spectra, weights_low, weights_refl_low, reTF, imTF, TF2, proplag, S_m, 1.)
            elif i_DS <= k < i_SH:                  
                for j in range(k):
                    proplag = np.sum(truelags[j:k])                    
                    if j < i_DS:
                        tot_low += PSDCalc(j, k, freqs, prop_spectra, weights_low, weights_refl_low, reTF, imTF, TF2, proplag, S_m, D_DS)
                    else:
                        tot_low += PSDCalc(j, k, freqs, prop_spectra, weights_low, weights_refl_low, reTF, imTF, TF2, proplag, S_m, 1.)
            else:                  
                for j in range(k):
                    proplag = np.sum(truelags[j:k])                    
                    if j < i_DS:
                         tot_low += PSDCalc(j, k, freqs, prop_spectra, weights_low, weights_refl_low, reTF, imTF, TF2, proplag, S_m, D_DS*D_SH)
                    elif i_DS <= j < i_SH:
                         tot_low += PSDCalc(j, k, freqs, prop_spectra, weights_low, weights_refl_low, reTF, imTF, TF2, proplag, S_m, D_SH)
                    else:
                         tot_low += PSDCalc(j, k, freqs, prop_spectra, weights_low, weights_refl_low, reTF, imTF, TF2, proplag, S_m, 1.)
    tot_low = tot_low / (np.sum(weights_low+weights_refl_low) + L_Lo_Disc_C)**2
    
    
    '''Compute the modelled rms-normalized power spectrum of the intermediate band (tot_int).'''                         
    tot_int = np.zeros((len(freqs)), dtype ='complex')
    for k in range(N_r):
        tot_int += prop_spectra[k] * ( weights_int[k]**2 + 2 * weights_int[k] * weights_refl_int[k] * reTF + weights_refl_int[k]**2 * TF2)
        if k ==0:
            pass
        else:
            if k < i_DS:
                for j in range(k):
                    proplag = np.sum(truelags[j:k])
                    tot_int += PSDCalc(j, k, freqs, prop_spectra, weights_int, weights_refl_int, reTF, imTF, TF2, proplag, S_m, 1.)
            elif i_DS <= k < i_SH:                  
                for j in range(k):
                    proplag = np.sum(truelags[j:k])                    
                    if j < i_DS:
                        tot_int += PSDCalc(j, k, freqs, prop_spectra, weights_int, weights_refl_int, reTF, imTF, TF2, proplag, S_m, D_DS) 
                    else:
                        tot_int += PSDCalc(j, k, freqs, prop_spectra, weights_int, weights_refl_int, reTF, imTF, TF2, proplag, S_m, 1.)   
            else:                  
                for j in range(k):
                    proplag = np.sum(truelags[j:k])                    
                    if j < i_DS:
                         tot_int += PSDCalc(j, k, freqs, prop_spectra, weights_int, weights_refl_int, reTF, imTF, TF2, proplag, S_m, D_DS*D_SH)
                    elif i_DS <= j < i_SH:
                         tot_int += PSDCalc(j, k, freqs, prop_spectra, weights_int, weights_refl_int, reTF, imTF, TF2, proplag, S_m, D_SH)
                    else:
                         tot_int += PSDCalc(j, k, freqs, prop_spectra, weights_int, weights_refl_int, reTF, imTF, TF2, proplag, S_m, 1.)
    tot_int = tot_int / (np.sum(weights_int+weights_refl_int)+L_Int_Disc_C)**2  
    
    
    '''Compute the modelled rms-normalized power spectrum of the high band (tot_hi).'''                                 
    tot_hi = np.zeros((len(freqs)), dtype ='complex')
    for k in range(N_r):
        tot_hi += prop_spectra[k] * ( weights_hi[k]**2 + 2 * weights_hi[k] * weights_refl_hi[k] * reTF + weights_refl_hi[k]**2 * TF2)
        if k ==0:
            pass
        else:
            if k < i_DS:
                for j in range(k):
                    proplag = np.sum(truelags[j:k])
                    tot_hi += PSDCalc(j, k, freqs, prop_spectra, weights_hi, weights_refl_hi, reTF, imTF, TF2, proplag, S_m, 1.)
            elif i_DS <= k < i_SH:                  
                for j in range(k):
                    proplag = np.sum(truelags[j:k])                    
                    if j < i_DS:
                        tot_hi += PSDCalc(j, k, freqs, prop_spectra, weights_hi, weights_refl_hi, reTF, imTF, TF2, proplag, S_m, D_DS)
                    else:
                        tot_hi += PSDCalc(j, k, freqs, prop_spectra, weights_hi, weights_refl_hi, reTF, imTF, TF2, proplag, S_m, 1.)         
            else:                  
                for j in range(k):
                    proplag = np.sum(truelags[j:k])                    
                    if j < i_DS:
                         tot_hi += PSDCalc(j, k, freqs, prop_spectra, weights_hi, weights_refl_hi, reTF, imTF, TF2, proplag, S_m, D_DS*D_SH)
                    elif i_DS <= j < i_SH:
                         tot_hi += PSDCalc(j, k, freqs, prop_spectra, weights_hi, weights_refl_hi, reTF, imTF, TF2, proplag, S_m, D_SH)
                    else:
                         tot_hi += PSDCalc(j, k, freqs, prop_spectra, weights_hi, weights_refl_hi, reTF, imTF, TF2, proplag, S_m, 1.)
                         
    tot_hi = tot_hi / (np.sum(weights_hi+weights_refl_hi)+L_Hi_Disc_C)**2


    '''Compute the real and imaginary components of the cross spectrum between the low and high bands, and then from these, the lag.'''
    Recomp_LH = Recomp(freqs, prop_spectra, truelags, weights_low, weights_hi, weights_refl_low, weights_refl_hi, L_Lo_Disc_C, L_Hi_Disc_C, reTF, imTF, TF2, S_m, D_DS, D_SH, i_DS, i_SH)
    Imcomp_LH = Imcomp(freqs,prop_spectra, truelags, weights_low, weights_hi, weights_refl_low, weights_refl_hi,L_Lo_Disc_C, L_Hi_Disc_C, reTF, imTF, TF2, S_m, D_DS, D_SH, i_DS, i_SH)
    lag_LH = np.arctan2(real(Imcomp_LH), real(Recomp_LH)) /(2*pi*freqs)

    '''Compute the real and imaginary components of the cross spectrum between the intermediate and high bands, and then from these, the lag.'''    
    Recomp_IH = Recomp(freqs,prop_spectra, truelags, weights_int, weights_hi, weights_refl_int, weights_refl_hi,L_Int_Disc_C, L_Hi_Disc_C, reTF, imTF, TF2, S_m, D_DS, D_SH, i_DS, i_SH)
    Imcomp_IH = Imcomp(freqs,prop_spectra, truelags, weights_int, weights_hi, weights_refl_int, weights_refl_hi,L_Int_Disc_C, L_Hi_Disc_C, reTF, imTF, TF2, S_m, D_DS, D_SH, i_DS, i_SH)
    lag_IH = np.arctan2(real(Imcomp_IH), real(Recomp_IH)) /(2*pi*freqs)

    '''Compute the real and imaginary components of the cross spectrum between the low and intermediate bands, and then from these, the lag.'''        
    Recomp_LI = Recomp(freqs,prop_spectra, truelags, weights_low, weights_int, weights_refl_low, weights_refl_int,L_Lo_Disc_C, L_Int_Disc_C, reTF, imTF, TF2, S_m, D_DS, D_SH, i_DS, i_SH)
    Imcomp_LI = Imcomp(freqs,prop_spectra, truelags, weights_low, weights_int, weights_refl_low, weights_refl_int,L_Lo_Disc_C, L_Int_Disc_C, reTF, imTF, TF2, S_m, D_DS, D_SH, i_DS, i_SH)     
    lag_LI = np.arctan2(real(Imcomp_LI), real(Recomp_LI)) /(2*pi*freqs)

    '''Rebin everything up so that it's on the same frequency grid as the data.'''
    tot_low_binned = np.array(())
    tot_int_binned = np.array(())
    tot_hi_binned = np.array(())
    lag_LH_binned = np.array(())
    lag_IH_binned = np.array(())
    lag_LI_binned = np.array(())
    freqs_binned = np.array(())
    inds = np.digitize(freqs, data_fbins)
    for i in range(1, inds[-1]):
        if i in inds:
            i_min, i_max = min(np.argwhere(inds == i))[0], max(np.argwhere(inds==i))[0]
            freqs_binned = np.append(freqs_binned, (freqs[i_min] + freqs[i_max])/2)
            tot_low_binned = np.append(tot_low_binned, np.average(tot_low[i_min:i_max+1]))
            tot_int_binned = np.append(tot_int_binned, np.average(tot_int[i_min:i_max+1]))
            tot_hi_binned = np.append(tot_hi_binned, np.average(tot_hi[i_min:i_max+1]))
            lag_LH_binned = np.append(lag_LH_binned, np.average(lag_LH[i_min:i_max+1]))
            lag_IH_binned = np.append(lag_IH_binned, np.average(lag_IH[i_min:i_max+1]))
            lag_LI_binned = np.append(lag_LI_binned, np.average(lag_LI[i_min:i_max+1]))


    #######################################################################################################################
    #++++++++++++++++++++++++++=Here we compute the lag-energy spectrum+++++++++++++++++++++++++++++++++++++++++++++++++++#
    #######################################################################################################################
    
    '''Restore the frequency grid to the original raw values as we will need to rebin/slice it differently for the lag-E spectrum.'''
    N_freq = 2 ** int(log2(M_len/dt_sim))
    freqs = abs(fourier.fftfreq(N_freq, dt_sim))[:N_freq/2+1]
    
    '''For the lag-energy spectrum, we re-compute the frequency grid as we need adequate sampling in the three frequency ranges of interest.'''   
    freqs_dat = abs(fourier.fftfreq(di_dat, dt_dat))[1:di_dat/2+1]
    data_fbins2 = np.linspace(freq_slo_min, freq_slo_max, NfBand)
    data_fbins2 = np.append(data_fbins2, np.linspace(freq_mid_min, freq_mid_max, NfBand))
    data_fbins2 = np.append(data_fbins2, np.linspace(freq_fas_min, freq_fas_max, NfBand))
    inds_dat = np.digitize(freqs_dat, data_fbins2)
    
    freqs_sim = abs(fourier.fftfreq(N_freq, dt_sim))[1:N_freq/2+1]
    inds_sim = np.digitize(freqs_sim, data_fbins2)
    
    to_be_deleted = []
    for i_d in range(inds_dat[-1]):
        if i_d not in inds_sim:
            to_be_deleted.append(i_d)
    data_fbins2 = np.delete(data_fbins2, to_be_deleted)
    
    '''Re-compute the transfer function and its derived quantities for the lag-E frequency-gridding.'''
    TF = TFrebin(TF_raw, fbins = data_fbins2)
    TF2 = np.conj(TF) * TF
    reTF = real(TF)
    imTF = imag(TF) 
    
    '''Extract and rebin the energy range for our lag-E spectra.'''
    data_raw = np.genfromtxt('lagenergy/O1_lagE_1_30Hz.dat', skip_header = 1)
    E_BDM, dE_BDM = data_raw[:,0], data_raw[:,1]
    
    bins = E_BDM - dE_BDM
    bins = np.append(bins, E_BDM[-1] + dE_BDM[-1])
    
    digitized = np.digitize(E_raw, bins)
    E_range2 = np.asarray([E_raw[digitized == i].mean() for i in range(1, len(bins))])
    dE = np.asarray([2*dE_raw[digitized == i].sum() for i in range(1, len(bins))])
    F_all = np.asarray([F_all_raw[digitized == i].mean() for i in range(1, len(bins))])
    F_disc = np.asarray([F_disc_raw[digitized == i].mean() for i in range(1, len(bins))])
    F_hard = np.asarray([F_hard_raw[digitized == i].mean() for i in range(1, len(bins))])
    F_soft = np.asarray([F_soft_raw[digitized == i].mean() for i in range(1, len(bins))])
    F_refl_soft = np.asarray([F_refl_raw_soft[digitized == i].mean() for i in range(1, len(bins))])
    F_refl_hard = np.asarray([F_refl_raw_hard[digitized == i].mean() for i in range(1, len(bins))])
    
    dE = dE[np.logical_not(np.isnan(E_range2))]
    F_all = F_all[np.logical_not(np.isnan(E_range2))]
    F_disc = F_disc[np.logical_not(np.isnan(E_range2))]
    F_hard = F_hard[np.logical_not(np.isnan(E_range2))]
    F_soft = F_soft[np.logical_not(np.isnan(E_range2))]
    F_refl_soft = F_refl_soft[np.logical_not(np.isnan(E_range2))]
    F_refl_hard = F_refl_hard[np.logical_not(np.isnan(E_range2))]
    E_range2 = E_range2[np.logical_not(np.isnan(E_range2))]
    
    i_ref_min, i_ref_max = min(np.argwhere(E_range2 >= 0.5))[0], len(E_range2)-1
    
    '''Compute the energy weights in each lag-E bin for processing through the fourier-space model.'''
    '''Produce weights for every energy band, and the reference band.'''
    L_Soft = (F_soft*dE)
    L_Hard = (F_hard*dE)        
    L_Soft_Refl = F_refl_soft*dE
    L_Hard_Refl = F_refl_hard*dE
    L_Soft_Rep =  F_disc*dE * f_soft_rep
    L_Hard_Rep =  F_disc*dE * f_hard_rep
    L_Soft_reference = - (F_soft*dE) + np.sum((F_soft*dE)[i_ref_min:i_ref_max])
    L_Hard_reference = - (F_hard*dE) + np.sum((F_hard*dE)[i_ref_min:i_ref_max])    
    L_Soft_Refl_reference = - F_refl_soft*dE + np.sum((F_refl_soft*dE)[i_ref_min:i_ref_max])
    L_Hard_Refl_reference = - F_refl_hard*dE + np.sum((F_refl_hard*dE)[i_ref_min:i_ref_max])    
    L_Soft_Rep_reference = - F_disc*dE * f_soft_rep + np.sum((F_disc*dE)[i_ref_min:i_ref_max])* f_soft_rep
    L_Hard_Rep_reference = - F_disc*dE * f_hard_rep + np.sum((F_disc*dE)[i_ref_min:i_ref_max])* f_hard_rep      

    L_Disc_V = (F_disc*dE) * f_disc_V   
    L_Disc_C = (F_disc*dE) * (1 - f_disc_V - f_soft_rep - f_hard_rep)
    L_Disc_V_reference = - (F_disc*dE) * f_disc_V + np.sum((F_disc*dE)[i_ref_min:i_ref_max]) * f_disc_V
    L_Disc_C_reference =  - (F_disc*dE) * (1 - f_disc_V - f_soft_rep - f_hard_rep) + np.sum((F_disc*dE)[i_ref_min:i_ref_max]) * ( 1 - f_disc_V - f_soft_rep - f_hard_rep)   
    
    weights_reference = np.zeros((len(E_range2), N_r))
    weights_refl_reference = np.zeros((len(E_range2), N_r))
    
    weights_energy = np.zeros((len(E_range2), N_r))
    weights_refl_energy = np.zeros((len(E_range2), N_r))
    
    for i in range(len(E_range2)):
        weights_energy[i, :i_DS] = weights[:i_DS] * L_Disc_V[i]
        weights_energy[i, i_DS:i_SH] = weights[i_DS:i_SH] * L_Soft[i]        
        weights_energy[i, i_SH:] = weights[i_SH:] * L_Hard[i]
        weights_refl_energy[i, :i_DS] = weights[:i_DS] * 0
        weights_refl_energy[i, i_DS:i_SH] = weights[i_DS:i_SH] * (L_Soft_Refl[i] + L_Soft_Rep[i])        
        weights_refl_energy[i, i_SH:] = weights[i_SH:] * (L_Hard_Refl[i]+L_Hard_Rep[i])
        
        weights_reference[i, :i_DS] = weights[:i_DS] * L_Disc_V_reference[i]
        weights_reference[i, i_DS:i_SH] = weights[i_DS:i_SH] * L_Soft_reference[i]
        weights_reference[i, i_SH:] = weights[i_SH:] * L_Hard_reference[i]
        weights_refl_reference[i, :i_DS] = weights[:i_DS] * 0
        weights_refl_reference[i, i_DS:i_SH] = weights[i_DS:i_SH] * (L_Soft_Refl_reference[i] + L_Soft_Rep_reference[i])
        weights_refl_reference[i, i_SH:] = weights[i_SH:] * (L_Hard_Refl_reference[i]+L_Hard_Rep_reference[i])        

    unprop_spectra = real(lorentzian(rs[0], drs[0], 0, freqs, i_DS, i_SH, r_o, r_i, B_disc, m_disc, B_flow, m_flow, dt_sim, F_vard1, F_vardC, r_vard1, wid1))
    unprop_spectra[0] = 0.
    for i in range(1, N_r):
        input_generator = real(lorentzian(rs[i], drs[i], i, freqs, i_DS,i_SH, r_o, r_i, B_disc, m_disc, B_flow, m_flow, dt_sim, F_vard1, F_vardC, r_vard1, wid1))
        input_generator[0] = 0.
        unprop_spectra = np.vstack((unprop_spectra, input_generator))
    
    prop_spectra = unprop_spectra[0]
    a_data = unprop_spectra[0] * np.exp(-2*S_m*truelags[0]*freqs)
    b_data = unprop_spectra[1]

    if i_DS == 1:
        a_data[:] = a_data[:] / D_DS**2
    a_data = fourier.ifft(a_data) *1/(2*dt_sim)
    b_data = fourier.ifft(b_data) *1/(2*dt_sim)
    c_data = fourier.fft(a_data * b_data)
    c_data = c_data / (N_freq**2*dt_sim)
    c_data[0] = 1.
    c_data += unprop_spectra[0] + unprop_spectra[1]
    prop_spectra = np.vstack( (prop_spectra, c_data) )
    
    for i in range(2, N_r):
        a_data = prop_spectra[-1] * np.exp(-2*S_m*truelags[i-1]*freqs)
        a_data[0] =0.
        b_data = unprop_spectra[i]
        if i == i_DS:
            a_data = a_data / D_DS**2
            a_data = fourier.ifft(a_data) *1/(2*dt_sim)
            b_data = fourier.ifft(b_data) *1/(2*dt_sim)
            c_data = fourier.fft(a_data * b_data)
            c_data = c_data / (N_freq**2*dt_sim)
            c_data += prop_spectra[-1]/D_DS**2 * np.exp(-2*S_m*truelags[i-1]*freqs) + unprop_spectra[i]
            c_data[0] = 1.
        elif i == i_SH:
            a_data = a_data / D_SH**2
            a_data = fourier.ifft(a_data) *1/(2*dt_sim)
            b_data = fourier.ifft(b_data) *1/(2*dt_sim)
            c_data = fourier.fft(a_data * b_data)
            c_data = c_data / (N_freq**2*dt_sim)
            c_data += prop_spectra[-1]/D_SH**2  * np.exp(-2*S_m*truelags[i-1]*freqs) + unprop_spectra[i]
            c_data[0] = 1.
        else:            
            a_data = fourier.ifft(a_data) *1/(2*dt_sim)
            b_data = fourier.ifft(b_data) *1/(2*dt_sim)
            c_data = fourier.fft(a_data * b_data)
            c_data = c_data / (N_freq**2*dt_sim)
            c_data += prop_spectra[-1] * np.exp(-2*S_m*truelags[i-1]*freqs) + unprop_spectra[i]
            c_data[0] = 1.
        
        prop_spectra = np.vstack((prop_spectra, c_data))

    prop_spectra = real(prop_spectra[:, 1:N_freq/2+1])
    freqs = freqs[1:]
    
    freqs_selected = np.array(())
    prop_spectra_selected = np.zeros(N_r)
    
    prop_spectra = np.transpose(prop_spectra)
    
    inds = np.digitize(freqs, data_fbins2)  
    
    for i in range(1, inds[-1]+1):
        if i in inds:
            i_min = min(np.argwhere(inds == i))[0]
            i_max = max(np.argwhere(inds == i))[0]
            freqs_selected = np.append(freqs_selected, freqs[i_min])
            prop_spectra_selected = np.vstack((prop_spectra_selected, prop_spectra[i_min, :]))
            if i_min != i_max:
                freqs_selected = np.append(freqs_selected, freqs[i_max])
                prop_spectra_selected = np.vstack((prop_spectra_selected, prop_spectra[i_max, :]))
    
    prop_spectra = np.transpose(prop_spectra_selected)[:,1:]
    freqs = freqs_selected
    
    lag_slo_energy = np.zeros(len(E_range2))
    dlag_slo_energy = np.zeros(len(E_range2))
    lag_mid_energy = np.zeros(len(E_range2))
    dlag_mid_energy = np.zeros(len(E_range2))
    lag_fas_energy = np.zeros(len(E_range2))
    dlag_fas_energy = np.zeros(len(E_range2))    
    
    i_freq_slo_min, i_freq_slo_max = min(np.argwhere(freqs >= 0.02))[0] , min(np.argwhere(freqs > 0.3))[0]
    i_freq_mid_min, i_freq_mid_max = min(np.argwhere(freqs >= 0.3))[0] , min(np.argwhere(freqs > 1.))[0]
    i_freq_fas_min, i_freq_fas_max = min(np.argwhere(freqs >= 1.))[0] , min(np.argwhere(freqs > 30.))[0]
    
    for i in range(len(E_range2)):
        Recomp_i = Recomp(freqs, prop_spectra, truelags, weights_reference[i], weights_energy[i], weights_refl_reference[i], weights_refl_energy[i],\
                          L_Disc_C_reference[i], L_Disc_C[i] , reTF, imTF, TF2, S_m, D_DS, D_SH, i_DS, i_SH)
        Imcomp_i = Imcomp(freqs, prop_spectra, truelags, weights_reference[i], weights_energy[i], weights_refl_reference[i], weights_refl_energy[i],\
                          L_Disc_C_reference[i] ,L_Disc_C[i] , reTF, imTF, TF2, S_m, D_DS, D_SH, i_DS, i_SH)
        lag_i = np.arctan2(real(Imcomp_i), real(Recomp_i)) /(2*pi*freqs)
        
        lag_slo_energy[i] = np.average(lag_i[i_freq_slo_min: i_freq_slo_max])
        dlag_slo_energy[i] = np.std(lag_i[i_freq_slo_min:i_freq_slo_max]) / sqrt(i_freq_slo_max-i_freq_slo_min-1.)
        
        lag_mid_energy[i] = np.average(lag_i[i_freq_mid_min: i_freq_mid_max])
        dlag_mid_energy[i] = np.std(lag_i[i_freq_mid_min:i_freq_mid_max]) / sqrt(i_freq_mid_max-i_freq_mid_min-1.)

        lag_fas_energy[i] = np.average(lag_i[i_freq_fas_min: i_freq_fas_max])
        dlag_fas_energy[i] = np.std(lag_i[i_freq_fas_min:i_freq_fas_max]) / sqrt(i_freq_fas_max-i_freq_fas_min-1.)        

    return tot_low_binned, tot_int_binned, tot_hi_binned, lag_LH_binned, lag_IH_binned, lag_LI_binned, lag_slo_energy, dlag_slo_energy, lag_mid_energy, dlag_mid_energy, lag_fas_energy, dlag_fas_energy, E_range2

Z = outputs(Amp1, gamma, B_disc, m_disc, B_flow, m_flow, F_vard1, F_vardC,r_vard1,\
             wid1, r_o, r_i, D_DS, D_SH, S_m, r_disc, f_disc_V)

tot_low, tot_int, tot_hi, lag_LH, lag_IH, lag_LI, lag_slo_energy, dlag_slo_energy, lag_mid_energy, dlag_mid_energy, lag_fas_energy, dlag_fas_energy, E_range2 =\
    Z[0], Z[1], Z[2], Z[3], Z[4], Z[5], Z[6], Z[7], Z[8], Z[9], Z[10], Z[11], Z[12]


'''Plot all results as shown in the paper!'''
fig = plt.figure(figsize=(3, 5))
gs = gridspec.GridSpec(5, 1)
ax3 = plt.subplot(gs[0:2, :])
ax4 = plt.subplot(gs[2, :])
ax5 = plt.subplot(gs[3, :])
ax6 = plt.subplot(gs[4, :])

ax3.plot(freqs_binned, freqs_binned*tot_low, c= 'r', marker='x')
ax3.plot(freqs_binned, freqs_binned*tot_int, c= 'g', marker='x')                   
ax3.plot(freqs_binned, freqs_binned*tot_hi, c= 'b', marker='x')
ax3.fill_between(freqs_binned, fP_s_obs_low - fdP_s_obs_low, fP_s_obs_low + fdP_s_obs_low, color = 'pink', alpha = 1, label = '3-5kev data')
ax3.fill_between(freqs_binned, fP_s_obs_int - fdP_s_obs_int, fP_s_obs_int + fdP_s_obs_int, color = 'green', alpha = 0.35, label = '10-20kev data')
ax3.fill_between(freqs_binned, fP_s_obs_hi - fdP_s_obs_hi, fP_s_obs_hi + fdP_s_obs_hi, color = 'blue', alpha = 0.35, label = '20-35kev data')
ax3.tick_params(axis = 'both', bottom='on', top='on', left='on', right='on', direction= 'in')
ax3.set_ylim(0.0002, 0.035)
ax3.set_xlim(0.045,40)
ax3.set_yscale('log')
ax3.set_xscale('log')
ax3.set_ylabel(r'$fP_f$ $([rms/mean]^2)$')

ax4.set_xlim(0.045,40)
ax4.set_ylabel(r'$Int-Low$ $lag$ $(s)$')
ax4.set_yscale('log')
ax4.set_xscale('log')
ax4.tick_params(axis = 'both', bottom='on', top='on', left='on', right='on', direction= 'in')
ax4.plot(freqs_binned, lag_LI, c= 'r')
ax4.errorbar(freqs_binned, taus_IL, dtaus_IL, fmt = 's', color = 'darkred', ecolor = 'darkred',capsize = 0)

ax5.set_xlim(0.045,40)
ax5.set_yscale('log')
ax5.set_xscale('log')
ax5.set_xscale('log')
ax5.set_ylabel(r'$High-Int$ $lag$ $(s)$')
ax5.tick_params(axis = 'both', bottom='on', top='on', left='on', right='on', direction= 'in')
ax5.errorbar(freqs_binned, taus_HI, dtaus_HI, fmt = 'D', color = 'darkgreen', ecolor = 'darkgreen',capsize = 0)
ax5.plot(freqs_binned, lag_IH, c= 'g')

ax6.set_xlim(0.045,40)
ax6.set_ylabel(r'$High-Low$ $lag$ $(s)$')
ax6.set_yscale('log')
ax6.set_xscale('log')
ax6.set_xlabel(r'$f$ $(Hz)$')
ax6.tick_params(axis = 'both', bottom='on', top='on', left='on', right='on', direction= 'in')
ax6.errorbar(freqs_binned, taus_HL, dtaus_HL, fmt = 'o', color = 'darkblue', ecolor = 'darkblue',capsize = 0)
ax6.plot(freqs_binned, lag_LH, c= 'b')



f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

ax1.set_xscale('log')
ax1.set_xlabel('E (keV)')
ax1.set_ylabel('lag(s)')
data_raw = np.genfromtxt('lagenergy/O1_lagE_002_03Hz.dat', skip_header = 1)
E_BDM, dE_BDM, slolag, dslolag = data_raw[:,0], data_raw[:,1], data_raw[:,2], data_raw[:,3]
ax1.errorbar(E_BDM, slolag, dslolag, xerr = dE_BDM, color = 'darkred', label = 'data',alpha = 0.5)
ax1.fill_between(E_BDM, lag_slo_energy-dlag_slo_energy, lag_slo_energy+dlag_slo_energy, color = 'red', alpha = 1., label = 'model')
ax1.tick_params(axis = 'both', bottom='on', top='on', left='on', right='on', direction= 'in')

ax2.set_xscale('log')
ax2.set_xlabel('E $(keV)$')
ax2.set_ylabel('lag(s)')
data_raw = np.genfromtxt('lagenergy/O1_lagE_03_1Hz.dat', skip_header = 1)
E_BDM, dE_BDM, midlag, dmidlag = data_raw[:,0], data_raw[:,1], data_raw[:,2], data_raw[:,3]
ax2.errorbar(E_BDM, midlag, dmidlag, xerr = dE_BDM, color = 'darkgreen', label = 'data',alpha = 1.)
ax2.fill_between(E_BDM, lag_mid_energy-dlag_mid_energy, lag_mid_energy+dlag_mid_energy, color = 'green', alpha = 0.5, label = 'model')
ax2.tick_params(axis = 'both', bottom='on', top='on', left='on', right='on', direction= 'in')

ax3.set_xscale('log')
ax3.set_xlabel('E $(keV)$')
ax3.set_ylabel('lag(s)')
data_raw = np.genfromtxt('lagenergy/O1_lagE_1_30Hz.dat', skip_header = 1)
E_BDM, dE_BDM, faslag, dfaslag = data_raw[:,0], data_raw[:,1], data_raw[:,2], data_raw[:,3]
ax3.errorbar(E_BDM, faslag, dfaslag, xerr = dE_BDM, color = 'darkblue', label = 'data',alpha = 1.)
ax3.fill_between(E_BDM, lag_fas_energy-dlag_fas_energy, lag_fas_energy+dlag_fas_energy, color = 'b', alpha = 0.5, label = 'model')
ax3.tick_params(axis = 'both', bottom='on', top='on', left='on', right='on', direction= 'in')
f.tight_layout()
