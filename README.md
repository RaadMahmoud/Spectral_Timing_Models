Here you'll find the code and data used to produce the timing results of Mahmoud, Done & De Marco 2018 (MNRAS; https://academic.oup.com/mnras/article/486/2/2137/5432362) for the low-mass X-ray binary GX 339-4. This builds on our earlier papers: Mahmoud & Done 2018a (https://academic.oup.com/mnras/article/473/2/2084/4157805) and Mahmoud & Done 2018b (https://academic.oup.com/mnras/article/480/3/4040/5067328).

Commenting is in progress (or certainly on the list). But please don't hesitate to send questions to mahmoud.raad@yahoo.co.uk!

Contained in the main .py file are:

-- Physical constants as global variables.

-- Global variables derived from the prior spectral fitting.

-- A collection of useful time series processing routines, all required for our results (apart from the function "Coherence"), but also easily applied to your own light curves.

-- Implementation of those routines to process the observed light curves into Fourier space power spectra and lag spectra.

-- Subroutines for the analytic modelling of the components of fourier power spectra and lag spectra.

-- The function "output" which calls those subroutines and produces the modelled power spectra, lag-frequency spectra and lag-energy spectra.

To reproduce the plots found in MDDM18, just download and run spectral_timing_model_MDDM18.py!


