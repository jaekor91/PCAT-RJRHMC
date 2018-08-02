"""
RHMC sampler 
Version: Final thesis
Features: Transdimensional and fully implements RHMC
Conventions:
- All flux are in units of counts. The user only deals with magnitudes and the magitude to
flux (in counts) conversion is done internally.
- All object inputs are in the form (Nobjs, 3) with each row corresponding to an object 
and its mag, x, y information. This is converted to three Nobjs-dimensional vectors internally.
- The initial point is saved in 0-index position. If the user asks for N samples, then 
the returned array cotains N+1 samples including the initial point.
- Similarly, for each RHMC move, if Nsteps are asked for, then there are total of Nsteps+1 points
including the initial and final points.
- Only allows running of a single chain.
"""

from utils import *

class sampler(object):
	def __init__(self, Nobjs_max = 1000, Nsteps=10, Niter=100,\
		dt=5e-2, rho_xy=0.05, rho_f=4., alpha=2., prob_moves = [1.0, 0.0, 0.0], \
		K = 1., B_alpha = 2., B_beta = 2.):
		"""
		Generate sampler object and sets placeholders for various parameters
		used throughout inference.

		Args: The variable names are consistent with the ones introduced in the text.
		----
		- prob_moves: Since the inter-model moves are always considered as a pair, no need to use
		separate probabilities.
			- 0: Intra-model moves
			- 1: BD
			- 2: MS
		"""
		# Placeholder for data
		self.D = None

		# Placeholder for model
		self.M = None

		# Default "experimental" set-up
		self.N_rows, self.N_cols, self.flux_to_count, self.PSF_FWHM_pix, \
			self.B_count, self.mB, self.f_min, self.f_max, self.arcsec_to_pix = self.default_exp_setup()

		# Maximum number of objects allowed.
		self.Nobjs_max = Nobjs_max

		# Number of iterations/steps
		self.Niter = Niter
		self.Nsteps = Nsteps

		# ----- Allocate space for the entire trajectories.
		# No need to save momenta
		# Variables are saved in high performance friendly format. 
		self.q = np.zeros((Niter+1, 3, Nobjs_max)) # Channels: 0 - f, 1 - x, 2 - y.
		self.p = np.zeros((Niter+1, 3, Nobjs_max))
		self.E = np.zeros((Niter+1, 2))
		self.N = np.zeros(Niter+1, dtype=int)# The total number of **objects** at a particular point in the inference.

		# Global time step
		self.dt = dt

		# Factors that appear in H computation.
		self.rho_xy = rho_xy # Affects the step size in all range.
		self.rho_f = rho_f # Affects the step size in the bright limit. The bigger the smaller the step size.

		# Compute factors to be used repeatedly.
		self.compute_factors()

		# Image display range min and max.
		self.vmin = None
		self.vmax = None

		# Flux prior f**-alpha (alpha must be greater 1)
		self.alpha = alpha
		assert self.alpha >= 1

		# Split merge parameters
		self.K = K # The gaussian width for dr
		self.B_alpha = B_alpha
		self.B_beta = B_beta

		# Proposal type defined in a dictionary
		self.move_types = {0: "intra", 1: "birth", 2: "death", 3: "split", 4: "merge"}

		return

	def default_exp_setup(self):
		#---- A note on conversion
		# From Stephen: If you want to replicate the SDSS image of M2, you could use:
		# 0.4 arcsec per pixel, seeing of 1.4 arcsec
		# Background of 179 ADU per pixel, gain of 4.62 (so background of 179/4.62 = 38.7 photoelectrons per pixel)
		# 0.00546689 nanomaggies per ADU (ie. 183 ADU = 22.5 magnitude; see mag2flux(22.5) / 0.00546689)
		# Interpretation: Measurement goes like: photo-electron counts per pixel ---> ADU ---> nanomaggies.
		# The first conversion is called gain.
		# The second conversion is ADU to flux.

		# Flux to counts conversion
		# flux_to_count = 1./(ADU_to_flux * gain)

		#---- Global parameters
		arcsec_to_pix = 0.4
		PSF_FWHM_arcsec = 1.4
		PSF_FWHM_pix = PSF_FWHM_arcsec / arcsec_to_pix # The quantity used.
		PSF_sigma = PSF_FWHM_arcsec
		gain = 4.62 # photo-electron counts to ADU
		ADU_to_flux = 0.00546689 # nanomaggies per ADU
		B_ADU = 179 # Background in ADU.
		B_count = B_ADU/gain
		flux_to_count = 1./(ADU_to_flux * gain) # Flux to count conversion

		#---- Default mag set up
		mB = 23 # Backgroud magnitude per pixel.
		B_count = mag2flux(mB) * flux_to_count 
		f_min = mag2flux(mB) * flux_to_count # Limiting magnitude
		f_max = mag2flux(15)

		# Size of the image
		num_rows = num_cols = 32 # Pixel index goes from 0 to num_rows-1

		return num_rows, num_cols, flux_to_count, PSF_FWHM_pix, B_count, mB, f_min, f_max, arcsec_to_pix

	def compute_factors(self):
		"""
		Compute constant factors that are useful RHMC_diag method.
		"""
		self.g0, self.g1, self.g2 = factors(self.N_rows, self.N_cols, self.N_rows/2., self.N_cols/2., self.PSF_FWHM_pix)
		return
