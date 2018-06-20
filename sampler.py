# RHMC sampler 
# Version: Final thesis
# Features: Transdimensional and fully implements RHMC
# Conventions:
# - All flux are in units of counts. The user only deals with magnitudes and the magitude to
# flux (in counts) conversion is done internally.
# - All object inputs are in the form (Nobjs, 3) with each row corresponding to an object 
# and its mag, x, y information. This is converted to three Nobjs-dimensional vectors internally.
# - The initial point is saved in 0-index position. If the user asks for N samples, then 
# the returned array cotains N+1 samples including the initial point.
# - Only allows running of a single chain.

from utils import *

class sampler(object):
	def __init__(self, save_traj = False, Nobjs_max = False, Nsteps=10, Niter=100,\
		dt=5e-1, g_xx=2., g_ff=2., g_ff2=2., use_prior=True, alpha=2., prob_moves = [1.0, 0.0, 0.0, 0.0], \
		K_split = 1., beta_a = 2., beta_b = 2.):
		"""
		Generate sampler object and sets placeholders for various parameters
		used throughout inference.

		Allocate memory for trajectory.
		"""
		# Placeholder for data
		self.D = None

		# Placeholder for model
		self.M = None

		# Default experimental set-up
		self.num_rows, self.num_cols, self.flux_to_count, self.PSF_FWHM_pix, \
			self.B_count, self.mB, self.f_lim, self.arcsec_to_pix = self.default_exp_setup()

		# Maximum number of objects allowed.
		self.Nobjs_max = Nobjs_max

		# Number of iterations/steps
		self.Niter = Niter
		self.Nsteps = Nsteps

		# ----- Allocate space for the entire trajectories.
		# No need to save momenta
		# Variables are saved in high performance friendly format. 
		self.q = np.zeros((Niter+1, Nsteps+1, 3, Nobjs_max)) # Channels: 0 - f, 1 - x, 2 - y.

		# Global time step
		self.dt = dt

		# Factors that appear in H computation.
		self.g_xx = g_xx # Affects the step size in all range.
		self.g_ff = g_ff # Affects the step size in the bright limit. The bigger the smaller the step size.
		self.g_ff2 = g_ff2 # Affects the step size in the faint limit. The bigger the smaller the step size.

		# Compute factors to be used repeatedly.
		self.compute_factors()

		# Image display range min and max.
		self.vmin = None
		self.vmax = None

		# Flux prior f**-alpha (alpha must be greater 1)
		self.use_prior = use_prior
		self.alpha = alpha

		# Split merge parameters
		self.K_split = K_split
		self.beta_a = beta_a
		self.beta_b = beta_b

		# Proposal type defined in a dictionary
		self.move_types = {0: "within", 1: "birth", 2: "death", 3: "split", 4: "merge"}

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
		PSF_FWHM_pix = PSF_FWHM_arcsec / arcsec_to_pix
		PSF_sigma = PSF_FWHM_arcsec
		gain = 4.62 # photo-electron counts to ADU
		ADU_to_flux = 0.00546689 # nanomaggies per ADU
		B_ADU = 179 # Background in ADU.
		B_count = B_ADU/gain
		flux_to_count = 1./(ADU_to_flux * gain) # Flux to count conversion

		#---- Default mag set up
		mB = 23 # Backgroud magnitude per pixel.
		B_count = mag2flux(mB) * flux_to_count 
		f_lim = mag2flux(mB) * flux_to_count # Limiting magnitude

		# Size of the image
		num_rows = num_cols = 32 # Pixel index goes from 0 to num_rows-1
		return num_rows, num_cols, flux_to_count, PSF_FWHM_pix, B_count, mB, f_lim, arcsec_to_pix

	def compute_factors(self):
		"""
		Compute constant factors that are useful RHMC_diag method.
		"""
		self.g0, self.g1, self.g2 = factors(self.num_rows, self.num_cols, self.num_rows/2., self.num_cols/2., self.PSF_FWHM_pix)
		return
