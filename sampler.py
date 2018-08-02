"""
RHMC sampler 
Version: Final thesis
Features: Transdimensional and fully implements RHMC
Conventions:
- All flux are in units of counts. 
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
		# Placeholder for mock data and the underlying truth q0
		self.D = None
		self.q0 = None

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
		# For q and p, only save the initial points, which are samples from the previous.
		self.q = np.zeros((Niter+1, 3, Nobjs_max)) # Channels: 0 - f, 1 - x, 2 - y.
		self.p = np.zeros((Niter+1, 3, Nobjs_max))
		# For energies, save both initial and final: total Ec, V, and T.
		self.E = np.zeros((Niter+1, 2))
		self.V = np.zeros((Niter+1, 2)) # Potential
		self.T = np.zeros((Niter+1, 2))	# Kinetic
		self.N = np.zeros(Niter+1, dtype=int) # The total number of **objects** at the initial point.
		self.A = np.zeros(Niter, dtype=bool) # Was the proposal accepted?
		self.moves = np.zeros(Niter+1, dtype=int) # Record what sort of proposals were made.

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
		self.P_move = prob_moves

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
		f_max = mag2flux(15) * flux_to_count 

		# Size of the image
		num_rows = num_cols = 32 # Pixel index goes from 0 to num_rows-1

		return num_rows, num_cols, flux_to_count, PSF_FWHM_pix, B_count, mB, f_min, f_max, arcsec_to_pix

	def compute_factors(self):
		"""
		Compute constant factors that are useful RHMC_diag method.
		"""
		self.g0, self.g1, self.g2 = factors(self.N_rows, self.N_cols, self.N_rows/2., self.N_cols/2., self.PSF_FWHM_pix)
		return

	def gen_sample_true(self, Nsample):
		"""
		Given the sample Nsample, draw samples from the xyf-prior.
		"""
		f = gen_pow_law_sample(self.alpha, self.f_min, self.f_max, Nsample=Nsample)
		x = np.random.random(size=Nsample) * self.N_rows
		y = np.random.random(size=Nsample) * self.N_cols

		self.q0 = np.array([f, x, y])
		return

	def gen_sample_model(self, Nsample):
		"""
		Given the sample number Nsample, draw first model samples from the xyf-prior.
		"""
		f = gen_pow_law_sample(self.alpha, self.f_min, self.f_max, Nsample=Nsample)
		x = np.random.random(size=Nsample) * self.N_rows
		y = np.random.random(size=Nsample) * self.N_cols

		self.q[0, 0, :Nsample] = f
		self.q[0, 1, :Nsample] = x
		self.q[0, 2, :Nsample] = y
		self.N[0] = Nsample

		return		

	def gen_mock_data(self):
		"""
		Given the truth samples q0, generate the mock image.
		"""
		assert self.q0 is not None

		# Generate an image with background.
		data = np.ones((self.N_rows, self.N_cols), dtype=float) * self.B_count

		# Add one star at a time.
		for i in xrange(self.q0.shape[1]):
			fs, xs, ys = self.q0[:, i]
			data +=  fs * gauss_PSF(self.N_rows, self.N_cols, xs, ys, FWHM = self.PSF_FWHM_pix)

		# Poission realization D of the underlying truth D0
		self.D = poisson_realization(data)		

		return


	def display_data(self, show=True, save=None, figsize=(5, 5), num_ticks = 6, \
			vmin=None, vmax=None):
		"""
		Display the data image
		"""
		#---- Contrast
		# If the user does not provide the contrast
		if vmin is None:
			# then check whether there is contrast stored up. If not.
			if self.vmin is None:
				D_raveled = self.D.ravel()
				self.vmin = np.percentile(D_raveled, 0.)
				self.vmax = np.percentile(D_raveled, 90.)
			vmin = self.vmin
			vmax = self.vmax

		fig, ax = plt.subplots(1, figsize = figsize)
		ax.imshow(self.D,  interpolation="none", cmap="gray", vmin=vmin, vmax = vmax)
		yticks = ticker.MaxNLocator(num_ticks)
		xticks = ticker.MaxNLocator(num_ticks)		
		ax.yaxis.set_major_locator(yticks)
		ax.xaxis.set_major_locator(xticks)		
		if show:
			plt.show()
		if save is not None:
			plt.savefig(save, dpi=200, bbox_inches = "tight")
		plt.close()


	def gen_noise_profile(self, N_trial = 1000, sig_fac=10):
		"""
		Given the truth q0, obtain error profile.
		"""
		assert self.q0 is not None

		# Generate an image with background.
		truth = np.ones((self.N_rows, self.N_cols), dtype=float) * self.B_count

		# Add one star at a time.
		for i in xrange(self.q0.shape[1]):
			fs, xs, ys = self.q0[:, i]
			truth +=  fs * gauss_PSF(self.N_rows, self.N_cols, xs, ys, FWHM = self.PSF_FWHM_pix)

		res_list = []
		for _ in xrange(N_trial):
		    # Poission realization D of the underlying truth D0
		    res_list.append(poisson_realization(truth) - truth)
		res = np.vstack(res_list).ravel()

		sig = np.sqrt(self.B_count)
		bins = np.arange(-sig_fac * sig, sig_fac * sig, sig/5.)
		hist, _ = np.histogram(res, bins = bins, normed=True)

		bin_centers = (bins[1:] + bins[:-1])/2.

		self.hist_noise = hist
		self.centers_noise = bin_centers

		return

	def h_xy(self, f):
		"""
		Given flux of objects, return the covariance matrix elements 
		corresponding to the position variables.
		"""

		return self.rho_xy / ((f*self.g1)**-1 + self.B_count * (self.g2**(-1)) * (f**-2))

	def h_f(self, f):
		"""
		Given flux of objects, return the covariance matrix elements 
		corresponding to the position variables.
		"""
		return self.rho_f / (f + self.g0**-1 * self.B_count)		

	def sample_momentum(self, f):
		"""
		Given the flux of objects, generate the corresponding momentum.
		"""
		d = f.size # Dimension of the vector

		# Diagonal covariance matrix components for positions 
		h_xy = self.h_xy(f)
		px = np.sqrt(h_xy) * np.random.randn(d)
		py = np.sqrt(h_xy) * np.random.randn(d)

		# Diagonal covariance matrix components for flux
		h_f = self.h_f(f)
		pf = np.sqrt(h_f) * np.random.randn(d)

		return pf, px, py


	def Vq(self, f, x, y):
		"""
		The potential energy corresponding to the position variables.

		Negative log of the posterior pi(q).
		"""
		return 0

	def Tqp(self, f, x, y, pf, px, py):
		"""
		The potential energy corresponding to the position variables.

		Negative log of the momentum distribution pi(p|q).
		"""
		return 0

	def RHMC_single_step(self, f, x, y, pf, px, py, delta=1e-6, counter_max=1000):
		"""
		Perform single step RHMC.
		"""

		return f, x, y, pf, px, py

	def do_inference(self, delta=1e-6, counter_max = 1000, verbose=False):
		"""
		Perform inference based on the data generated by the truth sample
		with the generate initial model samples. 

		The arguments to this function are parameters for the RHMC 
		integrator. 
		"""
		# ---- Main iterations start here
		# Recall that our convention is to save the initial point corresponding 
		# to the iteration, which has already been done during the previous iteration.
		# The last iteration is i = Niter, which sets the initial point in index Niter+1 and termiantes.
		for i in xrange(self.Niter): 
			# ---- Set the initial position. Copy is a pre-caution.
			f_tmp = np.copy(self.q[i, 0, :self.N[i]])
			x_tmp = np.copy(self.q[i, 1, :self.N[i]])
			y_tmp = np.copy(self.q[i, 2, :self.N[i]])

			# ---- Resample momentum and save it
			pf_tmp, px_tmp, py_tmp = self.sample_momentum(f_tmp)
			self.p[i, 0, :self.N[i]] = pf_tmp
			self.p[i, 1, :self.N[i]] = px_tmp
			self.p[i, 2, :self.N[i]] = py_tmp
			# The initial q has already been set from the previous step.

			# ---- Compute the initial energies and record
			self.V[i, 0] = self.Vq(f_tmp, x_tmp, y_tmp)
			self.T[i, 0] = self.Tqp(f_tmp, x_tmp, y_tmp, pf_tmp, px_tmp, py_tmp)
			self.E[i, 0] = self.V[i, 0] + self.T[i, 0]

			# ---- Roll dice and choose which proposal to make.
			move_type = np.random.choice([0, 1, 2], p=self.P_move, size=1)[0]
			# Save the move type
			self.moves[i] = move_type

			# ---- Make the transition
			if move_type == 0: # Intra-model move
				#---- Looping over steps
				for l in xrange(self.Nsteps):
					f_tmp, x_tmp, y_tmp, pf_tmp, px_tmp, py_tmp =\
						self.RHMC_single_step(\
						f_tmp, x_tmp, y_tmp, pf_tmp, px_tmp, py_tmp, \
						delta = delta, counter_max = counter_max)
				factor = 0
			elif move_type == 1: # BD
				pass
			elif move_type == 2: # <S
				pass 

			# ---- Compute the final energies and record
			self.V[i, 1] = self.Vq(f_tmp, x_tmp, y_tmp)
			self.T[i, 1] = self.Tqp(f_tmp, x_tmp, y_tmp, pf_tmp, px_tmp, py_tmp)
			self.E[i, 1] = self.V[i, 1] + self.T[i, 1]

			# ---- Accept or reject the proposal and record
			dE = self.E[i, 1] - self.E[i, 0]
			ln_alpha0 = -dE + factor # factor are other factors in ln_alpha0 other than -dE.
			lnu = np.log(np.random.random(1))
			if (ln_alpha0 >= 0) or (lnu < ln_alpha0): # Accepted
				# Record acceptance
				self.A[i] = True

				# Save the final q as the initial point
				self.N[i+1] = f_tmp.size # This must be set first.
				self.q[i+1, 0, :self.N[i+1]] = f_tmp
				self.q[i+1, 1, :self.N[i+1]] = x_tmp
				self.q[i+1, 2, :self.N[i+1]] = y_tmp
			else:
				# Save the current iteration's initial point as the next initial point.
				self.N[i+1] = self.N[i]
				self.q[i+1, 0, :self.N[i+1]] = self.q[i, 0, :self.N[i]]
				self.q[i+1, 1, :self.N[i+1]] = self.q[i, 1, :self.N[i]]
				self.q[i+1, 2, :self.N[i+1]] = self.q[i, 2, :self.N[i]]

		# 	elif move_type == 1: # If birth or death
		# 		# Roll the dice to determine whether it's birth or death
		# 		# True - Birth and False - Death.
		# 		birth_death = np.random.choice([True, False], p=[0.5, 0.5])
				
		# 		# Save which type of move was proposed.
		# 		if birth_death: # True - Birth
		# 			self.move[l] = 1 
		# 		else:
		# 			self.move[l] = 2

		# 		# Initial: q0, p0 (Dim = N)
		# 		#---- RHMC steps
		# 		for i in xrange(1, self.Nsteps+1, 1):
		# 			q_tmp, p_tmp = self.RHMC_single_step(q_tmp, p_tmp, delta = delta, counter_max = counter_max)
		# 		p_tmp *= -1 
		# 		# After 1st RHMC: qL, -pL (Dim = N)

		# 		#---- Birth or death move
		# 		q_tmp, p_tmp, factor = self.birth_death_move(q_tmp, p_tmp, birth_death = birth_death)
		# 		# RJ move: q*L, -p*L (Dim = N +- 1)

		# 		#---- RHMC steps
		# 		# Perform RHMC integration.
		# 		for i in xrange(1, self.Nsteps+1, 1):
		# 			q_tmp, p_tmp = self.RHMC_single_step(q_tmp, p_tmp, delta = delta, counter_max = counter_max)
		# 		p_tmp *= -1 					
		# 		# After second RHMC: q*0, p*0 (Dim = N +- 1)
		
		# 	elif move_type == 2: # If it merge or split
		# 		# Roll the dice to determine whether it's merge or split
		# 		# True - Split and False - Merge
		# 		split_merge = np.random.choice([True, False], p=[0.5, 0.5])
				
		# 		# Save which type of move was proposed.
		# 		if split_merge: # True - Split
		# 			self.move[l] = 3
		# 		else:
		# 			self.move[l] = 4

		# 		# Initial: q0, p0 (Dim = N)
		# 		#---- RHMC steps
		# 		for i in xrange(1, self.Nsteps+1, 1):
		# 			q_tmp, p_tmp = self.RHMC_single_step(q_tmp, p_tmp, delta = delta, counter_max = counter_max)
		# 		p_tmp *= -1 
		# 		# After 1st RHMC: qL, -pL (Dim = N)

		# 		#---- Split or merge move
		# 		q_tmp, p_tmp, factor = self.split_merge_move(q_tmp, p_tmp, split_merge = split_merge)
		# 		# RJ move: q*L, -p*L (Dim = N +- 1)

		# 		#---- RHMC steps
		# 		# Perform RHMC integration.
		# 		for i in xrange(1, self.Nsteps+1, 1):
		# 			q_tmp, p_tmp = self.RHMC_single_step(q_tmp, p_tmp, delta = delta, counter_max = counter_max)
		# 		p_tmp *= -1 					
		# 		# After second RHMC: q*0, p*0 (Dim = N +- 1)


		return		