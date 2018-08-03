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
		self.p = np.zeros((Niter, 3, Nobjs_max))
		# For energies, save both initial and final: total Ec, V, and T.
		self.E = np.zeros((Niter, 2))
		self.V = np.zeros((Niter, 2)) # Potential
		self.T = np.zeros((Niter, 2))	# Kinetic
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
		# Prior factor (per object) often used. Combines the constant log factors from both position and flux priors.
		C_f = (1-alpha) / ((self.f_max/self.flux_to_count)**(1-alpha) - (self.f_min/self.flux_to_count)**(1-alpha)) 
		self.ln_C_prior = np.log(C_f) - np.log(self.N_rows * self.N_cols) 

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
		f_min = mag2flux(mB-2) * flux_to_count # Limiting magnitude
		f_max = mag2flux(13) * flux_to_count 

		# Size of the image
		N_rows = N_cols = 32 # Pixel index goes from 0 to N_rows-1

		return N_rows, N_cols, flux_to_count, PSF_FWHM_pix, B_count, mB, f_min, f_max, arcsec_to_pix

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
		f = gen_pow_law_sample(self.alpha, self.f_min/self.flux_to_count, \
			self.f_max/self.flux_to_count, Nsample=Nsample) * self.flux_to_count
		x = np.random.random(size=Nsample) * (self.N_rows-1)
		y = np.random.random(size=Nsample) * (self.N_cols-1)

		self.q0 = np.array([f, x, y])
		return

	def gen_sample_model(self, Nsample):
		"""
		Given the sample number Nsample, draw first model samples from the xyf-prior.
		"""
		f = gen_pow_law_sample(self.alpha, self.f_min/self.flux_to_count, \
			self.f_max/self.flux_to_count, Nsample=Nsample) * self.flux_to_count
		x = np.random.random(size=Nsample) * (self.N_rows-1)
		y = np.random.random(size=Nsample) * (self.N_cols-1)

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
		if self.vmin is None:
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
		if save is not None:
			plt.savefig(save, dpi=200, bbox_inches = "tight")
		if show:
			plt.show()			
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


	def gen_model_image(self, f, x, y):
		"""
		Given the model sample parameters, generate the model image.
		"""
		# Generate an image with background.
		model= np.ones((self.N_rows, self.N_cols), dtype=float) * self.B_count

		# Add one star at a time.
		for s in xrange(f.size):
			model +=  f[s] * gauss_PSF(self.N_rows, self.N_cols, x[s], y[s], FWHM = self.PSF_FWHM_pix)

		return model

	def Vq(self, f, x, y):
		"""
		The potential energy corresponding to the position variables.

		Negative log of the posterior pi(q):
		-ln pi(q) = -(poisson log likelihood) - (log priors)
		log priors = -alpha * sum_s(ln_f_s) + Ns * [ln(C_f) - ln(N_rows * N_cols)]
		where Ns is the number of stars
		"""
		# Number of stars
		Ns = f.size

		# Generate model image
		model = self.gen_model_image(f, x, y)

		# Compute the poisson log likelihood
		ln_poisson = np.sum(self.D * np.log(model) - model)

		# Compute the log of prior. Note that it is important to keep track of the right 
		# number of constants to be used.
		ln_prior = -self.alpha * np.sum(np.log(f)) + Ns * self.ln_C_prior

		return -ln_poisson - ln_prior

	def Tqp(self, f, pf, px, py):
		"""
		The potential energy corresponding to the position variables.

		Negative log of the momentum distribution pi(p|q).
		"""
		# Compute h matrix
		h_xy = self.h_xy(f)
		h_f = self.h_f(f)		

		# Compute the term corresponding to the argument of the exponential
		term1 = np.sum(pf**2 / h_f) + np.sum(px**2 / h_xy) + np.sum(py**2 / h_xy)

		# Comput the term corresponding to the log determinant
		term2 = np.sum(2 * np.log(h_xy) + np.log(h_f))

		return (term1 + term2)/2. # No need to negate

	def dh_xydf(self, f):
		"""
		Just the derivate of h_xy function.
		"""
		grad = self.rho_xy * \
			(1./(self.g1 * f**2) + 2 * self.B_count/(self.g2 * f**3)) * \
			((f*self.g1)**-1 + self.B_count * (self.g2**(-1)) * (f**-2))**-2
		# Note that negative sign cancels.

		# Thresholding if flux outside the proper bounds
		# This derives from the fact that the potential is constant there.
		grad[f < self.f_min] = 0
		grad[f > self.f_max] = 0		

		return grad

	def dh_fdf(self, f):
		"""
		Simply the gradient of h_f
		"""
		return -self.rho_f / (f + self.g0**-1 * self.B_count)**2

	def dVdq(self, f, x, y):
		"""
		Gradient of the negative log posterior.
		"""

		# Compute the current model.
		Lambda = self.gen_model_image(f, x, y)

		# Variable to be recycled
		rho = (self.D/Lambda)-1.# (D_lm/Lambda_lm - 1)
		# Compute f, x, y gradient for each object
		lv = np.arange(0, self.N_rows)
		mv = np.arange(0, self.N_cols)
		mv, lv = np.meshgrid(lv, mv)
		var = (self.PSF_FWHM_pix/2.354)**2 

		# Place holder for gradient variables. Because the gradient has to be 
		# computed individual over the entire data it would be hard to make this vectorized
		dVdf = np.zeros(f.size)
		dVdx = np.zeros(f.size)
		dVdy = np.zeros(f.size)		
		for s in range(f.size):
			fs, xs, ys = f[s], x[s], y[s]
			PSF = gauss_PSF(self.N_rows, self.N_cols, xs, ys, FWHM=self.PSF_FWHM_pix)
			dVdf[s] = -np.sum(rho * PSF) + self.alpha / fs # Note that there is always the prior.
			dVdx[s] = -np.sum(rho * (lv - xs + 0.5) * PSF) * fs / var
			dVdy[s] = -np.sum(rho * (mv - ys + 0.5) * PSF) * fs / var

		return dVdf, dVdx, dVdy

	def dphidq(self, f, x, y):
		"""
		As in the general metric paper.
		"""
		# Gradient contribution
		dVdf, dVdx, dVdy = self.dVdq(f, x, y)

		# Compute h matrix
		h_xy = self.h_xy(f)
		h_f = self.h_f(f)

		# Compute the gradient of h matrix
		dh_xydf = self.dh_xydf(f)
		dh_fdf = self.dh_fdf(f)

		# For each object compute the gradient
		grad_logdet_f = (2 * dh_xydf / h_xy) + (dh_fdf / h_f)		

		return dVdf+grad_logdet_f, dVdx, dVdy

	def dtaudf(self, f, pf, px, py):
		"""
		As in the general metric paper.
		All the other ones evalute to zero.
		"""
		# Compute h matrix
		h_xy = self.h_xy(f)
		h_f = self.h_f(f)

		# Compute the gradient of h matrix
		dh_xydf = self.dh_xydf(f)
		dh_fdf = self.dh_fdf(f)

		dtaudf = -0.5 * ((px**2 + py**2) * dh_xydf / h_xy**2 + pf**2 * dh_fdf / h_f**2)

		return dtaudf

	def dtaudp(self, f, pf, px, py):
		# Compute h matrix
		h_xy = self.h_xy(f)
		h_f = self.h_f(f)

		return pf/h_f, px/h_xy, py/h_xy

	def RHMC_single_step(self, f, x, y, pf, px, py, delta=1e-6, counter_max=1000):
		"""
		Perform single step RHMC.
		"""
		# First update phi-hat
		dphidf, dphidx, dphidy = self.dphidq(f, x, y)
		pf = pf - (self.dt/2.) * dphidf
		px = px - (self.dt/2.) * dphidx
		py = py - (self.dt/2.) * dphidy

		# p-tau update
		rho_f = np.copy(pf)
		rho_x = np.copy(px)
		rho_y = np.copy(py)		
		dpf = np.infty
		dpx = np.infty
		dpy = np.infty
		dp = max(dpf, dpx, dpy)
		counter = 0
		while (dp > delta) and (counter < counter_max):
			dtaudf = self.dtaudf(f, pf, px, py) 
			pf_prime = rho_f - (self.dt/2.) * dtaudf
			px_prime = np.copy(rho_x)
			py_prime = np.copy(rho_y)
			# Determine dp max 			
			dpf = np.max(np.abs(pf - pf_prime))
			dpx = np.max(np.abs(px - px_prime))
			dpy = np.max(np.abs(py - py_prime))
			dp = max(dpf, dpx, dpy)
			# Copy 
			pf = np.copy(pf_prime)
			px = np.copy(px_prime)
			py = np.copy(py_prime)			
			counter +=1

		# q-tau update
		sig_f = np.copy(f)
		sig_x = np.copy(x)
		sig_y = np.copy(y)
		df = np.infty
		dx = np.infty
		dy = np.infty
		dq = max(df, dx, dy)
		counter = 0				
		while (dq > delta) and (counter < counter_max):
			dtaudpf1, dtaudpx1, dtaudpy1 = self.dtaudp(sig_f, pf, px, py)
			dtaudpf2, dtaudpx2, dtaudpy2 = self.dtaudp(f, pf, px, py)
			f_prime = sig_f + (self.dt/2.) * (dtaudpf1 + dtaudpf2)
			x_prime = sig_x + (self.dt/2.) * (dtaudpx1 + dtaudpx2)
			y_prime = sig_y + (self.dt/2.) * (dtaudpy1 + dtaudpy2)
			# Compute the max difference
			df = np.max(np.abs(f - f_prime))
			dx = np.max(np.abs(x - x_prime))
			dy = np.max(np.abs(y - y_prime))
			dq = max(df, dx, dy)
			# Copy
			f = np.copy(f_prime)
			x = np.copy(x_prime)
			y = np.copy(y_prime)			
			counter +=1					

		# p-tau update
		dtaudf = self.dtaudf(f, pf, px, py) 
		pf = pf - (self.dt/2.) * dtaudf

		# Last update phi-hat
		dphidf, dphidx, dphidy = self.dphidq(f, x, y)
		pf = pf - (self.dt/2.) * dphidf
		px = px - (self.dt/2.) * dphidx
		py = py - (self.dt/2.) * dphidy

		# Boundary condition checks
		pf[(f < self.f_min)] *= -1
		pf[(f > self.f_max)] *= -1
		px[x < 0] *= -1
		px[x > (self.N_rows-1)] *= -1
		py[y < 0] *= -1
		py[y > (self.N_rows-1)] *= -1

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
			self.T[i, 0] = self.Tqp(f_tmp, pf_tmp, px_tmp, py_tmp)
			self.E[i, 0] = self.V[i, 0] + self.T[i, 0]

			# ---- Roll dice and choose which proposal to make.
			move_type = np.random.choice([0, 1, 2], p=self.P_move, size=1)[0]

			# ---- Make the transition
			if move_type == 0: # Intra-model move
				# Save the move type
				self.moves[i] = 0

				#---- Looping over steps
				for l in xrange(self.Nsteps):
					f_tmp, x_tmp, y_tmp, pf_tmp, px_tmp, py_tmp =\
						self.RHMC_single_step(f_tmp, x_tmp, y_tmp, pf_tmp, px_tmp, py_tmp, \
											delta = delta, counter_max = counter_max)
				factor = 0
			elif move_type == 1: # BD
				# Roll the dice to determine whether it's birth or death
				# True - Birth and False - Death.
				birth_death = np.random.choice([True, False], p=[0.5, 0.5])
				
				# Save which type of move was proposed.
				if birth_death: # True - Birth
					self.moves[i] = 1 
				else:
					self.moves[i] = 2

				# Initial: q0, p0 (Dim = N)
				#---- RHMC steps
				for l in xrange(self.Nsteps):
					f_tmp, x_tmp, y_tmp, pf_tmp, px_tmp, py_tmp =\
						self.RHMC_single_step(f_tmp, x_tmp, y_tmp, pf_tmp, px_tmp, py_tmp, \
											delta = delta, counter_max = counter_max)
				pf_tmp *= -1
				px_tmp *= -1
				py_tmp *= -1				
				# After 1st RHMC: qL, -pL (Dim = N)

				#---- Birth or death move
				f_tmp, x_tmp, y_tmp, pf_tmp, px_tmp, py_tmp, factor = \
					self.birth_death_move(f_tmp, x_tmp, y_tmp, pf_tmp, \
						px_tmp, py_tmp, birth_death = birth_death)
				# RJ move: q*L, -p*L (Dim = N +- 1)

				#---- RHMC steps
				# Perform RHMC integration.
				for l in xrange(self.Nsteps):
					f_tmp, x_tmp, y_tmp, pf_tmp, px_tmp, py_tmp =\
						self.RHMC_single_step(f_tmp, x_tmp, y_tmp, pf_tmp, px_tmp, py_tmp, \
											delta = delta, counter_max = counter_max)
				# pf_tmp *= -1
				# px_tmp *= -1
				# py_tmp *= -1
				# After second RHMC: q*0, p*0 (Dim = N +- 1)
			elif move_type == 2: # MS
				# Roll the dice to determine whether it's merge or split
				# True - Merge and False - Split
				merge_split = np.random.choice([True, False], p=[0.5, 0.5])
				
				# Save which type of move was proposed.
				if merge_split: # True - merge
					self.moves[i] = 3
				else:
					self.moves[i] = 4

				# Initial: q0, p0 (Dim = N)
				#---- RHMC steps
				for l in xrange(self.Nsteps):
					f_tmp, x_tmp, y_tmp, pf_tmp, px_tmp, py_tmp =\
						self.RHMC_single_step(f_tmp, x_tmp, y_tmp, pf_tmp, px_tmp, py_tmp, \
											delta = delta, counter_max = counter_max)
				pf_tmp *= -1
				px_tmp *= -1
				py_tmp *= -1				
				# After 1st RHMC: qL, -pL (Dim = N)

				#---- Birth or death move
				f_tmp, x_tmp, y_tmp, pf_tmp, px_tmp, py_tmp, factor = \
					self.merge_split_move(f_tmp, x_tmp, y_tmp, pf_tmp, \
						px_tmp, py_tmp, merge_split = merge_split)
				# RJ move: q*L, -p*L (Dim = N +- 1)

				#---- RHMC steps
				# Perform RHMC integration.
				for l in xrange(self.Nsteps):
					f_tmp, x_tmp, y_tmp, pf_tmp, px_tmp, py_tmp =\
						self.RHMC_single_step(f_tmp, x_tmp, y_tmp, pf_tmp, px_tmp, py_tmp, \
											delta = delta, counter_max = counter_max)
				# pf_tmp *= -1
				# px_tmp *= -1
				# py_tmp *= -1
				# After second RHMC: q*0, p*0 (Dim = N +- 1)

			# ---- Compute the final energies and record
			self.V[i, 1] = self.Vq(f_tmp, x_tmp, y_tmp)
			self.T[i, 1] = self.Tqp(f_tmp, pf_tmp, px_tmp, py_tmp)
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

		return


	def birth_death_move(self, f, x, y, pf, px, py, birth_death = None):
		"""
		Implement birth/death move.
		Birth if birth_death = True, death if birth_death = False.
		"""
		# if (birth_death is None) or (self.alpha is None) or (self.f_min is None)\
		# 	or (self.f_max is None): # Prior must be provided.
		# 	assert False		

		if birth_death: # If birth
			if f.size == self.Nobjs_max-1:
				factor = -np.infty
				f_new, x_new, y_new, pf_new, px_new, py_new = f, x, y, pf, px, py
			else:
				# Create the output array and paste in the old
				f_new = np.zeros(f.size+1)
				x_new = np.zeros(f.size+1)
				y_new = np.zeros(f.size+1)			
				f_new[:-1] = f
				x_new[:-1] = x
				y_new[:-1] = y
				pf_new = np.zeros(f.size+1)
				px_new = np.zeros(f.size+1)
				py_new = np.zeros(f.size+1)			
				pf_new[:-1] = pf
				px_new[:-1] = px
				py_new[:-1] = py

				# Draw new source parameters.
				xs = np.random.random() * (self.N_rows - 1.)
				ys = np.random.random() * (self.N_cols - 1.)
				fs = gen_pow_law_sample(self.alpha, self.f_min/self.flux_to_count, \
					self.f_max/self.flux_to_count, 1)[0] * self.flux_to_count
				f_new[-1] = fs
				x_new[-1] = xs
				y_new[-1] = ys

				# Sample momentum based on the new source value.
				pfs, pxs, pys = self.sample_momentum(fs)
				pf_new[-1], px_new[-1], py_new[-1] = pfs, pxs, pys

				# Factor to be added to ln_alpha0
				factor = self.alpha * np.log(fs) - 3/2. + self.Tqp(fs, pfs, pxs, pys) - self.ln_C_prior
		else: # If death
			if f.size == 1: 
				factor = -np.infty
				f_new, x_new, y_new, pf_new, px_new, py_new = f, x, y, pf, px, py
			else:
				# Create the output array and paste in the old
				f_new = np.zeros(f.size-1)
				x_new = np.zeros(f.size-1)
				y_new = np.zeros(f.size-1)			
				pf_new = np.zeros(f.size-1)
				px_new = np.zeros(f.size-1)
				py_new = np.zeros(f.size-1)			
				
				# Randomly select an object to kill.
				i_kill = np.random.randint(0, f.size, size=1)[0]
				fs = f[i_kill]
				xs = x[i_kill]
				ys = y[i_kill]
				pfs = pf[i_kill]
				pxs = px[i_kill]
				pys = py[i_kill]

				# Appropriately trim
				f_new[:i_kill] =  f[:i_kill]
				x_new[:i_kill] =  x[:i_kill]
				y_new[:i_kill] =  y[:i_kill]
				pf_new[:i_kill]  = pf[:i_kill]
				px_new[:i_kill]  = px[:i_kill]
				py_new[:i_kill]  = py[:i_kill]
				f_new[i_kill:] =  f[i_kill+1:]
				x_new[i_kill:] =  x[i_kill+1:]
				y_new[i_kill:] =  y[i_kill+1:]
				pf_new[i_kill:]  = pf[i_kill+1:]
				px_new[i_kill:]  = px[i_kill+1:]
				py_new[i_kill:]  = py[i_kill+1:]

				# Factor to be added to ln_alpha0
				factor = -self.alpha * np.log(fs) + 3/2. - self.Tqp(fs, pfs, pxs, pys) + self.ln_C_prior

		return f_new, x_new, y_new, pf_new, px_new, py_new, factor



	def merge_split_move(self, f, x, y, pf, px, py, merge_split = None):
		"""
		Implement birth/death move.
		Merge if merge_split = True, split if merge_split = False.
		"""
		# if (merge_split is None) or (self.alpha is None) or (self.f_min is None)\
		# 	or (self.f_max is None): # Prior must be provided.
		# 	assert False		

		if merge_split: # If birth
			if f.size == self.Nobjs_max-1:
				factor = -np.infty
				f_new, x_new, y_new, pf_new, px_new, py_new = f, x, y, pf, px, py
			else:
				# Create the output array and paste in the old
				f_new = np.zeros(f.size+1)
				x_new = np.zeros(f.size+1)
				y_new = np.zeros(f.size+1)			
				pf_new = np.zeros(f.size+1)
				px_new = np.zeros(f.size+1)
				py_new = np.zeros(f.size+1)

				# Choose the source to be split
				i_star = np.random.randint(0, f.size, size=1)[0]
				f_star, x_star, y_star = f[i_star], x[i_star], y[i_star]
				pf_star, px_star, py_star = pf[i_star], px[i_star], py[i_star]

				# Draw the auxiliary variables
				dx, dy = np.random.randn(2) * self.K # distance
				dr_sq = dx**2 + dy**2 # Square distance			
				F = BETA.rvs(self.B_alpha, self.B_beta, size=1)[0] # Flux split fraction

				# Split the star
				f_prime = F * f_star
				x_prime = x_star + (1-F) * dx
				y_prime = y_star + (1-F) * dy			
				f_dprime = (1-F) * f_star
				x_dprime = x_star - F * dx
				y_dprime = y_star - F * dy

				# Sample momentum for the new stars
				pf_prime, px_prime, py_prime = self.sample_momentum(f_prime)
				pf_dprime, px_dprime, py_dprime = self.sample_momentum(f_dprime)

				# -- Put them all together in the above arrays
				# Add up to before split star
				f_new[:i_star] = f[:i_star]
				x_new[:i_star] = x[:i_star]
				y_new[:i_star] = y[:i_star]
				pf_new[:i_star] = pf[:i_star]
				px_new[:i_star] = px[:i_star]
				py_new[:i_star] = py[:i_star]
				# Add the remaining
				f_new[i_star:-2] = f[i_star+1:]
				x_new[i_star:-2] = x[i_star+1:]
				y_new[i_star:-2] = y[i_star+1:]
				pf_new[i_star:-2] = pf[i_star+1:]
				px_new[i_star:-2] = px[i_star+1:]
				py_new[i_star:-2] = py[i_star+1:]
				# Add the two new
				f_new[-2:] =np.array([f_prime, f_dprime])
				x_new[-2:] =np.array([x_prime, x_dprime])
				y_new[-2:] =np.array([y_prime, y_dprime])
				pf_new[-2:] = np.concatenate([pf_prime, pf_dprime])
				px_new[-2:] = np.concatenate([px_prime, px_dprime])
				py_new[-2:] = np.concatenate([py_prime, py_dprime])

				# Factor to be added to ln_alpha0
				factor = (-3/2.) + np.log(f_star) - BETA.logpdf(F, self.B_alpha, self.B_beta) \
					+ np.log(2 * np.pi * self.K**2) + (dr_sq / (2 * self.K**2)) \
					+ self.Tqp(f_star, pf_star, px_star, py_star) - self.Tqp(f_prime, pf_prime, px_prime, py_prime)\
					- self.Tqp(f_dprime, pf_dprime, px_dprime, py_dprime)
		else: # If merge
			if f.size == 1: 
				factor = -np.infty
				f_new, x_new, y_new, pf_new, px_new, py_new = f, x, y, pf, px, py
			else:				
				# Compute F-matrix and the associated probability
				F_matrix = f / (f.reshape((f.size, 1)) + f)
				ibool = np.abs(F_matrix-0.5) < 1e-10
				BETA_F = BETA.pdf(F_matrix, self.B_alpha, self.B_beta)
				BETA_F[ibool] = 0. # Eliminate self-merging possibility.

				# Compute distance matrix and the corresponding probability.
				R_sq = (x.reshape((f.size, 1)) - x)**2 + (y.reshape((f.size, 1)) - y)**2
				Q_dxdy = np.exp(-R_sq / (2. * self.K**2)) / (2. * np.pi * self.K**2)
				# Total probility
				P_choose = BETA_F * Q_dxdy
				P_choose /= np.sum(P_choose)

				#----Choose the objects to merge
				pair_num = np.random.choice(xrange(f.size**2), p=P_choose.ravel())
				idx_prime = pair_num // f.size
				idx_dprime = pair_num % f.size
				if idx_prime > idx_dprime:
					idx_prime, idx_dprime = idx_dprime, idx_prime

				#----Import the objects to merge and compute the result.
				# Unpack
				f_prime, x_prime, y_prime = f[idx_prime], x[idx_prime], y[idx_prime]
				f_dprime, x_dprime, y_dprime = f[idx_dprime], x[idx_dprime], y[idx_dprime]
				pf_prime, px_prime, py_prime = pf[idx_prime], px[idx_prime], py[idx_prime]
				pf_dprime, px_dprime, py_dprime = pf[idx_dprime], px[idx_dprime], py[idx_dprime]				

				# Comptue u varibles
				F = f_prime / (f_prime + f_dprime)
				dx = x_prime - x_dprime
				dy = y_prime - y_dprime
				dr_sq = dx**2 + dy**2

				# Compute merged
				f_star = f_prime + f_dprime
				x_star = F * x_prime + (1-F) * x_dprime
				y_star = F * y_prime + (1-F) * y_dprime

				# Draw momenta for the merge
				pf_star, px_star, py_star = self.sample_momentum(f_star)

				# --Put everything in the right order
				f_new = np.concatenate([f[:idx_prime], f[idx_prime+1:idx_dprime], f[idx_dprime+1:], np.array([f_star])])
				x_new = np.concatenate([x[:idx_prime], x[idx_prime+1:idx_dprime], x[idx_dprime+1:], np.array([x_star])])
				y_new = np.concatenate([y[:idx_prime], y[idx_prime+1:idx_dprime], y[idx_dprime+1:], np.array([y_star])])
				pf_new = np.concatenate([pf[:idx_prime], pf[idx_prime+1:idx_dprime], pf[idx_dprime+1:], pf_star])
				px_new = np.concatenate([px[:idx_prime], px[idx_prime+1:idx_dprime], px[idx_dprime+1:], px_star])
				py_new = np.concatenate([py[:idx_prime], py[idx_prime+1:idx_dprime], py[idx_dprime+1:], py_star])

				# Factor to be added to ln_alpha0
				factor = (+3/2.) - np.log(f_star) + BETA.logpdf(F, self.B_alpha, self.B_beta) \
					- np.log(2 * np.pi * self.K**2) - (dr_sq / (2 * self.K**2)) \
					- self.Tqp(f_star, pf_star, px_star, py_star) + self.Tqp(f_prime, pf_prime, px_prime, py_prime)\
					+ self.Tqp(f_dprime, pf_dprime, px_dprime, py_dprime)

		return f_new, x_new, y_new, pf_new, px_new, py_new, factor


	def diagnostics_all(self, idx_iter = -1, figsize = (16, 11), \
						color_truth="red", color_model="blue", ft_size = 15, num_ticks = 5, \
						show=False, save=None, title_str = None, \
						m=-20, b = 5, s0=20, y_min=5, m_min = 15., m_max = 21.5):
		"""
		- idx_iter: Index of the iteration to plot.
		- m, b, s0, y_min: Parameters for the scatter plot.
		"""
		assert self.vmin is not None

		# --- Extract X, Y, Mag variables
		# Truth 
		F0 = self.q0[0]
		X0 = self.q0[1]
		Y0 = self.q0[2]
		S0 = linear_func(self.flux2mag_converter(F0), m=m, b = b, s0=s0, y_min=y_min)
		# Model
		F = self.q[idx_iter, 0, :self.N[idx_iter]]
		X = self.q[idx_iter, 1, :self.N[idx_iter]]
		Y = self.q[idx_iter, 2, :self.N[idx_iter]]
		S = linear_func(self.flux2mag_converter(F), m=m, b = b, s0=s0, y_min=y_min)

		# --- Make the plot
		fig, ax_list = plt.subplots(2, 3, figsize=figsize)
		# ---- Joining certain axis
		ax_list[0, 0].get_shared_x_axes().join(ax_list[0, 0], ax_list[1, 0])
		ax_list[0, 0].get_shared_y_axes().join(ax_list[0, 0], ax_list[0, 1])
		ax_list[0, 0].get_shared_y_axes().join(ax_list[0, 0], ax_list[1, 1])
		ax_list[0, 0].get_shared_x_axes().join(ax_list[0, 0], ax_list[1, 1])
		ax_list[0, 0].get_shared_y_axes().join(ax_list[0, 0], ax_list[1, 2])
		ax_list[0, 0].get_shared_x_axes().join(ax_list[0, 0], ax_list[1, 2])

		# (0, 0): Image
		ax_list[0, 0].imshow(self.D, cmap="gray", interpolation="none", vmin=self.vmin, vmax=self.vmax)
		# Truth locs
		ax_list[0, 0].scatter(Y0, X0, c=color_truth, s=S0, edgecolor="none", marker="x")
		# Model locs
		ax_list[0, 0].scatter(Y, X, c=color_model, s=S, edgecolor="none", marker="x")
		# Decorations
		ax_list[0, 0].set_title("Data", fontsize=ft_size)
		ax_list[0, 0].set_xlabel("Y", fontsize=ft_size)
		ax_list[0, 0].set_ylabel("X", fontsize=ft_size)
		yticks00 = ticker.MaxNLocator(num_ticks)
		xticks00 = ticker.MaxNLocator(num_ticks)
		ax_list[0, 0].yaxis.set_major_locator(yticks00)
		ax_list[0, 0].xaxis.set_major_locator(xticks00)
		ax_list[0, 0].set_xlim([-1.5, self.N_rows])
		ax_list[0, 0].set_ylim([-1.5, self.N_cols])

		# (0, 1): Mag - X
		ax_list[0, 1].scatter(self.flux2mag_converter(F0), X0, c=color_truth, s=S0, edgecolor="none", marker="x")
		ax_list[0, 1].scatter(self.flux2mag_converter(F), X, c=color_model, s=S, edgecolor="none", marker="x")
		# Decorations
		ax_list[0, 1].axvline(x=self.flux2mag_converter(self.f_min), c="green", lw=1.5, ls="--")
		ax_list[0, 1].set_ylabel("X", fontsize=ft_size)
		ax_list[0, 1].set_xlabel("Mag", fontsize=ft_size)
		ax_list[0, 1].set_xlim([m_min, m_max])		
		yticks10 = ticker.MaxNLocator(num_ticks)
		xticks10 = ticker.MaxNLocator(num_ticks)
		ax_list[0, 1].yaxis.set_major_locator(yticks10)
		ax_list[0, 1].xaxis.set_major_locator(xticks10)	


		# (1, 0): Y - Mag
		ax_list[1, 0].scatter(Y0, self.flux2mag_converter(F0), c=color_truth, s=S0, edgecolor="none", marker="x")
		ax_list[1, 0].scatter(Y, self.flux2mag_converter(F), c=color_model, s=S, edgecolor="none", marker="x")
		# Decorations
		ax_list[1, 0].axhline(y=self.flux2mag_converter(self.f_min), c="green", lw=1.5, ls="--")
		ax_list[1, 0].set_ylabel("Mag", fontsize=ft_size)
		ax_list[1, 0].set_xlabel("Y", fontsize=ft_size)
		ax_list[1, 0].set_ylim([m_min, m_max])		
		yticks10 = ticker.MaxNLocator(num_ticks)
		xticks10 = ticker.MaxNLocator(num_ticks)
		ax_list[1, 0].yaxis.set_major_locator(yticks10)
		ax_list[1, 0].xaxis.set_major_locator(xticks10)	

		# (1, 1): Model
		model = self.gen_model_image(F, X, Y)
		ax_list[1, 1].imshow(model, cmap="gray", interpolation="none", vmin=self.vmin, vmax=self.vmax)
		ax_list[1, 1].scatter(Y0, X0, c=color_truth, s=S0, edgecolor="none", marker="x")
		ax_list[1, 1].scatter(Y, X, c=color_model, s=S, edgecolor="none", marker="x")
		ax_list[1, 1].set_title("Model", fontsize=ft_size)
		yticks11 = ticker.MaxNLocator(num_ticks)
		xticks11 = ticker.MaxNLocator(num_ticks)
		ax_list[1, 1].yaxis.set_major_locator(yticks11)
		ax_list[1, 1].xaxis.set_major_locator(xticks11)
		ax_list[1, 1].set_xlabel("Y", fontsize=ft_size)
		ax_list[1, 1].set_ylabel("X", fontsize=ft_size)

		# (1, 2): Residual
		sig_fac = 7.
		residual = self.D - model
		sig = np.sqrt(self.B_count)
		ax_list[1, 2].imshow(residual, cmap="gray", interpolation="none", vmin=-sig_fac * sig, vmax=sig_fac * sig)
		ax_list[1, 2].scatter(Y0, X0, c=color_truth, s=S0, edgecolor="none", marker="x")
		ax_list[1, 2].scatter(Y, X, c=color_model, s=S, edgecolor="none", marker="x")
		ax_list[1, 2].set_title("Residual", fontsize=ft_size)
		yticks12 = ticker.MaxNLocator(num_ticks)
		xticks12 = ticker.MaxNLocator(num_ticks)
		ax_list[1, 2].yaxis.set_major_locator(yticks12)
		ax_list[1, 2].xaxis.set_major_locator(xticks12)
		ax_list[1, 2].set_xlabel("Y", fontsize=ft_size)
		ax_list[1, 2].set_ylabel("X", fontsize=ft_size)

		# (0, 2): Residual histogram
		sig_fac2 = 10. # Histogram should plot wider range
		sig = np.sqrt(self.B_count)
		bins = np.arange(-sig_fac2 * sig, sig_fac2 * sig, sig/5.)
		ax_list[0, 2].step(self.centers_noise, self.hist_noise * (self.N_rows * self.N_cols), color="blue", lw=1.5)
		ax_list[0, 2].hist(residual.ravel(), bins=bins, color="black", lw=1.5, histtype="step")
		ax_list[0, 2].set_xlim([-sig_fac2 * sig, sig_fac2 * sig])
		ax_list[0, 2].set_ylim([0, np.max(self.hist_noise) * 1.1 * (self.N_rows * self.N_cols)])		
		ax_list[0, 2].set_title("Res. hist", fontsize=ft_size)
		yticks02 = ticker.MaxNLocator(num_ticks)
		xticks02 = ticker.MaxNLocator(num_ticks)
		ax_list[0, 2].yaxis.set_major_locator(yticks02)
		ax_list[0, 2].xaxis.set_major_locator(xticks02)

		# Add title
		if title_str is not None:
			plt.suptitle(title_str, fontsize=25)

		if save is not None:
			plt.savefig(save, dpi=200, bbox_inches = "tight")
		if show:
			plt.show()			
		plt.close()

	def flux2mag_converter(self, F):
		return flux2mag(F / self.flux_to_count) # The division is necessary because flux is already in counts units.

	def print_accept_rate(self):
		print np.mean(self.A)