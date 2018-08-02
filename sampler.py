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
		C_f = (1-alpha) / (self.f_max**(1-alpha) - self.f_min**(1-alpha)) 
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
		f_min = mag2flux(mB) * flux_to_count # Limiting magnitude
		f_max = mag2flux(15) * flux_to_count 

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
			px_prime = rho_x
			py_prime = rho_y
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
		pf[(f > self.f_min)] *= -1
		px[x < 0] *= -1
		px[x > self.N_rows] *= -1
		py[y < 0] *= -1
		py[y > self.N_rows] *= -1

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