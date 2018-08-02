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
		self.q = np.zeros((Niter+1, 3, Nobjs_max)) # Channels: 0 - f, 1 - x, 2 - y.
		self.p = np.zeros((Niter+1, 3, Nobjs_max))
		self.E = np.zeros((Niter+1, 2))
		self.V = np.zeros((Niter+1, 2)) # Potential
		self.T = np.zeros((Niter+1, 2))	# Kinetic
		self.N = np.zeros(Niter+1, dtype=int) # The total number of **objects** at a particular point in the inference.
		self.A = np.zeros(self.Niter, dtype=bool) # Was the proposal accepted?
		self.moves = np.zeros(self.Niter, dtype=int) # Record what sort of proposals were made.

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

	def do_inference(self, delta=1e-6, counter_max = 1000, verbose=False):
		"""
		Perform inference based on the data generated by the truth sample
		with the generate initial model samples. 

		The arguments to this function are parameters for the RHMC 
		integrator. 
		"""
		#---- Set the very first initial point.
		# q_init = np.copy(self.q)

		# #---- Perform the iterations
		# for l in xrange(self.Niter+1):
		# 	# ---- Adjust parameters according to schedule
		# 	if schedule_g_ff2 is not None:
		# 		if l < schedule_g_ff2.size:
		# 			self.g_ff2 = schedule_g_ff2[l]
		# 	if schedule_beta is not None:
		# 		if l < schedule_beta.size:
		# 			self.beta = schedule_beta[l]

		# 	# ---- Compute the initial q, p and energies.
		# 	# The initial q_tmp has already been set at the end of the previous run.			
		# 	# Resample momentum
		# 	H_diag = self.H(q_tmp, grad=False)
		# 	p_tmp = self.u_sample(self.d) * np.sqrt(H_diag)

		# 	# Compute the initial energies
		# 	V_initial = self.V(q_tmp, f_pos=f_pos)
		# 	T_initial = self.T(p_tmp, H_diag)
		# 	E_initial = V_initial + T_initial # Necessary to save to compute dE

		# 	#---- Save the initial point and energies
		# 	if save_traj:
		# 		self.q_chain[l, 0] = q_tmp
		# 		self.p_chain[l, 0] = p_tmp
		# 		self.V_chain[l, 0] = V_initial
		# 		self.E_chain[l, 0] = E_initial
		# 		self.T_chain[l, 0] = T_initial			
		# 	else:
		# 		# Only time energy is saved in the whole iteration.
		# 		self.q_chain[l, :self.Nobjs*3] = q_tmp
		# 		self.p_chain[l, :self.Nobjs*3] = p_tmp
		# 		self.V_chain[l] = V_initial
		# 		self.E_chain[l] = E_initial
		# 		self.T_chain[l] = T_initial					
		# 	self.N_chain[l] = self.Nobjs # By convention the initial number of objects is saved.

		# 	# ---- Roll dice and choose which proposal to make.
		# 	move_type = np.random.choice([0, 1, 2], p=self.P_move, size=1)[0]

		# 	# ---- Regular RHMC integration
		# 	if move_type == 0:
		# 		self.move_chain[l] = 0 # Save which type of move was proposed.

		# 		#---- Looping over steps
		# 		for i in xrange(1, self.Nsteps+1, 1):
		# 			q_tmp, p_tmp = self.RHMC_single_step(q_tmp, p_tmp, delta = delta, counter_max = counter_max)

		# 			# Intermediate variables save if asked
		# 			if save_traj:
		# 				# Diagonal H update
		# 				H_diag = self.H(q_tmp, grad=False)
		# 				self.q_chain[l, i] = q_tmp
		# 				self.p_chain[l, i] = p_tmp
		# 				self.V_chain[l, i] = self.V(q_tmp, f_pos=f_pos)
		# 				self.T_chain[l, i] = self.T(p_tmp, H_diag)
		# 				self.E_chain[l, i] = self.V_chain[l, i] + self.T_chain[l, i]

		# 		# Compute the energy difference between the initial and the final energy
		# 		if save_traj: # If the energy has been already saved.
		# 			E_final = self.E_chain[l, -1]
		# 		else:
		# 			H_diag = self.H(q_tmp, grad=False)				
		# 			E_final = self.V(q_tmp, f_pos=f_pos) + self.T(p_tmp, H_diag)
		# 		dE = E_final - E_initial

		# 		# Accept or reject and set the next initial point accordingly.
		# 		lnu = np.log(np.random.random(1))
		# 		if (dE < 0) or (lnu < -dE): # If accepted.
		# 			self.A_chain[l] = 1
		# 		else: # Otherwise, proposal rejected.
		# 			# Reseting the position variable to the previous.
		# 			if save_traj:
		# 				q_tmp = self.q_chain[l, 0]
		# 			else:
		# 				q_tmp = self.q_chain[l, :self.Nobjs * 3]						
		# 	elif move_type == 1: # If birth or death
		# 		# Roll the dice to determine whether it's birth or death
		# 		# True - Birth and False - Death.
		# 		birth_death = np.random.choice([True, False], p=[0.5, 0.5])
				
		# 		# Save which type of move was proposed.
		# 		if birth_death: # True - Birth
		# 			self.move_chain[l] = 1 
		# 		else:
		# 			self.move_chain[l] = 2

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

		# 		# Compute the final energy after move.
		# 		H_diag = self.H(q_tmp, grad=False)
		# 		E_final = self.V(q_tmp, f_pos=f_pos) + self.T(p_tmp, H_diag)

		# 		# Difference
		# 		dE = E_final - E_initial

		# 		# Compute the log-probability of accept or reject.
		# 		ln_alpha0 = -dE	+ factor		
		# 		lnu = np.log(np.random.random(1))
		# 		if (ln_alpha0 > 0) or (lnu < ln_alpha0): # If accepted.
		# 			self.A_chain[l] = 1
		# 		else: # Otherwise, proposal rejected.
		# 			if birth_death: # undoing the change in numbers
		# 				self.Nobjs -= 1
		# 				self.d -= 3
		# 			else:
		# 				self.Nobjs += 1
		# 				self.d += 3				
		# 			q_tmp = self.q_chain[l, :self.Nobjs * 3]			
		# 	elif move_type == 2: # If it merge or split
		# 		# Roll the dice to determine whether it's merge or split
		# 		# True - Split and False - Merge
		# 		split_merge = np.random.choice([True, False], p=[0.5, 0.5])
				
		# 		# Save which type of move was proposed.
		# 		if split_merge: # True - Split
		# 			self.move_chain[l] = 3
		# 		else:
		# 			self.move_chain[l] = 4

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

		# 		# Compute the final energy after move.
		# 		H_diag = self.H(q_tmp, grad=False)
		# 		E_final = self.V(q_tmp, f_pos=f_pos) + self.T(p_tmp, H_diag)

		# 		# Difference
		# 		dE = E_final - E_initial

		# 		# Compute the log-probability of accept or reject.
		# 		ln_alpha0 = -dE	+ factor		
		# 		lnu = np.log(np.random.random(1))
		# 		if (ln_alpha0 > 0) or (lnu < ln_alpha0): # If accepted.
		# 			self.A_chain[l] = 1
		# 		else: # Otherwise, proposal rejected.
		# 			if split_merge: # undoing the change in numbers
		# 				self.Nobjs -= 1
		# 				self.d -= 3
		# 			else:
		# 				self.Nobjs += 1
		# 				self.d += 3				
		# 			q_tmp = self.q_chain[l, :self.Nobjs * 3]	

		# 	if verbose and ((l%50) == 0):
		# 		print "/---- Completed iteration %d" % l
		# 		print "N_objs: %d\n" % self.Nobjs
		# 		self.R_accept_report(idx_iter = l, run_window = 10)

		# 		# Produce a diagnostic plot
		# 		self.diagnostics_all(q_true, show=False, idx_iter = l, idx_step=0, save="iter-%d.png" % l,\
  #                  m=-15, b =10, s0=23, y_min=5.)				
		# 		print "\n\n"


		# # ---- Compute the total acceptance rate.
		# print "Finished. Final report."
		# self.R_accept_report(idx_iter = -1, running = False)		

		return		