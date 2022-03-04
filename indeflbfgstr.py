import torch
from torch.optim import Optimizer
from functools import reduce
import numpy as np
import scipy.linalg as sl
from pdb import set_trace
import math

def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    # ported from https://github.com/torch/optim/blob/master/polyinterp.lua
    # Compute bounds of interpolation area
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    # Code for most common case: cubic interpolation of 2 points
    #   w/ function and derivative values for both
    # Solution in this case (where x2 is the farthest point):
    #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
    #   d2 = sqrt(d1^2 - g1*g2);
    #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2
    if d2_square >= 0:
        d2 = d2_square.sqrt()
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.


def _strong_wolfe(obj_func,
                  x,
                  t,
                  d,
                  f,
                  g,
                  gtd,
                  c1=1e-4,
                  c2=0.9,
                  tolerance_change=1e-9,
                  max_ls=25):
    # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
    d_norm = d.abs().max()
    g = g.clone(memory_format=torch.contiguous_format)
    # evaluate objective and gradient using initial step
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
    gtd_new = g_new.reshape(-1).dot(d.reshape(-1))

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        # check conditions
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if abs(gtd_new) <= -c2 * gtd:
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # interpolate
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = _cubic_interpolate(
            t_prev,
            f_prev,
            gtd_prev,
            t,
            f_new,
            gtd_new,
            bounds=(min_step, max_step))

        # next step
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.reshape(-1).dot(d.reshape(-1))
        ls_iter += 1

    # reached max number of iterations?
    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]

    # zoom phase: we now have a point satisfying the criteria, or
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    insuf_progress = False
    # find high and low points in bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
    while not done and ls_iter < max_ls:
        # line-search bracket is so small
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            break

        # compute new trial value
        t = _cubic_interpolate(bracket[0], bracket_f[0], bracket_gtd[0],
                               bracket[1], bracket_f[1], bracket_gtd[1])

        # test that we are making sufficient progress:
        # in case `t` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   + we have made insufficient progress in the last step, or
        #   + `t` is at one of the boundary,
        # we will move `t` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # interpolation close to boundary
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                # evaluate at 0.1 away from boundary
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate new point
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.reshape(-1).dot(d.reshape(-1))
        ls_iter += 1

        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd:
                # Wolfe conditions satisfied
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                # old high becomes new low
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            # new point becomes new low
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[low_pos] = gtd_new

    # return stuff
    t = bracket[low_pos]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]
    return f_new, g_new, t, ls_func_evals


class indefLBFGS(Optimizer):
	"""docstring for LSR1thirdNorm"""
	def __init__(self, 
				 params, 
				 eta,
				 eta1,
				 history_size,
				 deltacap=100, 
				 lr=1,
				 tolerance_change=1e-9,
				 tolerance_grad=1e-7,
				 max_iters=4,
				 max_eval=None):
		if max_eval is None:
			max_eval = max_iters * 5 // 4
		
		if lr<=0:
			raise ValueError("Invalid initial learning rate")

		defaults = dict(lr=lr,
						history_size=history_size,
						eta=eta,
						eta1=eta1,
						n_iters=10,
						max_iters=max_iters,
						tolerance_change=tolerance_change,
						tolerance_grad=tolerance_grad, 
						line_search_fn=None,
						max_eval=max_eval,
						deltacap = deltacap
						)

		super(indefLBFGS, self).__init__(params, defaults)

		if len(self.param_groups)!=1:
			raise ValueError("Cubic LSR1 doesn't support per-parameter options"
							"(parameter groups)")

		self._params = self.param_groups[0]['params']
		self._numel_cache = None
		self.deltacap = deltacap

	def _directional_evaluate(self, closure, x, t, d):
		self._add_grad(t, d)
		loss = float(closure())
		flat_grad = self._gather_flat_grad()
		self._set_param(x)
		return loss, flat_grad

	def _numel(self):
		if self._numel_cache is None:
			self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)

		return self._numel_cache

	def _gather_flat_grad(self):
		views = []
		for p in self._params:
			p.retain_grad()
			if p.grad is None:
				view = p.new(p.numel()).zero_()
			elif p.grad.is_sparse:
				view = p.grad.to_dense().view(-1)
			else:
				view = p.grad.view(-1)
			views.append(view)
		return torch.cat(views, axis=0)


	def _set_param(self, params_data):
		for p, p_data in zip(self._params, params_data):
			p.copy_(p_data)

	def _add_param(self, update):
		offset = 0
		for p, p_data in zip(self._params, update):
			numel = p.numel()
			p.add_(update[offset:offset+numel].view_as(p), alpha=1)
			offset+=numel
		assert offset == self._numel()

	def _clone_param(self):
		return [p.clone(memory_format=torch.contiguous_format) for p in self._params]


	def _add_grad(self, step_size, update):
		offset=0
		for p in self._params:
			numel = p.numel()
			p.add_(update[offset:offset+numel].view_as(p), alpha=step_size)
			offset+=numel

		assert offset==self._numel()


	@torch.no_grad()
	def step(self, closure=None):

		group = self.param_groups[0]
		eta2 = group['eta']
		history_size = group['history_size']
		lr = group['lr']
		max_eval = group['max_eval']
		n_iters = group['n_iters']
		max_iters = group['max_iters']
		tolerance_change = group['tolerance_change']
		tolerance_grad = group['tolerance_grad']
		line_search_fn = group['line_search_fn']
		lr = group['lr']
		state = self.state[self._params[0]]
		orig_loss = closure()
		loss = float(orig_loss)	

		global device 
		device = self._params[0].device.type
		n_iters = 0
		# Assert only one param group exists
		assert len(self.param_groups)==1

		if closure is None:
			raise ValueError("Function not provided !")
		else:
			closure = torch.enable_grad()(closure)
			orig_loss = closure()

		#Define the global state
		flat_grad = self._gather_flat_grad()
		S = state.get('S')
		Y = state.get('Y')
		SS = state.get('SS')
		SY = state.get('SY')
		YY = state.get('YY')
		sstar = state.get('sstar')
		t = state.get('t')
		state.setdefault('func_evals',0)
		state.setdefault('n_iters',0)
		prev_flat_grad = state.get('prev_flat_grad')
		prev_loss = state.get('prev_loss')
		current_evals = 1
		flag = state.get('flag')
		if flag is None:
			flag = True
		delta = state.get('delta')
		gamma = state.get('gamma')

		if delta is None:
			delta = 10

		n_iters = 0
		# print(delta)
		while n_iters < max_iters:
			print('Iteration: {}  Iterate: {}   Gradient: {}   Functional value: {}' .format(state['n_iters'], self._params[0].data, self._gather_flat_grad().data, float(closure())))
			n_iters+=1
			if state['n_iters'] == 4:
				set_trace()
			state['n_iters']+=1
			if state['n_iters'] == 1:
				sstar = flat_grad.neg()
				set_trace()
				S = None
				Y = None
				SS = None
				YY = None
				SY = None

			else:
				if flag:
					y = flat_grad.sub(prev_flat_grad)
					sstar = sstar.mul(t)
					if S is None:

						S = sstar.unsqueeze(1)
						Y = y.unsqueeze(1)
						SS = sstar.dot(sstar)[None, None]
						SY = sstar.dot(y)[None, None]
						YY = y.dot(y)[None,None]

					elif S.shape[1]<history_size:

						SY = torch.vstack((torch.hstack((SY, S.T @ y.unsqueeze(1))), torch.hstack((sstar.unsqueeze(1).T @ Y , sstar.unsqueeze(1).T @ y.unsqueeze(1)))))
						SS = torch.vstack((torch.hstack((SS, S.T @ sstar.unsqueeze(1))), torch.hstack((sstar.unsqueeze(1).T @ S , sstar.unsqueeze(1).T @ sstar.unsqueeze(1)))))
						YY = torch.vstack((torch.hstack((YY, Y.T @ y.unsqueeze(1))), torch.hstack((y.unsqueeze(1).T @ Y , y.unsqueeze(1).T @ y.unsqueeze(1)))))
						S = torch.cat([S, sstar.unsqueeze(1)], axis=1)
						Y = torch.cat([Y, y.unsqueeze(1)], axis=1)

					else:

						SY = torch.vstack((torch.hstack((SY[1:,1:], S[:,1:].T @ y.unsqueeze(1))), torch.hstack((sstar.unsqueeze(1).T @ Y[:,1:] , sstar.unsqueeze(1).T @ y.unsqueeze(1)))))
						SS = torch.vstack((torch.hstack((SS[1:,1:], S[:,1:].T @ sstar.unsqueeze(1))), torch.hstack((sstar.unsqueeze(1).T @ S[:,1:] , sstar.unsqueeze(1).T @ sstar.unsqueeze(1)))))
						YY = torch.vstack((torch.hstack((YY[1:,1:], Y[:,1:].T @ y.unsqueeze(1))), torch.hstack((y.unsqueeze(1).T @ Y[:,1:] , y.unsqueeze(1).T @ y.unsqueeze(1)))))
						S = torch.cat([S[:,1:], sstar.unsqueeze(1)], axis=1)
						Y = torch.cat([Y[:,1:], y.unsqueeze(1)], axis=1)


				# set_trace()
				Psi, Psip, sstar, gamma, g, M, S, Y, SS, YY, SY = self.LBFGS(S, SS, YY, SY, Y, flat_grad, gamma, delta)
				delta, flag = self.trustRegion(g, Psi, Psip, gamma, closure, sstar, M, delta)


			if prev_flat_grad is None:
				prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
			else:
				prev_flat_grad.copy_(flat_grad)
			prev_loss = loss

			if state['n_iters'] == 1:
				t = min(1., 1. / flat_grad.abs().sum()) * lr
			else:
				t = 1


			ls_func_evals = 0
			if line_search_fn is not None:
				# perform line search, using user function
				if line_search_fn != "strong_wolfe":
					raise RuntimeError("only 'strong_wolfe' is supported")
				else:
					x_init = self._clone_param()

					def obj_func(x, t, d):
						return self._directional_evaluate(closure, x, t, d)

					loss, flat_grad, t, ls_func_evals = _strong_wolfe(
						obj_func, x_init, t, sstar, loss, flat_grad, gtd)

				# set_trace()

				self._add_grad(t, sstar)
				opt_cond = flat_grad.abs().max() <= tolerance_grad

			else:
				# no line search, simply move with fixed-step
				if flag:
					# set_trace()
					self._add_grad(t, sstar)

					if n_iters != max_iters:
						# re-evaluate function only if not in last iteration
						# the reason we do this: in a stochastic setting,
						# no use to re-evaluate that function here
						with torch.enable_grad():
							loss = float(closure())

						ls_func_evals = 1

						flat_grad = self._gather_flat_grad()
						y = flat_grad - prev_flat_grad
						if torch.norm(y) == 0.0:
							set_trace()

				opt_cond = flat_grad.abs().max() <= tolerance_grad
				


            # update func eval
			current_evals += ls_func_evals
			state['func_evals'] += ls_func_evals

			if n_iters == max_iters:
				break


		state['sstar'] = sstar
		state['SS'] = SS
		state['YY'] = YY
		state['SY'] = SY
		state['Y'] = Y
		state['S'] = S
		state['gamma'] = gamma
		state['prev_flat_grad'] = prev_flat_grad
		state['prev_loss'] = prev_loss
		state['t'] = t
		state['delta'] = delta
		state['flag'] = flag
		return orig_loss

	def LBFGS(self, S, SS, YY, SY, Y, g, gammaIn, delta):
		tol = 1e-10
		gamma = Y[:,-1].dot(Y[:,-1])/Y[:,-1].dot(S[:,-1])
		if gamma < 0:
			try:

				A = torch.tril(SY) + torch.tril(SY,-1).T
				B = SS

				v = torch.from_numpy(sl.eigh(A.cpu().numpy(),B.cpu().numpy(), eigvals_only=True))
				eABmin = min(v)
				if(eABmin>0):
					gamma = max(0.5*eABmin, 1e-6)
				else:
					gamma = min(1.5*eABmin, -1e-6)

			except:	
				gamma = gammaIn

		Psi = torch.hstack([gamma*S, Y])

		
		PsiPsi = Psi.T @ Psi
		try:
			R=torch.linalg.cholesky(PsiPsi.transpose(-2,-1).conj()).transpose(-2,-1).conj()
			Q = Psi @ torch.inverse(R)

			if R.any().isnan():
				set_trace()
		except:
			Q, R = torch.linalg.qr(Psi)

		
		upper = torch.hstack([gamma*SS, torch.tril(SY,-1)])
		lower = torch.hstack([torch.tril(SY,-1).T, -torch.diag_embed(torch.diag(SY))])
		
		M = torch.vstack([upper, lower])
		invM = torch.inverse(M)


		try:
			RMR = R @ invM @ R.T
		except RuntimeError:
			set_trace()

		RMR = 0.5*(RMR + RMR.T)
		try:
			D,P = torch.linalg.eigh(RMR)
		except:
			set_trace()
			
		U_par = Q @ P
		if len(U_par.shape)==1:
			U_par = U_par.unsqueeze(1)

		sizeD = D.shape[0]

		Lambda_one = D + gamma * torch.ones(D.shape[0])
		Lambda = torch.cat((Lambda_one, gamma.unsqueeze(0)))
		Lambda = Lambda* (torch.abs(Lambda)>0)
		lambdamin = torch.min(Lambda[0], gamma)


		Psig = Psi.T @ g 

		g_parallel = U_par.T @ g

		a_kp2 = torch.sqrt(torch.abs(g.T @ g - g_parallel.T @ g_parallel))

		if a_kp2 < tol:
			a_kp2 = torch.zeros(1)


		a_j = torch.cat((g_parallel, a_kp2.unsqueeze(0))) if not a_kp2.shape else torch.cat((g_parallel, a_kp2))		

		if (lambdamin>0) and torch.norm(a_j/Lambda)<=delta:
			sigmaStar = 0
			sstar = self.ComputeBySMW(gamma, g, Y, S, gamma, Psig, invM, Psi, PsiPsi)


		elif (lambdamin<=0) and self.phiBar_f(-lambdamin,delta,Lambda,a_j) >0:
			sigmaStar = -lambdamin
			index_pseudo = torch.where(torch.abs(Lambda + sigmaStar)>tol)[0]
			v = torch.zeros(sizeD+1)
			v[index_pseudo] = a_j[index_pseudo]/(Lambda[index_pseudo]+sigmaStar)

			if torch.abs(gamma+sigmaStar)<tol:
				sstar = -U_par @ v[:sizeD]
			else:
				sstar = -U_par @ v[:sizeD] + (1/(gamma+sigmaStar)) * (Psi @ (torch.inverse(PsiPsi) @ Psig)) - (g/(gamma+sigmaStar))

			if lambdamin<0:
			
				alpha = torch.sqrt(delta**2 - sstar.T @ sstar)
				pHatStar = sstar
				if torch.abs(lambdamin-Lambda[0])<tol:
					zstar = (1/torch.norm(U_par[:,0]))*alpha*U_par[:,0]

				else:
					e = torch.zeros(g.shape)
					found = 0
					for j in range(sizeD):
						e[i] = 1
						u_min = e - U_par @ U_par[i,:].T
						if torch.norm(u_min)>tol:
							found = 1
							break

						e[i] = 0
						# set_trace()

					if found == 0:
						e[m+1]=1
						u_min = e- U_par @ U_par[i,:]

				sstar = pHatStar + zstar


		else:
			if lambdamin > 0:
				sigmaStar = self.Newton(0, 5, tol, delta, Lambda, a_j)

			else:
				sigmaHat = torch.max(a_j/delta - Lambda)
				if sigmaHat > -lambdamin:
					sigmaStar = self.Newton(sigmaHat, 5, tol, delta, Lambda, a_j)
				else:
					sigmaStar = self.Newton(-lambdamin, 5, tol, delta, Lambda, a_j)
			
			sstar = self.ComputeBySMW(gamma+sigmaStar, g, Y, S, gamma, Psig, invM, Psi, PsiPsi)

		if sstar.any().isnan():
			set_trace()

		Psip = Psi.T @ sstar
		return Psi, Psip, sstar, gamma, g, invM, S, Y, SS, YY, SY


	def phiBar_f(self, sigma, delta, D, a_j):
		m = a_j.shape[0]
		D= D + sigma*torch.ones(m)
		eps_tol = 1e-10

		if (torch.sum(torch.abs(a_j)<eps_tol) >0 ) or (torch.sum(torch.abs(torch.diag(D)) < eps_tol) > 0):
			pnorm2 = 0
			for i in range(m):
				if (torch.abs(a_j[i]) > eps_tol) and (torch.abs(D[i]) < eps_tol):
					phiBar = -1/delta
					return phiBar
				elif (torch.abs(a_j[i]) > eps_tol) and (torch.abs(D[i]) > eps_tol):
					pnorm2 = pnorm2 + (a_j[i]/D[i])**2


			phiBar = torch.sqrt(1/pnorm2) - 1/delta
			return phiBar


		p = a_j/D
		phiBar =  1/torch.norm(p) - 1/delta
		return phiBar

	def lmarquardt(self, g, Psi, Psis, gamma, closure, sstar, invM):
		q = g.T @ sstar + 0.5*(gamma*(sstar.T @ sstar)  - Psis.T @ invM @ Psis)
		x_init = self._clone_param()
		f1 = float(closure())
		f2,_ = self._directional_evaluate(closure, x_init,1, sstar)
		return (f1 - f2)/(-q)


	def trustRegion(self, g, Psi, Psip, gamma, closure, sstar, M, delta):

		rhok = self.lmarquardt(g, Psi, Psip, gamma, closure, sstar, M)
		# print('delta: {}'.format(delta))
		# print('rhok: {}'.format(rhok))
		if rhok < self.defaults['eta']:
			delta = 0.25*delta

		else:
			if rhok > self.defaults['eta1'] and torch.norm(sstar) == delta:
				delta = min(2*delta, self.deltacap)

		if rhok> self.defaults['eta']:
			flag = True

		else:
			flag = False

		# set_trace()
		return delta, flag

	def Newton(self, x0, maxIter, tol, delta, Lambda, a_j):
		x = x0
		k = 0

		f, g  = self.phiBar_fg(x,delta,Lambda,a_j)

		while (torch.abs(f) > 1e-20) and (k < maxIter):
		    x  = x - f / g
		    f, g  = self.phiBar_fg(x,delta,Lambda,a_j)
		    k = k + 1
		
		return x

	def phiBar_fg(self, sigma, delta, D, a_j):
		m = a_j.shape[0] 
		D = D + sigma*torch.ones(m)
		eps_tol  = 1e-10
		phiBar_g = 0

		if (torch.sum(torch.abs(a_j) < eps_tol) > 0) or (torch.sum(torch.abs(D) < eps_tol ) > 0):    
			pnorm2 = torch.zeros(1)
			for i in range(m):
				if (torch.abs(a_j[i]) > eps_tol) and (torch.abs(D[i]) < eps_tol):
					phiBar   = torch.tensor(-1/delta);
					phiBar_g = 1/torch.sqrt(torch.tensor(eps_tol));
					return phiBar, phiBar_g

				elif torch.abs(a_j[i]) > eps_tol and torch.abs(D[i]) > eps_tol:
					pnorm2   = pnorm2   +  (a_j[i]/D[i])**2
					phiBar_g = phiBar_g + ((a_j[i])**2)/((D[i])**3)

			normP = torch.sqrt(pnorm2)
			phiBar = 1/normP - 1/delta
			phiBar_g = phiBar_g/(normP**3)
			return phiBar, phiBar_g

		p = a_j/D
		normP  = torch.norm(p)
		phiBar = 1/normP - 1/delta

		phiBar_g = torch.sum((a_j**2)/(D**3))
		phiBar_g = phiBar_g/(normP**3)

		return phiBar, phiBar_g

	def ComputeBySMW(self, tauStar, g, Y, S,gamma, Psig, invM, Psi, PsiPsi):
		vw = tauStar**2*invM + tauStar*PsiPsi 
		tmp = torch.inverse(vw) @ Psig
		Psitmp = Psi @ tmp
		sstar = -g/tauStar + Psitmp
		return sstar
