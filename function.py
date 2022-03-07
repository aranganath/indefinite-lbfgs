import torch
from torch.autograd import Variable
from indeflbfgstr import indefLBFGS
from ARCLSR1q import ARCLSR1
from torch import optim
import pickle as pkl

x = torch.tensor([500.0, 500.0], requires_grad=True)

y = torch.sum((x[1:] - x[:-1]**2)**2 + (torch.ones(x[:-1].shape[0]) - x[:-1])**2)

opt = 'indefLBFGS'

if opt == 'LBFGS':
	optimizer = optim.LBFGS([x], history_size=5, max_iter=1)

elif opt == 'indefLBFGS':
	optimizer = indefLBFGS([x], eta = 0.1, eta1=0.5, history_size=1, max_iters=1, deltacap=100, verbose=True)


L = []
for _ in range(100):
	optimizer.zero_grad()

	y = torch.sum((x[1:] - x[:-1]**2)**2 + (torch.ones(x[:-1].shape[0]) - x[:-1])**2)
	
	def closure():
		if torch.is_grad_enabled():
			optimizer.zero_grad()
		y = torch.sum((x[1:] - x[:-1]**2)**2 + (torch.ones(x[:-1].shape[0]) - x[:-1])**2)
		if y.requires_grad:
			y.backward()
		return y

	    #And optimizes its weights here
	y.backward()
	optimizer.step(closure=closure)
	print("x:" +str(x.data))
	print("Loss: "+str(y.item()))
	L.append(y.item())

# with open('./results/'+opt+'.pkl','wb') as handle:
# 	pkl.dump(L, handle)
