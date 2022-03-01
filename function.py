import torch
from torch.autograd import Variable
from indeflbfgstr import indefLBFGS
from ARCLSR1q import ARCLSR1
from torch import optim

x = Variable(torch.ones(5)*10, requires_grad=True)

y = torch.sum((x[1:] - x[:-1]**2)**2 + (torch.ones(x[:-1].shape[0]) - x[:-1])**2)

optimizer = indefLBFGS([x], eta = 0.0, eta1=0.0, history_size=2, max_iters=2)
# optimizer = ARCLSR1([x], gamma1 = 1, gamma2 =1.2, eta1 = 0.0, eta2 = 0.0, history_size =2, mu=1e5)
# optimizer = optim.LBFGS([x], history_size=2, max_iter=2)

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
