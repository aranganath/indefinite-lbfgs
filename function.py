import torch
from torch.autograd import Variable
from indeflbfgstr import indefLBFGS
from ARCLSR1q import ARCLSR1

x = Variable(torch.ones(10)*0.5, requires_grad=True)

y = torch.sum(100*(x[1:] - x[:-1]**2) + (torch.ones(x.shape[0]-1) - x[:-1])**2)

optimizer = indefLBFGS([x], eta = 0.15, eta1=0.3, history_size=10, max_iters=10)
optimizer = ARCLSR1([x], gamma1 = 1, gamma2 =10, eta1 = 0.15, eta2 = 0.25, history_size =10, mu=100)

for _ in range(10):
	def closure():
		return torch.sum(100*(x[1:] - x[:-1]**2) + (torch.ones(x.shape[0]-1) - x[:-1])**2)

	y.backward(retain_graph=True)
	optimizer.step(closure=closure)
	print(y)
