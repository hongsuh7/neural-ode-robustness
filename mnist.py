import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchdiffeq import odeint_adjoint as odeint

import matplotlib.pyplot as plt 
import numpy as np
import time

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(0.5, 0.5)]) # normalize to [-1,1]

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=4)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=True, num_workers=4)

############################################################################

class MyNet(nn.Module):
	def __init__(self, path):
		super(MyNet, self).__init__()
		self.path = path

	def num_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def load(self, adversarial = False):
		if adversarial:
			self.load_state_dict(torch.load('./' + self.path + '_adv.pth'))
		else:
			self.load_state_dict(torch.load('./' + self.path + '.pth'))

############################################################################

class ODEFunc(nn.Module):
	def __init__(self, dim):
		super(ODEFunc, self).__init__()
		self.gn = nn.GroupNorm(min(32, dim), dim)
		self.conv = nn.Conv2d(dim, dim, 3, padding = 1)
		self.nfe = 0 # time counter

	def forward(self, t, x):
		self.nfe += 1
		x = self.gn(x)
		x = F.relu(x)
		x = self.conv(x)
		x = self.gn(x)
		# currently odenet takes way too long to train on my macbook, make it smaller
		#x = F.relu(x)
		#x = self.conv(x)
		#x = self.gn(x)
		return x

############################################################################

class ODEBlock(nn.Module):
	def __init__(self, odefunc):
		super(ODEBlock, self).__init__()
		self.odefunc = odefunc
		self.integration_time = torch.tensor([0, 1]).float()

	def forward(self, x):
		out = odeint(self.odefunc, x, self.integration_time, rtol=1e-1, atol=1e-1)
		# first dimension is x(0) and second is x(1),
		# so we just want the second
		return out[1]

############################################################################

class ODENet(MyNet):
	def __init__(self):
		super(ODENet, self).__init__('mnist_odenet')
		self.conv1 = nn.Conv2d(1, 6, 3)
		self.gn = nn.GroupNorm(6, 6)
		self.odefunc = ODEFunc(6)
		self.odeblock = ODEBlock(self.odefunc)
		self.pool = nn.AvgPool2d(2)
		self.fc = nn.Linear(6 * 13 * 13, 10)

	def forward(self, x):
		# 26 x 26
		x = self.conv1(x)
		x = F.relu(self.gn(x))

		# stays 26 x 26
		x = self.odeblock(x)

		# 13 x 13
		x = self.pool(x)

		# fully connected layer
		x = x.view(-1, 6*13*13)
		x = self.fc(x)

		return x

############################################################################

class SmallCNN(MyNet):
	def __init__(self):
		super(SmallCNN, self).__init__('mnist_cnn')
		self.conv1 = nn.Conv2d(1, 6, 3)
		self.gn = nn.GroupNorm(6,6)
		self.conv2 = nn.Conv2d(6, 6, 3, padding = 1)
		self.pool = nn.AvgPool2d(2)
		self.fc = nn.Linear(6 * 13 * 13, 10)

	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(self.gn(x))

		# CNNBlock
		x = self.gn(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = self.gn(x)

		# FC
		x = self.pool(x)
		x = x.view(-1, 6*13*13)
		x = self.fc(x)
		return x

############################################################################

# Uses Runge-Kutta 4 to approximate the ODE block. Too slow.
class SmallResNetRK4(MyNet):
	def __init__(self):
		super(SmallResNetRK4, self).__init__('mnist_resnet_rk4')
		self.conv1 = nn.Conv2d(1, 6, 3)
		self.gn = nn.GroupNorm(6,6)
		self.conv2 = nn.Conv2d(6, 6, 3, padding = 1)
		self.pool = nn.AvgPool2d(2)
		self.fc = nn.Linear(6 * 13 * 13, 10)

	def forward(self, x):
		h = 1/2
		x = self.conv1(x)
		x = F.relu(self.gn(x))

		for _ in range(2):
			k1 = self.gn(self.conv2(F.relu(self.gn(x))))
			k2 = self.gn(self.conv2(F.relu(self.gn(x + h * k1/2))))
			k3 = self.gn(self.conv2(F.relu(self.gn(x + h * k2/2))))
			k4 = self.gn(self.conv2(F.relu(self.gn(x + h * k3))))
			x = x + (1/6) * h * (k1 + 2*k2 + 2*k3 + k4)

		# FC
		x = self.pool(x)
		x = x.view(-1, 6*13*13)
		x = self.fc(x)
		return x

############################################################################

class SmallResNetEuler(MyNet):
	def __init__(self, m):
		super(SmallResNetEuler, self).__init__('mnist_resnet_euler' + str(m))
		self.conv1 = nn.Conv2d(1, 6, 3)
		self.gn = nn.GroupNorm(6,6)
		self.conv2 = nn.Conv2d(6, 6, 3, padding = 1)
		self.pool = nn.AvgPool2d(2)
		self.fc = nn.Linear(6 * 13 * 13, 10)
		self.m = m

	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(self.gn(x))

		# Resblock
		for _ in range(self.m):
			x = x + (1/self.m) * self.gn(self.conv2(F.relu(self.gn(x))))

		# FC
		x = self.pool(x)
		x = x.view(-1, 6*13*13)
		x = self.fc(x)
		return x

############################################################################

def train(net, epochs, adversarial = False):
	n = 60000 / (5*16) 
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
	if adversarial:
		for epoch in range(epochs):  # loop over the dataset multiple times
			running_loss = 0.0
			for i, data in enumerate(trainloader, 0):
				# get the inputs; data is a list of [inputs, labels]
				inputs, labels = data
				inputs.requires_grad = True

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward + backward + optimize + adversary
				outputs = net(inputs)
				loss = criterion(outputs, labels)
				loss.backward()

				# adversary info
				grad = inputs.grad.data

				# optimize parameters
				optimizer.step()

				# adversarial training: optimize parameters
				outputs = net(torch.clamp(inputs + 0.15 * grad.sign(), -1,1))
				loss = 0.5 * criterion(outputs, labels)
				loss.backward()
				optimizer.step()

				# print statistics
				running_loss += loss.item()
				if i % n == n-1:    
					print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / n))
					running_loss = 0.0

		print('Finished Training')
		torch.save(net.state_dict(), './' + net.path + '_adv.pth')

	else:
		for epoch in range(epochs):  # loop over the dataset multiple times
			running_loss = 0.0
			for i, data in enumerate(trainloader, 0):
				# get the inputs; data is a list of [inputs, labels]
				inputs, labels = data

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward + backward + optimize
				outputs = net(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()

				# print statistics
				running_loss += loss.item()
				if i % n == n-1:    
					print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / n))
					running_loss = 0.0

		print('Finished Training')
		torch.save(net.state_dict(), './' + net.path + '.pth')


def test(net):
	initial_time = time.time()
	correct = 0
	total = 0
	with torch.no_grad():
		for data in testloader:
			images, labels = data
			batch_size = images.shape[0]
			outputs = net(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	final_time = time.time()
	print('Accuracy of the ' + net.path + ' network on the test set: %.2f %%' % (100 * correct / total))
	print('Time: %.2f seconds' % (final_time - initial_time))
	return(100 * correct / total)

def gaussian_test(net, sd):
	initial_time = time.time()
	correct = 0
	total = 0
	with torch.no_grad():
		for data in testloader:
			images, labels = data
			batch_size = images.shape[0]
			noise = np.random.normal(size = (batch_size,1,28,28), scale = sd)
			noise = torch.from_numpy(noise).float()
			outputs = net(torch.clamp(images + noise, -1, 1))
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	final_time = time.time()
	print('Standard deviation: %.1f' % sd)
	print('Accuracy of the ' + net.path + ' network on Gaussian noise attack: %.2f %%' % (100 * correct / total))
	print('Time: %.2f seconds' % (final_time - initial_time))
	return(100 * correct / total)

def fgsm_test(net, eps, show_image = False):
	initial_time = time.time()
	correct = 0
	total = 0
	for data in testloader:
		images, labels = data
		images.requires_grad = True
		net.zero_grad()

		outputs = net(images)
		loss = F.nll_loss(outputs, labels)
		loss.backward()
		grad = images.grad.data
		with torch.no_grad():
			perturbed_images = torch.clamp(images + eps * grad.sign(), -1,1)
			perturbed_outputs = net(perturbed_images)
			_, predicted = torch.max(perturbed_outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

	final_time = time.time()
	print('Epsilon: %.1f' % eps)
	print('Accuracy of the ' + net.path + ' network on an FGSM attack: %.2f %%' % (100 * correct / total))
	print('Time: %.2f seconds' % (final_time - initial_time))
	if show_image:
		imshow(perturbed_images[0,0])
	return(100 * correct / total)

def pgd_test(net, eps, niter = 10, show_image = False):
	initial_time = time.time()
	correct = 0
	total = 0
	for data in testloader:
		images, labels = data
		images.requires_grad = True
		net.zero_grad()

		original_images = images
		outputs = net(images)

		for i in range(niter):
			loss = F.nll_loss(outputs, labels)
			loss.backward()
			grad = images.grad.data
			with torch.no_grad():
				images = torch.clamp(images + eps/niter * grad.sign(), -1,1)
			images.requires_grad = True
			net.zero_grad()
			outputs = net(images)
			_, predicted = torch.max(outputs.data, 1)

		total += labels.size(0)
		correct += (predicted == labels).sum().item()
	final_time = time.time()
	print('Epsilon: %.1f' % eps)
	print('Accuracy of the ' + net.path + ' network on a PGD attack: %.2f %%' % (100 * correct / total))
	print('Time: %.2f seconds' % (final_time - initial_time))
	if show_image:
		imshow(images[0,0])
	return(100 * correct / total)

