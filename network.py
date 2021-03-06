import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()

		self.p1 = 0.5

		self.b1_conv1 = nn.Conv2d(3, 8, 3, stride = 2, padding=1)
		self.b1_bn1 = nn.BatchNorm2d(8)
		self.b1_conv2 = nn.Conv2d(8, 16, 3, stride = 1, padding=1)
		self.b1_bn2 = nn.BatchNorm2d(16)
		self.b1_conv3 = nn.Conv2d(16, 32, 3, stride = 1, padding=1)
		self.b1_bn3 = nn.BatchNorm2d(32)


		
		self.b2_conv1 = nn.Conv2d(64, 48, 3, stride = 1, padding=1)
		self.b2_bn1 = nn.BatchNorm2d(48)
		self.b2_deconv1 = nn.ConvTranspose2d(48, 32, 2, stride=2)
		self.b2_bn_1 = nn.BatchNorm2d(32)


		self.b2_conv2 = nn.Conv2d(32, 16, 3, stride = 1, padding=1)
		self.b2_bn2 = nn.BatchNorm2d(16)
		# self.b2_deconv2 = nn.ConvTranspose2d(16, 16, 2, stride=2)
		# self.b2_bn_2 = nn.BatchNorm2d(16)


		self.b2_conv3 = nn.Conv2d(16, 8, 3, stride = 1, padding=1)
		self.b2_bn3 = nn.BatchNorm2d(8)
		self.b2_conv4 = nn.Conv2d(8, 2, 3, stride = 1, padding=1)
		
		
		





	def forward(self, x1, m1, p):

		self.p1 = p


		x1 = self.b1_bn1(F.relu(self.b1_conv1(x1)))
		x1 = self.b1_bn2(F.relu(self.b1_conv2(x1)))
		x1 = self.b1_bn3(F.relu(self.b1_conv3(x1)))

		m1 = self.b1_bn1(F.relu(self.b1_conv1(m1)))
		m1 = self.b1_bn2(F.relu(self.b1_conv2(m1)))
		m1 = self.b1_bn3(F.relu(self.b1_conv3(m1)))

		x1 = torch.cat([x1, m1], dim=1)


		x1 = self.b2_bn1(F.relu(self.b2_conv1(x1)))
		x1 = self.b2_bn_1(F.relu(self.b2_deconv1(x1)))

		x1 = self.b2_bn2(F.relu(self.b2_conv2(x1)))
		# x1 = self.b2_bn_2(F.relu(self.b2_deconv2(x1)))
		x1 = self.b2_bn3(F.relu(self.b2_conv3(x1)))
		x1 = F.relu(self.b2_conv4(x1))
		

		prob = F.softmax(x1, dim=1)
		

		return x1, prob