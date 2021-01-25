import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchsummary
import numpy as np

class identity_block(nn.Module):
	
	def __init__(self, in_channels, filters, output = False, adjust_size=False):
		super(identity_block, self).__init__()  
		
		self.output = output
		if self.output:
			filters1, filters2, filters3, filters4 = filters
			if adjust_size:
				self.conv_4 = nn.Conv2d(filters3, filters4, kernel_size=(1,1), stride=1, padding=2)
			else:
				self.conv_4 = nn.Conv2d(filters3, filters4, kernel_size=(1,1), stride=1)
			self.bn_4 = nn.BatchNorm2d(num_features=filters4 )
		else:
			filters1, filters2, filters3 = filters

		self.conv_1 = nn.Conv2d(in_channels, filters1, kernel_size=(1,1), stride=1)
		self.bn_1 = nn.BatchNorm2d(num_features= filters1)

		self.conv_2 = nn.Conv2d(filters1, filters2, kernel_size=(3,3), stride=1, padding=1)
		self.bn_2 = nn.BatchNorm2d(num_features= filters2)

		self.conv_3 = nn.Conv2d(filters2, filters3, kernel_size=(1,1), stride=1)
		self.bn_3 = nn.BatchNorm2d(num_features= filters3)

		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		identity = x 
		out = self.conv_1(x)
		out = self.bn_1(out)
		out = self.relu(out)
		out = self.conv_2(out)
		out = self.bn_2(out)
		out = self.relu(out)
		out = self.conv_3(out)
		out = self.bn_3(out)
		out = out+identity
		out = self.relu(out)

		if self.output:
			out_ = self.conv_4(out)
			out_ = self.bn_4(out_)
			return out , out_
		return out

class conv_block(nn.Module):

	def __init__(self, in_channels, filters):
		super(conv_block, self).__init__()

		filters1, filters2, filters3 = filters
		self.conv_1 = nn.Conv2d(in_channels, filters1, kernel_size=(2,2), stride=(2,2))
		self.bn_1 = nn.BatchNorm2d(num_features= filters1)

		self.conv_2 = nn.Conv2d(filters1, filters2, kernel_size=(3,3), stride=1, padding=1)
		self.bn_2 = nn.BatchNorm2d(num_features= filters2)

		self.conv_3 = nn.Conv2d(filters2, filters3, kernel_size=(1,1), stride=1)
		self.bn_3 = nn.BatchNorm2d(num_features= filters3)

		self.relu = nn.ReLU(inplace=True)
	
	def forward(self, x):
		shortcut = x 
		out = self.conv_1(x)
		out = self.bn_1(out)
		out = self.relu(out)
		out = self.conv_2(out)
		out = self.bn_2(out)
		out = self.relu(out)
		out = self.conv_3(out)
		out_long = self.bn_3(out)

		out_short = self.conv_1(shortcut)
		out_short = self.bn_1(out_short)
		out = self.relu(out_long+out_short)
		return out

class SegNet(nn.Module):
	def __init__(self, D_in, D_out):
		super().__init__()
		""" Segnet is a derivation of ResNet-50 and UNet.
		Network is design to output 3 prediction maps S,M,L fromthe different sections of the network
		"""

		# D_in: 3-channels
		# D_out: # of segmentations

		self.header = nn.Sequential(nn.Conv2d(in_channels= D_in, out_channels= 128, kernel_size=7, stride=1, padding=3),\
																nn.BatchNorm2d(128),\
																nn.ReLU())
		self.identity_1_1 = identity_block(in_channels=128, filters=[128,256,128])
		self.identity_1_2 = identity_block(128, [256,512,128,128], output=True )

		self.conv_2_1 = conv_block(128, [256,512,256])
		self.identity_2_2 = identity_block(256, [256,512,256])
		self.identity_2_3 = identity_block(256, [256,512,256,128], output=True )

		self.conv_3_1 = conv_block(256, [512,1024,512])
		self.identity_3_2 = identity_block(512, [512,512,512])
		self.identity_3_3 = identity_block(512, [512,512,512,256], output=True )

		self.deconv_4_1 = nn.ConvTranspose2d(512, 256, 3,  stride=1, padding=1)
		self.identity_4_2 = identity_block(256, [256,512,256])
		self.identity_4_3 = identity_block(256, [256,512,256,256], output=True )

		self.deconv_5_1 = nn.ConvTranspose2d(256, 128, 3,  stride=2, padding=1, output_padding=1)
		self.identity_5_2 = identity_block(128, [128,256,128])
		self.identity_5_3 = identity_block(128, [128,256,128,128], output=True)

		self.deconv_6_1 = nn.ConvTranspose2d(128, 128, 3,  stride=2, padding=1, output_padding=1)
		self.identity_6_2 = identity_block(128, [128,256,128])
		self.identity_6_3 = identity_block(128, [128,256,128,128], output=True )

		self.conv_out_small = nn.Conv2d(256*2, D_out, 1, stride=1)
		self.conv_out_medium = nn.Conv2d(128*2, D_out, 1, stride=1)
		self.conv_out_large = nn.Conv2d(128*2, D_out, 1, stride=1)

		self.relu = nn.ReLU(inplace=True)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
				
		y = self.header(x)
		y = self.identity_1_1(y)
		y, out_cnt_1_1 = self.identity_1_2(y)
		
		y = self.conv_2_1(y)
		y = self.identity_2_2(y)
		y, out_cnt_2_1 = self.identity_2_3(y)

		y = self.conv_3_1(y)
		y = self.identity_3_2(y)
		y, out_cnt_3_1 = self.identity_3_3(y)


		y_deconv = self.deconv_4_1(y)
		y = self.relu(y_deconv)
		y = self.identity_4_2(y)
		y, out_skip_4_1 = self.identity_4_3(y)
		out_cnt_3_2 = self.relu(out_skip_4_1 + y_deconv)

		y_deconv = self.deconv_5_1(y)
		y = self.relu(y_deconv)
		y = self.identity_5_2(y)
		y, out_skip_5_1 = self.identity_5_3(y)
		out_cnt_2_2 = self.relu(out_skip_5_1 + y_deconv)

		y_deconv = self.deconv_6_1(y)
		y = self.relu(y_deconv)
		y = self.identity_6_2(y)
		y, out_skip_6_1 = self.identity_6_3(y)
		out_cnt_1_2 = self.relu(out_skip_6_1 + y_deconv)

		cat_s = torch.cat((out_cnt_3_1, out_cnt_3_2), 1)
		cat_m = torch.cat((out_cnt_2_1, out_cnt_2_2), 1)
		cat_l = torch.cat((out_cnt_1_1, out_cnt_1_2), 1)
		print(cat_s.shape)


		out_small = self.conv_out_small(cat_s)
		out_medium = self.conv_out_medium(cat_m)
		out_large = self.conv_out_large(cat_l)

		out_small = self.sigmoid(out_small)
		out_medium = self.sigmoid(out_medium)
		out_large = self.sigmoid(out_large)

		return (out_small, out_medium, out_large)

