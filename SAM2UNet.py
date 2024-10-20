from itertools import repeat
import math
from typing import Tuple
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_backbone import ResNet, buile_convnextv2, buile_segnext
from common import LayerNorm2d, Reshaper
from decoder import EMCAD
from decoder.DAPSAM.memory_prompt import PrototypePromptGenerate
from decoder.wrapped import SAMMaskDecoderWrapper_Med
from sam2.build_sam import build_sam2
from u_neck import RCM, Duck_Block
import utils

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
	from torch._six import container_abcs
else:
	import collections.abc as container_abcs

from config import config_base, path_config, config_neck, config_decoder, config_prompt_encoder

class DoubleConv(nn.Module):
	"""(convolution => [BN] => ReLU) * 2"""

	def __init__(self, in_channels, out_channels, mid_channels=None):
		super().__init__()
		if not mid_channels:
			mid_channels = out_channels
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(mid_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.double_conv(x)
	
	
class Up(nn.Module):
	"""Upscaling then double conv"""

	def __init__(self, in_channels, out_channels):
		super().__init__()

		self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

	def forward(self, x1, x2):
		x1 = self.up(x1)
		# input is CHW
		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]

		x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
						diffY // 2, diffY - diffY // 2])
		# if you have padding issues, see
		# https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
		# https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)


class Adapter(nn.Module):
	def __init__(self, blk) -> None:
		super(Adapter, self).__init__()
		self.block = blk
		dim = blk.attn.qkv.in_features
		self.prompt_learn = nn.Sequential(
			nn.Linear(dim, 32),
			nn.GELU(),
			nn.Linear(32, dim),
			nn.GELU()
		)

	def forward(self, x):
		prompt = self.prompt_learn(x)
		promped = x + prompt
		net = self.block(promped)
		return net
	

class BasicConv2d(nn.Module):
	def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
		super(BasicConv2d, self).__init__()
		self.conv = nn.Conv2d(in_planes, out_planes,
							  kernel_size=kernel_size, stride=stride,
							  padding=padding, dilation=dilation, bias=False)
		self.bn = nn.BatchNorm2d(out_planes)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		return x
	

class RFB_modified(nn.Module):
	def __init__(self, in_channel, out_channel):
		super(RFB_modified, self).__init__()
		self.relu = nn.ReLU(True)
		self.branch0 = nn.Sequential(
			BasicConv2d(in_channel, out_channel, 1),
		)
		self.branch1 = nn.Sequential(
			BasicConv2d(in_channel, out_channel, 1),
			BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
			BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
			BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
		)
		self.branch2 = nn.Sequential(
			BasicConv2d(in_channel, out_channel, 1),
			BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
			BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
			BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
		)
		self.branch3 = nn.Sequential(
			BasicConv2d(in_channel, out_channel, 1),
			BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
			BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
			BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
		)
		self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
		self.conv_res = BasicConv2d(in_channel, out_channel, 1)

	def forward(self, x):
		x0 = self.branch0(x)
		x1 = self.branch1(x)
		x2 = self.branch2(x)
		x3 = self.branch3(x)
		x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

		x = self.relu(x_cat + self.conv_res(x))
		return x




def to_2tuple(x):
	if isinstance(x, container_abcs.Iterable):
		return x
	return tuple(repeat(x, 2))


class OverlapPatchEmbed(nn.Module):
	""" Image to Patch Embedding
	"""

	def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
		super().__init__()
		img_size = to_2tuple(img_size)
		patch_size = to_2tuple(patch_size)

		self.img_size = img_size
		self.patch_size = patch_size
		self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
		self.num_patches = self.H * self.W
		self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
							  padding=(patch_size[0] // 2, patch_size[1] // 2))
		self.norm = nn.LayerNorm(embed_dim)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			trunc_normal_(m.weight, std=.02)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)
		elif isinstance(m, nn.Conv2d):
			fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
			fan_out //= m.groups
			m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
			if m.bias is not None:
				m.bias.data.zero_()

	def forward(self, x):
		x = self.proj(x)
		_, _, H, W = x.shape
		x = x.flatten(2).transpose(1, 2)
		x = self.norm(x)

		return x, H, W

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
	# type: (Tensor, float, float, float, float) -> Tensor
	r"""Fills the input Tensor with values drawn from a truncated
	normal distribution. The values are effectively drawn from the
	normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
	with values outside :math:`[a, b]` redrawn until they are within
	the bounds. The method used for generating the random values works
	best when :math:`a \leq \text{mean} \leq b`.
	Args:
		tensor: an n-dimensional `torch.Tensor`
		mean: the mean of the normal distribution
		std: the standard deviation of the normal distribution
		a: the minimum cutoff value
		b: the maximum cutoff value
	Examples:
		>>> w = torch.empty(3, 5)
		>>> nn.init.trunc_normal_(w)
	"""
	return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
	# Cut & paste from PyTorch official master until it's in a few official releases - RW
	# Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
	def norm_cdf(x):
		# Computes standard normal cumulative distribution function
		return (1. + math.erf(x / math.sqrt(2.))) / 2.

	if (mean < a - 2 * std) or (mean > b + 2 * std):
		warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
					  "The distribution of values may be incorrect.",
					  stacklevel=2)

	with torch.no_grad():
		# Values are generated by using a truncated uniform distribution and
		# then using the inverse CDF for the normal distribution.
		# Get upper and lower cdf values
		l = norm_cdf((a - mean) / std)
		u = norm_cdf((b - mean) / std)

		# Uniformly fill tensor with values from [l, u], then translate to
		# [2l-1, 2u-1].
		tensor.uniform_(2 * l - 1, 2 * u - 1)

		# Use inverse cdf transform for normal distribution to get truncated
		# standard normal
		tensor.erfinv_()

		# Transform to proper mean, std
		tensor.mul_(std * math.sqrt(2.))
		tensor.add_(mean)

		# Clamp to ensure it's in the proper range
		tensor.clamp_(min=a, max=b)
		return tensor

class PromptGenerator(nn.Module):
	def __init__(self, scale_factor, prompt_type, embed_dims, tuning_stage, depths, input_type,
				 freq_nums, handcrafted_tune, embedding_tune, adaptor, img_size):
		"""
		Args:
		"""
		super(PromptGenerator, self).__init__()
		self.scale_factor = scale_factor
		self.prompt_type = prompt_type
		self.embed_dims = embed_dims
		self.input_type = input_type
		self.freq_nums = freq_nums
		self.tuning_stage = tuning_stage
		self.depths = depths
		self.handcrafted_tune = handcrafted_tune
		self.embedding_tune = embedding_tune
		self.adaptor = adaptor

		# 默认 fft，这里不处理
		if self.input_type == 'gaussian':
			self.gaussian_filter = GaussianFilter()
		if self.input_type == 'srm':
			self.srm_filter = SRMFilter()
		if self.input_type == 'all':
			self.prompt = nn.Parameter(torch.zeros(3, img_size, img_size), requires_grad=False)

		if self.handcrafted_tune:
			if '1' in self.tuning_stage:
				self.handcrafted_generator1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=3,
														embed_dim=self.embed_dims[0] // self.scale_factor)
			if '2' in self.tuning_stage:
				self.handcrafted_generator2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2,
													   in_chans=self.embed_dims[0] // self.scale_factor,
													   embed_dim=self.embed_dims[1] // self.scale_factor)
			if '3' in self.tuning_stage:
				self.handcrafted_generator3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2,
													   in_chans=self.embed_dims[1] // self.scale_factor,
													   embed_dim=self.embed_dims[2] // self.scale_factor)
			if '4' in self.tuning_stage:
				self.handcrafted_generator4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2,
													   in_chans=self.embed_dims[2] // self.scale_factor,
													   embed_dim=self.embed_dims[3] // self.scale_factor)

		if self.embedding_tune:
			if '1' in self.tuning_stage:
				self.embedding_generator1 = nn.Linear(self.embed_dims[0], self.embed_dims[0] // self.scale_factor)
			if '2' in self.tuning_stage:
				self.embedding_generator2 = nn.Linear(self.embed_dims[1], self.embed_dims[1] // self.scale_factor)
			if '3' in self.tuning_stage:
				self.embedding_generator3 = nn.Linear(self.embed_dims[2], self.embed_dims[2] // self.scale_factor)
			if '4' in self.tuning_stage:
				self.embedding_generator4 = nn.Linear(self.embed_dims[3], self.embed_dims[3] // self.scale_factor)

		if self.adaptor == 'adaptor':
			print('adaptor_label = adaptor')
			if '1' in self.tuning_stage:
				for i in range(self.depths[0]+1):
					lightweight_mlp = nn.Sequential(
							nn.Linear(self.embed_dims[0] // self.scale_factor, self.embed_dims[0] // self.scale_factor),
							nn.GELU(),
						)
					setattr(self, 'lightweight_mlp1_{}'.format(str(i)), lightweight_mlp)
				self.shared_mlp1 = nn.Linear(self.embed_dims[0] // self.scale_factor, self.embed_dims[0])

			if '2' in self.tuning_stage:
				for i in range(self.depths[1]+1):
					lightweight_mlp = nn.Sequential(
							nn.Linear(self.embed_dims[1] // self.scale_factor, self.embed_dims[1] // self.scale_factor),
							nn.GELU(),
						)
					setattr(self, 'lightweight_mlp2_{}'.format(str(i)), lightweight_mlp)
				self.shared_mlp2 = nn.Linear(self.embed_dims[1] // self.scale_factor, self.embed_dims[1])

			if '3' in self.tuning_stage:
				for i in range(self.depths[2]+1):
					lightweight_mlp = nn.Sequential(
							nn.Linear(self.embed_dims[2] // self.scale_factor, self.embed_dims[2] // self.scale_factor),
							nn.GELU(),
						)
					setattr(self, 'lightweight_mlp3_{}'.format(str(i)), lightweight_mlp)
				self.shared_mlp3 = nn.Linear(self.embed_dims[2] // self.scale_factor, self.embed_dims[2])

			if '4' in self.tuning_stage:
				for i in range(self.depths[3]+1):
					lightweight_mlp = nn.Sequential(
							nn.Linear(self.embed_dims[3] // self.scale_factor, self.embed_dims[3] // self.scale_factor),
							nn.GELU(),
						)
					setattr(self, 'lightweight_mlp4_{}'.format(str(i)), lightweight_mlp)
				self.shared_mlp4 = nn.Linear(self.embed_dims[3] // self.scale_factor, self.embed_dims[3])

		elif self.adaptor == 'fully_shared':
			print('adaptor_label = fully_shared')
			self.fully_shared_mlp1 = nn.Sequential(
						nn.Linear(self.embed_dims[0] // self.scale_factor, self.embed_dims[0] // self.scale_factor),
						nn.GELU(),
						nn.Linear(self.embed_dims[0] // self.scale_factor, self.embed_dims[0])
					)
			self.fully_shared_mlp2 = nn.Sequential(
						nn.Linear(self.embed_dims[1] // self.scale_factor, self.embed_dims[1] // self.scale_factor),
						nn.GELU(),
						nn.Linear(self.embed_dims[1] // self.scale_factor, self.embed_dims[1])
					)
			self.fully_shared_mlp3 = nn.Sequential(
						nn.Linear(self.embed_dims[2] // self.scale_factor, self.embed_dims[2] // self.scale_factor),
						nn.GELU(),
						nn.Linear(self.embed_dims[2] // self.scale_factor, self.embed_dims[2])
					)
			self.fully_shared_mlp4 = nn.Sequential(
						nn.Linear(self.embed_dims[3] // self.scale_factor, self.embed_dims[3] // self.scale_factor),
						nn.GELU(),
						nn.Linear(self.embed_dims[3] // self.scale_factor, self.embed_dims[3])
					)

		elif self.adaptor == 'fully_unshared':
			print('adaptor_label = fully_unshared')
			for i in range(self.depths[0]):
				fully_unshared_mlp1 = nn.Sequential(
					nn.Linear(self.embed_dims[0] // self.scale_factor, self.embed_dims[0] // self.scale_factor),
					nn.GELU(),
					nn.Linear(self.embed_dims[0] // self.scale_factor, self.embed_dims[0])
				)
				setattr(self, 'fully_unshared_mlp1_{}'.format(str(i)), fully_unshared_mlp1)
			for i in range(self.depths[1]):
				fully_unshared_mlp1 = nn.Sequential(
					nn.Linear(self.embed_dims[1] // self.scale_factor, self.embed_dims[1] // self.scale_factor),
					nn.GELU(),
					nn.Linear(self.embed_dims[1] // self.scale_factor, self.embed_dims[1])
				)
				setattr(self, 'fully_unshared_mlp2_{}'.format(str(i)), fully_unshared_mlp1)
			for i in range(self.depths[2]):
				fully_unshared_mlp1 = nn.Sequential(
					nn.Linear(self.embed_dims[2] // self.scale_factor, self.embed_dims[2] // self.scale_factor),
					nn.GELU(),
					nn.Linear(self.embed_dims[2] // self.scale_factor, self.embed_dims[2])
				)
				setattr(self, 'fully_unshared_mlp3_{}'.format(str(i)), fully_unshared_mlp1)
			for i in range(self.depths[3]):
				fully_unshared_mlp1 = nn.Sequential(
					nn.Linear(self.embed_dims[3] // self.scale_factor, self.embed_dims[3] // self.scale_factor),
					nn.GELU(),
					nn.Linear(self.embed_dims[3] // self.scale_factor, self.embed_dims[3])
				)
				setattr(self, 'fully_unshared_mlp4_{}'.format(str(i)), fully_unshared_mlp1)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			trunc_normal_(m.weight, std=.02)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)
		elif isinstance(m, nn.Conv2d):
			fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
			fan_out //= m.groups
			m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
			if m.bias is not None:
				m.bias.data.zero_()

	def init_handcrafted(self, x):
		"""
		根据输入类型和调优阶段初始化手工制作的特征图。

		参数:
			x (Tensor): 输入数据，根据输入类型可能代表不同的信号或图像。

		返回:
			handcrafted1, handcrafted2, handcrafted3, handcrafted4 (Tensor): 根据调优阶段生成的手工制作特征图。
		"""
		# 根据输入类型处理输入数据
		if self.input_type == 'fft':
			x = self.fft(x, self.freq_nums, self.prompt_type)

		elif self.input_type == 'all':
			x = self.prompt.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)

		elif self.input_type == 'gaussian':
			x = self.gaussian_filter.conv_gauss(x)

		elif self.input_type == 'srm':
			x = self.srm_filter.srm_layer(x)

		# 返回 x
		B = x.shape[0]
		# 获取提示

		# 根据调优阶段生成手工制作的特征图
		if '1' in self.tuning_stage:
			handcrafted1, H1, W1 = self.handcrafted_generator1(x)
		else:
			handcrafted1 = None

		if '2' in self.tuning_stage:
			handcrafted2, H2, W2 = self.handcrafted_generator2(handcrafted1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous())
		else:
			handcrafted2 = None

		if '3' in self.tuning_stage:
			handcrafted3, H3, W3 = self.handcrafted_generator3(handcrafted2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous())
		else:
			handcrafted3 = None

		if '4' in self.tuning_stage:
			handcrafted4, H4, W4 = self.handcrafted_generator4(handcrafted3.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous())
		else:
			handcrafted4 = None

		return handcrafted1, handcrafted2, handcrafted3, handcrafted4

	def init_prompt(self, embedding_feature, handcrafted_feature, block_num):
		"""
		初始化提示信息，根据配置调整嵌入特征和手工特征。
		
		参数:
		- embedding_feature: 嵌入特征，通常是从某种预训练模型中提取的特征。
		- handcrafted_feature: 手工特征，通常是指通过特定算法设计的特征。
		- block_num: 块编号，用于选择特定的嵌入特征生成器。
		
		返回:
		- handcrafted_feature: 经过调整后的手工特征。
		- embedding_feature: 经过调整后的嵌入特征。
		"""
		# 如果配置了调整嵌入特征
		if self.embedding_tune:
			# 动态获取对应块编号的嵌入特征生成器
			embedding_generator = getattr(self, 'embedding_generator{}'.format(str(block_num)))
			# 使用嵌入特征生成器调整嵌入特征
			# 线性层处理
			embedding_feature = embedding_generator(embedding_feature)
		
		# 如果配置了调整手工特征
		if self.handcrafted_tune:
			# 保持手工特征不变
			handcrafted_feature = handcrafted_feature

		# 返回调整后的手工特征和嵌入特征
		return handcrafted_feature, embedding_feature

	def get_embedding_feature(self, x, block_num):
		if self.embedding_tune:
			embedding_generator = getattr(self, 'embedding_generator{}'.format(str(block_num)))
			embedding_feature = embedding_generator(x)

			return embedding_feature
		else:
			return None

	def get_handcrafted_feature(self, x, block_num):
		if self.handcrafted_tune:
			handcrafted_generator = getattr(self, 'handcrafted_generator{}'.format(str(block_num)))
			handcrafted_feature = handcrafted_generator(x)

			return handcrafted_feature
		else:
			return None

	def get_prompt(self, x, prompt, block_num, depth_num):
		"""
		根据给定的提示信息和当前的特征图，计算并返回更新后的特征图。

		该方法首先根据配置决定是否使用手工调整或嵌入调整来更新特征图。
		然后根据适配器的类型（轻量级MLP、完全共享MLP或完全不共享MLP），
		使用相应的MLP层对特征图进行进一步的处理。最后，将处理后的特征图与输入的特征图x相加，
		得到最终的输出特征图。

		参数:
		- x: 输入的特征图，将被更新。
		- prompt: 提示信息，包含可能用于更新特征图的多种提示。
		- block_num: 当前处理块的编号，用于动态获取MLP层。
		- depth_num: 当前处理深度的编号，用于动态获取MLP层。

		返回:
		- 更新后的特征图。
		"""
		# 初始化特征变量
		feat = 0
		# 获取提示信息的形状
		B, H, W =  prompt[1].shape[0],  prompt[1].shape[1],  prompt[1].shape[2]

		# 如果启用了手工调整功能，则将提示信息的第0个元素重塑后加到特征上
		if self.handcrafted_tune:
			feat += prompt[0].reshape(B, H, W, -1)

		# 如果启用了嵌入调整功能，则将提示信息的第1个元素加到特征上
		if self.embedding_tune:
			# 此处代码被注释掉，因为embedding_tune功能当前未启用
			# if False:
			feat += prompt[1]

		# 根据适配器类型，选择并应用相应的MLP层处理特征
		if self.adaptor == 'adaptor':
			# 动态获取轻量级MLP和共享MLP层，并依次应用到特征上
			lightweight_mlp = getattr(self, 'lightweight_mlp' + str(block_num) + '_' + str(depth_num))
			shared_mlp = getattr(self, 'shared_mlp' + str(block_num))

			feat = lightweight_mlp(feat)
			feat = shared_mlp(feat)

		elif self.adaptor == 'fully_shared':
			# 动态获取完全共享MLP层，并应用到特征上
			fully_shared_mlp = getattr(self, 'fully_shared_mlp' + str(block_num))
			feat = fully_shared_mlp(feat)

		elif self.adaptor == 'fully_unshared':
			# 动态获取完全不共享MLP层，并应用到特征上
			fully_unshared_mlp = getattr(self, 'fully_unshared_mlp' + str(block_num) + '_' + str(depth_num))
			feat = fully_unshared_mlp(feat)

		# 将处理后的特征与输入的特征图x相加，得到最终的输出特征图
		x = x + feat

		# 返回更新后的特征图
		return x

	def fft(self, x, rate, prompt_type):
		"""
		根据指定类型（高通或低通）对输入信号x进行傅里叶变换过滤。

		参数:
		x (torch.Tensor): 输入的信号数据，通常为二维图像。
		rate (float): 用于计算遮罩大小的比率，决定滤波器的截止频率。
		prompt_type (str): 提示类型，可以是'highpass'（高通滤波器）或'lowpass'（低通滤波器）。

		返回:
		torch.Tensor: 经过傅里叶变换和过滤后的信号，其大小和输入信号x相同。
		"""
		# 初始化一个与输入信号x形状相同的零张量作为遮罩
		mask = torch.zeros(x.shape).to('cuda')
		# 获取输入信号x的宽度和高度
		w, h = x.shape[-2:]
		# 计算遮罩的大小，基于给定的比率rate
		line = int((w * h * rate) ** .5 // 2)
		# 在遮罩的中心位置根据计算出的line值创建一个方形区域，并设置为1
		mask[:, :, w//2-line:w//2+line, h//2-line:h//2+line] = 1

		# 对输入信号x执行二维傅里叶变换，并进行fftshift以将低频移到中心
		fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))

		# 根据指定的prompt_type应用高通或低通滤波器
		if prompt_type == 'highpass':
			# 对于高通滤波器，保留高频成分
			fft = fft * (1 - mask)
		elif prompt_type == 'lowpass':
			# 对于低通滤波器，保留低频成分
			fft = fft * mask

		# 分离傅里叶变换结果的实部和虚部
		fr = fft.real
		fi = fft.imag

		# 使用实部和虚部进行逆傅里叶变换的准备
		fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
		# 执行二维逆傅里叶变换，并计算实部
		inv = torch.fft.ifft2(fft_hires, norm="forward").real

		# 计算逆变换结果的绝对值
		inv = torch.abs(inv)

		# 返回处理后的信号
		return inv



class GaussianFilter(nn.Module):
	def __init__(self):
		super(GaussianFilter, self).__init__()
		self.kernel = self.gauss_kernel()

	def gauss_kernel(self, channels=3):
		kernel = torch.tensor([[1., 4., 6., 4., 1],
							   [4., 16., 24., 16., 4.],
							   [6., 24., 36., 24., 6.],
							   [4., 16., 24., 16., 4.],
							   [1., 4., 6., 4., 1.]])
		kernel /= 256.
		kernel = kernel.repeat(channels, 1, 1, 1)
		kernel = kernel.to(torch.device)
		return kernel

	def conv_gauss(self, img):
		img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
		out = torch.nn.functional.conv2d(img, self.kernel, groups=img.shape[1])
		return out


class SRMFilter(nn.Module):
	def __init__(self):
		super(SRMFilter, self).__init__()
		self.srm_layer = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2,)
		filter1 = [[0, 0, 0, 0, 0],
				   [0, -1 / 4, 2 / 4, -1 / 4, 0],
				   [0, 2 / 4, -4 / 4, 2 / 4, 0],
				   [0, -1 / 4, 2 / 4, -1 / 4, 0],
				   [0, 0, 0, 0, 0]]
		filter2 = [[-1 / 12, 2 / 12, -2 / 12, 2 / 12, -1 / 12],
				   [2 / 12, -6 / 12, 8 / 12, -6 / 12, 2 / 12],
				   [-2 / 12, 8 / 12, -12 / 12, 8 / 12, -2 / 12],
				   [2 / 12, -6 / 12, 8 / 12, -6 / 12, 2 / 12],
				   [-1 / 12, 2 / 12, -2 / 12, 2 / 12, -1 / 12]]
		filter3 = [[0, 0, 0, 0, 0],
				   [0, 0, 0, 0, 0],
				   [0, 1 / 2, -2 / 2, 1 / 2, 0],
				   [0, 0, 0, 0, 0],
				   [0, 0, 0, 0, 0]]
		self.srm_layer.weight.data = torch.Tensor(
			[[filter1, filter1, filter1],
			 [filter2, filter2, filter2],
			 [filter3, filter3, filter3]]
		)

		for param in self.srm_layer.parameters():
			param.requires_grad = False

	def conv_srm(self, img):
		out = self.srm_layer(img)
		return out



class SAM2UNet(nn.Module):
	def __init__(self, checkpoint_path=None, adapter_type="adaptor") -> None:
		super(SAM2UNet, self).__init__()    
		model_cfg = "sam2_hiera_l.yaml"
		if checkpoint_path:
			model = build_sam2(model_cfg, checkpoint_path)
		else:
			model = build_sam2(model_cfg)
		# del model.sam_mask_decoder
		del model.sam_prompt_encoder
		del model.memory_encoder
		del model.memory_attention
		del model.mask_downsample
		del model.obj_ptr_tpos_proj
		del model.obj_ptr_proj
		del model.image_encoder.neck
		self.encoder = model.image_encoder.trunk

		if(config_prompt_encoder['prompt_type'] == 'DAPSAM'):
			self.mask_decoder = model.sam_mask_decoder
			self.prompt = PrototypePromptGenerate()			
			self.adapte_conv_down = Reshaper(16 , 1152, 32, 256)    
			self.adapte_conv_top = Reshaper(128 , 144, 128, 32)    
			self.mask_decoder = SAMMaskDecoderWrapper_Med(ori_sam=model)
			
			
		for param in self.encoder.parameters():
			param.requires_grad = False
		
		# blocks = []
		# for block in self.encoder.blocks:
		#     blocks.append(
		#         Adapter(block)
		#     )
		# self.encoder.blocks = nn.Sequential(
		#     *blocks
		# )
# ============================ Adapter ============================
		self.embed_dim = [144, 288, 576, 1152]
		self.depth = [2, 6, 36, 4]
		self.blocks = nn.ModuleList()
		self.scale_factor = 32
		self.prompt_type = 'highpass'
		self.tuning_stage = "1234"
		self.input_type = 'fft'
		self.freq_nums = 0.25
		self.handcrafted_tune = True
		self.embedding_tune = True
		self.adaptor = adapter_type
		self.prompt_generator = PromptGenerator(self.scale_factor, self.prompt_type, self.embed_dim,
												self.tuning_stage, self.depth,
												self.input_type, self.freq_nums,
												self.handcrafted_tune, self.embedding_tune, self.adaptor,
												img_size=config_base['image_size'])
# ============================ neck ============================
		for neck_type in config_neck['neck_type']:
			if(neck_type == 'None'):		# 此时，neck不操作
				break
			if(neck_type == 'RCM'):
				self.rcm1 = RCM(144)
				self.rcm2 = RCM(288)
				self.rcm3 = RCM(576)
				self.rcm4 = RCM(1152)
			
			if(neck_type == 'RFB'):
				self.rfb1 = RFB_modified(144, 64)
				self.rfb2 = RFB_modified(288, 64)
				self.rfb3 = RFB_modified(576, 64)
				self.rfb4 = RFB_modified(1152, 64)

			if(neck_type == 'Duck'):
				self.duck1 = Duck_Block(144, 64)
				self.duck2 = Duck_Block(288, 64)
				self.duck3 = Duck_Block(576, 64)
				self.duck4 = Duck_Block(1152, 64)
				
# ============================ decoder ============================
		if(config_decoder['decoder_type'] == 'UNet'):
			self.up1 = (Up(128, 64))
			self.up2 = (Up(128, 64))
			self.up3 = (Up(128, 64))
			self.up4 = (Up(128, 64))
			self.side1 = nn.Conv2d(64, 1, kernel_size=1)
			self.side2 = nn.Conv2d(64, 1, kernel_size=1)
			self.side3 = nn.Conv2d(64, 1, kernel_size=1)
			self.head = nn.Conv2d(64, 1, kernel_size=1)

		elif(config_decoder['decoder_type'] == 'EMCAD'):
			channels = [1152, 576, 288, 144]
			self.decoder = EMCAD(channels=channels, kernel_sizes=[1,3,5], 
						expansion_factor=2, dw_parallel=True, add=True, lgag_ks=3, activation='relu')
			print('Model %s created, param count: %d' %
						('EMCAD decoder: ', sum([m.numel() for m in self.decoder.parameters()])))
			self.out_head4 = nn.Conv2d(channels[0], 1, 1)
			self.out_head3 = nn.Conv2d(channels[1], 1, 1)
			self.out_head2 = nn.Conv2d(channels[2], 1, 1)
			self.out_head1 = nn.Conv2d(channels[3], 1, 1)

# ============================ cnn ============================
		if(config_base['cnn_label'] == 'none'):
			self.cnn_backbone = None

		elif(config_base['cnn_label'] == 'resnet50'):
			self.cnn_backbone = ResNet(50, False)

		elif(config_base['cnn_label'] == 'convnextv2_base'):
			self.cnn_backbone = buile_convnextv2['convnextv2_base']()
			checkpoint = torch.load(path_config['convnextv2_base_ckpt_path'], map_location='cpu')
			print("Load pre-trained checkpoint from: %s" % path_config['convnextv2_base_ckpt_path'])
			checkpoint_model = checkpoint['model']
			state_dict = self.cnn_backbone.state_dict()
			
			# checkpoint_model = remap_checkpoint_keys(checkpoint_model)
			utils.load_state_dict(self.cnn_backbone, checkpoint_model, prefix='')	
			# self.cnn_backbone.load_state_dict(checkpoint_model, strict=False)
		
			print("convnextv2_base_ckpt 添加成功 ~ ~")
			for param in self.cnn_backbone.parameters():
				param.requires_grad = False

		elif(config_base['cnn_label'] == 'segnext_base'): 
			self.cnn_backbone = buile_segnext['segnext_base']()
			checkpoint = torch.load(path_config['segnext_base_ckpt_path'], map_location='cpu')
			print("Load pre-trained checkpoint from: %s" % path_config['segnext_base_ckpt_path'])
			
			self.cnn_backbone.load_state_dict(checkpoint, strict=False)
		
			print("convnextv2_base_ckpt 添加成功 ~ ~")
			for param in self.cnn_backbone.parameters():
				param.requires_grad = False
			

		if(self.cnn_backbone != None):
			channel_list = self.cnn_backbone.channel_list
			feature_size = self.cnn_backbone.feature_size
			embed_dim = config_base['embed_dim']
			num_patchs = config_base['num_patchs']
			self.adapters_in = nn.ModuleList([Reshaper(feature_size[0], channel_list[0], num_patchs, embed_dim),     
											Reshaper(feature_size[1], channel_list[1], num_patchs, embed_dim),                                 
											Reshaper(feature_size[2], channel_list[2], num_patchs, embed_dim),    
											Reshaper(feature_size[3], channel_list[3], num_patchs, embed_dim),])
	
	def encoder_forward(self, feature_maps: torch.Tensor, x: torch.Tensor):
		'''
		每一次前向传播，每一张图片，先 init_handcrafted，通过 fft 获取一些特征信息，通过线性层处理Hiera的中间特征层
		接着通过 init_prompt，将两个特征层 embed
		通过get_prompt，两个特征层相加，再经过 Adapter Mlp 结构，返回最后的结构
		同一个块结构，使用同一个 Adapter Mlp ，对融合的特征层进行处理。
		'''
		inp = x
		x = self.encoder.patch_embed(x)
		# x: (B, H, W, C)
		handcrafted1, handcrafted2, handcrafted3, handcrafted4 = self.prompt_generator.init_handcrafted(inp)

		self.block1 = []
		self.block2 = []
		self.block3 = []
		self.block4 = []
		outputs = []

		for i, blk in enumerate(self.encoder.blocks):
			if i < 3:
				self.block1.append(blk)  # 第一个块包含前3个元素
			elif 2 < i < 9:
				self.block2.append(blk)  # 第二个块包含接下来的6个元素
			elif 8 < i < 45:
				self.block3.append(blk)  # 第三个块包含接下来的36个元素
			elif 44 < i:
				self.block4.append(blk)  # 其余元素组成第四个块

		# Add pos embed
		x = x + self.encoder._get_pos_embed(x.shape[1:3])

		if '1' in self.tuning_stage:
			prompt1 = self.prompt_generator.init_prompt(x, handcrafted1, 1)
		for i, blk in enumerate(self.block1):
			if '1' in self.tuning_stage:
				x = self.prompt_generator.get_prompt(x, prompt1, 1, i)
			x = blk(x)
		# x = self.norm1(x)
			if i == 1:
				feat = x.permute(0, 3, 1, 2)
				outputs.append(feat)

		if '2' in self.tuning_stage:
			prompt2 = self.prompt_generator.init_prompt(x, handcrafted2, 2)
		for i, blk in enumerate(self.block2):
			if '2' in self.tuning_stage:
				x = self.prompt_generator.get_prompt(x, prompt2, 2, i)
			x = blk(x)
		# x = self.norm2(x)
			if i == 4:
				feat = x.permute(0, 3, 1, 2)
				outputs.append(feat)

		if '3' in self.tuning_stage:
			prompt3 = self.prompt_generator.init_prompt(x, handcrafted3, 3)
		
		index = 0
		for i, blk in enumerate(self.block3):
			if '3' in self.tuning_stage:
				x = self.prompt_generator.get_prompt(x,prompt3, 3, i)
			# 插入cnn特征图
			if feature_maps != [None] * 4:
				if(i in [4, 14, 24, 34]):
					# print(f"此时x的shape {x.shape}")    # 4, 14, 24, 34  torch.Size([2, 32, 32, 576])
					# print(f"feature_maps的shape {feature_maps[index].shape}")
					x = x + feature_maps[index]

			x = blk(x)
		# x = self.norm3(x)
			if i == 34:
				feat = x.permute(0, 3, 1, 2)
				outputs.append(feat)

		if '4' in self.tuning_stage:
			prompt4 = self.prompt_generator.init_prompt(x, handcrafted4, 4)
		for i, blk in enumerate(self.block4):
			if '4' in self.tuning_stage:
				x = self.prompt_generator.get_prompt(x, prompt4, 4, i)
			x = blk(x)
		# x = self.norm4(x)
			if i == 2:
				feat = x.permute(0, 3, 1, 2)
				outputs.append(feat)
		return outputs
		
	def forward(self, x):
		feature_maps_in_new = [None] * 4
		if(self.cnn_backbone != None):
			feature_maps = self.cnn_backbone(x)
			for i in range(len(feature_maps)):
				feature_maps_in_new[i] = self.adapters_in[i](feature_maps[i])
				feature_maps_in_new[i] = feature_maps_in_new[i].permute(0, 2, 3, 1)

		x1, x2, x3, x4 = self.encoder_forward(feature_maps_in_new, x)
		ori_x1, ori_x2, ori_x3, ori_x4 = x1, x2, x3, x4
		# x1, x2, x3, x4 = self.encoder(x, self.prompt_generator, self.tuning_stage, feature_maps_in_new)
		# print(f"图像经过hiera的尺寸变化 {x.shape} {x1.shape} {x2.shape} {x3.shape} {x4.shape}")
		# 图像经过hiera的尺寸变化 
		# torch.Size([2, 3, 512, 512]) 
		# torch.Size([2, 144, 128, 128]) 
		# torch.Size([2, 288, 64, 64]) 
		# torch.Size([2, 576, 32, 32]) 
		# torch.Size([2, 1152, 16, 16])
		for neck_type in config_neck['neck_type']:
			if(neck_type == 'None'):		# 此时，neck不操作
				break

			if(neck_type == 'RCM'):
				x1, x2, x3, x4 = self.rcm1(x1), self.rcm2(x2), self.rcm3(x3), self.rcm4(x4)
			
			if(neck_type == 'RFB'):
				x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
		
			if(neck_type == 'Duck'):
				x1, x2, x3, x4 = self.duck1(x1), self.duck2(x2), self.duck3(x3), self.duck4(x4)
		
		
		if(config_decoder['decoder_type'] == 'UNet'):
			x = x4
			out3 = F.interpolate(self.side3(x), scale_factor=32, mode='bilinear')
			x = self.up1(x4, x3)
			out2 = F.interpolate(self.side2(x), scale_factor=16, mode='bilinear')
			x = self.up2(x, x2)
			out1 = F.interpolate(self.side1(x), scale_factor=8, mode='bilinear')
			x = self.up3(x, x1)
			out = F.interpolate(self.head(x), scale_factor=4, mode='bilinear')
		elif(config_decoder['decoder_type'] == 'EMCAD'):
			dec_outs = self.decoder(x4, [x3, x2, x1])
			
			p4 = self.out_head4(dec_outs[0])		# 1*1 conv 调通道
			p3 = self.out_head3(dec_outs[1])
			p2 = self.out_head2(dec_outs[2])
			p1 = self.out_head1(dec_outs[3])
			out3 = F.interpolate(p4, scale_factor=32, mode='bilinear')
			out2 = F.interpolate(p3, scale_factor=16, mode='bilinear')
			out1 = F.interpolate(p2, scale_factor=8, mode='bilinear')
			out = F.interpolate(p1, scale_factor=4, mode='bilinear')
			
			if(config_prompt_encoder['prompt_type'] == 'DAPSAM'):
				image_embedding = self.adapte_conv_down(ori_x4)				# 256*32*32
				multi_scale_feature = self.adapte_conv_top(dec_outs[3])		
				sparse_embeddings, prompt = self.prompt(image_embedding)
					
				low_res_masks, iou_predictions = self.mask_decoder(
					image_embeddings=image_embedding,
					image_pe=self.prompt.get_dense_pe(),
					sparse_prompt_embeddings=sparse_embeddings,
					dense_prompt_embeddings=prompt,
					multi_scale_feature = multi_scale_feature,
				)
				masks = self.postprocess_masks(
					low_res_masks,
					input_size=(config_base['image_size'], config_base['image_size']),
					original_size=(config_base['image_size'], config_base['image_size'])
				)
				outputs = {
					'masks': masks,
					'iou_predictions': iou_predictions,
					'low_res_logits': low_res_masks
				}
				return outputs

			
		# return out, out1, out2, out3

# if __name__ == "__main__":
# 	with torch.no_grad():
# 		model = SAM2UNet().cuda()
# 		x = torch.randn(1, 3, 512, 512).cuda()
# 		# feature_maps = ResNet(50,False)
# 		out, out1, out2 = model(x)
# 		print(out.shape, out1.shape, out2.shape)

	def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    	) -> torch.Tensor:
			masks = F.interpolate(
				masks,
				(config_base['image_size'], config_base['image_size']),
				mode="bilinear",
				align_corners=False,
			)
			masks = masks[..., : input_size[0], : input_size[1]]
			masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
			return masks
