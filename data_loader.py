# data loader
from __future__ import print_function, division
import glob
import torch
from skimage import io, transform, color
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import cv2

ratio_return_unchanged = 0.1
ratio_do_transform = 0.02

USE_PRIOR = True

# ===================== generate prior channel for input image =====================
def data_motion_blur(image, mask):
    if random.random()<ratio_return_unchanged:
        return image, mask
    
    degree = random.randint(5, 30)
    angle = random.randint(0, 360)
    
    M = cv2.getRotationMatrix2D((degree/2, degree/2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel/degree
    
    img_blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    mask_blurred = cv2.filter2D(mask, -1, motion_blur_kernel)
    
    cv2.normalize(img_blurred, img_blurred, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(mask_blurred, mask_blurred, 0, 1, cv2.NORM_MINMAX)
    return img_blurred, mask_blurred
    
def data_motion_blur_prior(prior):
    if random.random()<ratio_return_unchanged:
        return prior
    
    degree = random.randint(5, 30)
    angle = random.randint(0, 360)
    
    M = cv2.getRotationMatrix2D((degree/2, degree/2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel/degree
    
    prior_blurred = cv2.filter2D(prior, -1, motion_blur_kernel)
    return prior_blurred  
    
def data_Affine(image, mask, height, width, bias, ratio=ratio_do_transform):
    if random.random()<ratio_return_unchanged:
        return image, mask
    # bias = np.random.randint(-int(height*ratio),int(width*ratio), 12)
    # bias = np.random.randint(0, int(width*ratio), 12)


    # pts1 = np.float32([[0+bias[0], 0+bias[1]], [width+bias[2], 0+bias[3]], [0+bias[4], height+bias[5]]])
    # pts2 = np.float32([[0+bias[6], 0+bias[7]], [width+bias[8], 0+bias[9]], [0+bias[10], height+bias[11]]])

    pts1 = np.float32([[0+bias[0], 0+bias[1]], [width+bias[2], 0+bias[3]], [0+bias[4], height+bias[5]]])
    pts2 = np.float32([[0+bias[6], 0+bias[7]], [width+bias[8], 0+bias[9]], [0+bias[10], height+bias[11]]])

    M = cv2.getAffineTransform(pts1, pts2)
    #    M = abs(M)

    img_affine = cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask_affine = cv2.warpAffine(mask, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # img_affine = img_affine / np.max(img_affine) 
    # mask_affine = mask_affine / np.max(mask_affine) 

    return img_affine, mask_affine

def data_Affine_prior(prior, height, width, bias, ratio=ratio_do_transform):
    if random.random()<ratio_return_unchanged:
        return prior
    # bias = np.random.randint(-int(height*ratio),int(width*ratio), 12)
    # bias = np.random.randint(0, int(width*ratio), 12)

    pts1 = np.float32([[0+bias[0], 0+bias[1]], [width+bias[2], 0+bias[3]], [0+bias[4], height+bias[5]]])
    pts2 = np.float32([[0+bias[6], 0+bias[7]], [width+bias[8], 0+bias[9]], [0+bias[10], height+bias[11]]])
    M = cv2.getAffineTransform(pts1, pts2)
    #    M = abs(M)

    prior_affine = cv2.warpAffine(prior, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # prior_affine = prior_affine / np.max(prior_affine) 

    return prior_affine
    
def data_Perspective(image, mask, height, width, bias, ratio=ratio_do_transform):
    if random.random()<ratio_return_unchanged:
        return image, mask
    # bias = np.random.randint(-int(height*ratio),int(width*ratio), 16)
    # bias = np.random.randint(0, int(width*ratio), 16)

    pts1 = np.float32([[0+bias[0],0+bias[1]], [height+bias[2],0+bias[3]], 
                       [0+bias[4],width+bias[5]], [height+bias[6], width+bias[7]]])
    pts2 = np.float32([[0+bias[8],0+bias[9]], [height+bias[10],0+bias[11]], 
                       [0+bias[12],width+bias[13]], [height+bias[14], width+bias[15]]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    #    M = abs(M)

    img_perspective = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask_perspective = cv2.warpPerspective(mask, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
	
    # img_perspective = img_perspective / np.max(img_perspective)
    # mask_perspective = mask_perspective / np.max(mask_perspective)

    return img_perspective, mask_perspective

def data_Perspective_prior(prior, height, width, bias, ratio=ratio_do_transform):
    if random.random()<ratio_return_unchanged:
        return prior
    # bias = np.random.randint(-int(height*ratio),int(width*ratio), 16)
    # bias = np.random.randint(0, int(width*ratio), 16)

    pts1 = np.float32([[0+bias[0],0+bias[1]], [height+bias[2],0+bias[3]], 
                       [0+bias[4],width+bias[5]], [height+bias[6], width+bias[7]]])
    pts2 = np.float32([[0+bias[8],0+bias[9]], [height+bias[10],0+bias[11]], 
                       [0+bias[12],width+bias[13]], [height+bias[14], width+bias[15]]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    #    M = abs(M)

    prior_perspective = cv2.warpPerspective(prior, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # prior_perspective = prior_perspective / np.max(prior_perspective)

    return prior_perspective

def data_ThinPlateSpline(image, mask, height, width, ratio=ratio_do_transform):
    if random.random()<ratio_return_unchanged:
        return image, mask
    bias = np.random.randint(-int(height*ratio),int(width*ratio), 16)
    # bias = np.random.randint(0, int(width*ratio), 16)

    tps = cv2.createThinPlateSplineShapeTransformer()
    sshape = np.array([[0+bias[0],0+bias[1]], [height+bias[2],0+bias[3]], 
                       [0+bias[4],width+bias[5]], [height+bias[6], width+bias[7]]], np.float32)
    tshape = np.array([[0+bias[8],0+bias[9]], [height+bias[10],0+bias[11]], 
                       [0+bias[12],width+bias[13]], [height+bias[14], width+bias[15]]], np.float32)
    sshape = sshape.reshape(1,-1,2)
    tshape = tshape.reshape(1,-1,2)
    matches = list()
    matches.append(cv2.DMatch(0,0,0))
    matches.append(cv2.DMatch(1,1,0))
    matches.append(cv2.DMatch(2,2,0))
    matches.append(cv2.DMatch(3,3,0))
    
    tps.estimateTransformation(tshape, sshape, matches)
    res = tps.warpImage(image)
    res_mask = tps.warpImage(mask)
    return res, res_mask   

def data_ThinPlateSpline_prior(prior, height, width, ratio=ratio_do_transform):
    if random.random()<ratio_return_unchanged:
        return prior
    bias = np.random.randint(-int(height*ratio),int(width*ratio), 16)
    # bias = np.random.randint(0, int(width*ratio), 16)

    tps = cv2.createThinPlateSplineShapeTransformer()
    sshape = np.array([[0+bias[0],0+bias[1]], [height+bias[2],0+bias[3]], 
                       [0+bias[4],width+bias[5]], [height+bias[6], width+bias[7]]], np.float32)
    tshape = np.array([[0+bias[8],0+bias[9]], [height+bias[10],0+bias[11]], 
                       [0+bias[12],width+bias[13]], [height+bias[14], width+bias[15]]], np.float32)
    sshape = sshape.reshape(1,-1,2)
    tshape = tshape.reshape(1,-1,2)
    matches = list()
    matches.append(cv2.DMatch(0,0,0))
    matches.append(cv2.DMatch(1,1,0))
    matches.append(cv2.DMatch(2,2,0))
    matches.append(cv2.DMatch(3,3,0))
    
    tps.estimateTransformation(tshape, sshape, matches)
    prior = tps.warpImage(prior)
    return prior

# class Augment_prior(object):

# 	def __init__(self, prior_prob, bias_affine, bias_perspective):
# 		# assert isinstance(output_size,(int,tuple))
# 		# self.output_size = output_size
# 		self.prior_prob = prior_prob
# 		self.bias_affine = bias_affine
# 		self.bias_perspective = bias_perspective

# 	def __call__(self,sample):
# 		imidx, image, label = sample['imidx'], sample['image'],sample['label']
# 		# print('max_label', np.max(label))
# 		h, w = image.shape[:2]

# 		prior = np.zeros((h, w, 1))

# 		if self.prior_prob >= random.random(): # add new augmentation
# 			prior[:,:,0:] = label[:,:,0:].copy() #!!!
# 			prior = np.array(prior, dtype=np.float)
# 			# print('max_prior_1', np.max(prior))

# 			bias_affine = np.random.randint(-int(h*ratio_do_transform),int(w*ratio_do_transform), 12)
# 			bias_perspective = np.random.randint(-int(h*ratio_do_transform),int(w*ratio_do_transform), 16)

# 			if random.random() >= 0.5:
# 				# modify image + mask, use groundtruth as prior
# 				image = np.array(image)
# 				prior = np.array(label[:,:,0:], dtype=np.float)
# 				# print('max_prior_2', np.max(prior))

# 				# image, prior = data_motion_blur(image, prior)
# 				# print('max_prior_3', np.max(prior))

# 				image, prior = data_Affine(image, prior, h, w, bias=self.bias_affine, ratio=ratio_do_transform)
# 				# print('max_prior_4', np.max(prior))

# 				# image, prior = data_Perspective(image, prior, h, w, bias=self.bias_perspective, ratio=ratio_do_transform)
# 				# print('max_prior_5', np.max(prior))

# 				# image, prior = data_ThinPlateSpline(image, prior, h, w, ratio=ratio_do_transform)
# 				# print('max_prior_6', np.max(prior))

# 				prior = prior.reshape(h, w, 1)
# 			else:
# 				# modify prior, don't change image + mask
# 				# prior = data_motion_blur_prior(prior)
# 				# print('max_prior_7', np.max(prior))

# 				prior = data_Affine_prior(prior, h, w, bias=self.bias_affine, ratio=ratio_do_transform)
# 				# print('max_prior_8', np.max(prior))

# 				# prior = data_Perspective_prior(prior, h, w, bias=self.bias_perspective, ratio=ratio_do_transform)
# 				# print('max_prior_9', np.max(prior))

# 				# prior = data_ThinPlateSpline_prior(prior, h, w, ratio=ratio_do_transform)
# 				# print('max_prior_10', np.max(prior))

# 				prior = prior.reshape(h, w, 1)

# 		image = np.concatenate((image, prior), axis=-1) # conctenate image and prior as forth channel
# 		# print('image.shape_Augment_prior', image.shape)
# 		# print('max_Augment_prior', np.max(image[:,:,3]))
		
# 		return {'imidx':imidx, 'image':image,'label':label}


class Augment_prior(object):

	def __init__(self, prior_prob):
		# assert isinstance(output_size,(int,tuple))
		# self.output_size = output_size
		self.prior_prob = prior_prob
		# self.bias_affine = bias_affine
		# self.bias_perspective = bias_perspective

	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'],sample['label']

		h, w = image.shape[:2]
		height, width = label.shape[:2]

		image_array = image.astype(np.uint8)
		image_array = cv2.resize(image_array, (height, width))

		label_array = label.astype(np.uint8)
		# label_array = cv2.resize(label_array, (h, w))
		# label_array = np.reshape(label_array, (h, w, 1))
		image = Image.fromarray(image.astype(np.uint8))
		prior_array = np.zeros((height, width, 1))

		if self.prior_prob >= random.random(): # add new augmentation
			
			prior = Image.fromarray((label[:,:,0]).astype(np.uint8))
			prior_array = label_array = label.astype(np.uint8)

			image_label_array = np.concatenate((image_array, label_array), axis=-1)
			image_label = Image.fromarray(image_label_array.astype(np.uint8))

			if random.random() >= 0.5:

				image_label = transforms.RandomAffine(degrees=1, translate=(0.03, 0.03), scale=None, shear=(0.03, 0.03, 0.03, 0.03), resample=False, fillcolor=0)(image_label)
				
				image_label_array = np.array(image_label)
				image_array = image_label_array[:,:,:3]
				label_array = image_label_array[:,:,3:]

			else:
				# modify prior, don't change image + mask
				prior = transforms.RandomAffine(degrees=1, translate=(0.03, 0.03), scale=None, shear=(0.03, 0.03, 0.03, 0.03), resample=False, fillcolor=0)(prior)
				prior_array = np.array(prior)
				prior_array = prior_array[:,:,np.newaxis]

		if USE_PRIOR:
			image_array = np.concatenate((image_array, prior_array), axis=-1)

		return {'imidx':imidx, 'image':image_array,'label':label_array}

#==========================dataset load==========================
# class RescaleT(object):

# 	def __init__(self,output_size):
# 		assert isinstance(output_size,(int,tuple))
# 		self.output_size = output_size

# 	def __call__(self,sample):
# 		imidx, image, label = sample['imidx'], sample['image'],sample['label']

# 		h, w = image.shape[:2] 

# 		if isinstance(self.output_size,int):
# 			if h > w:
# 				new_h, new_w = self.output_size*h/w,self.output_size
# 			else:
# 				new_h, new_w = self.output_size,self.output_size*w/h
# 		else:
# 			new_h, new_w = self.output_size

# 		new_h, new_w = int(new_h), int(new_w)

# 		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
# 		# img = transform.resize(image,(new_h,new_w),mode='constant')
# 		# lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

# 		image = transform.resize(image,(self.output_size,self.output_size),mode='constant')
# 		label = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)

# 		# print('image.shape_RescaleT', image.shape)
# 		# print('max_Rescale', np.max(image[:,:,3]))

# 		return {'imidx':imidx, 'image':image,'label':label}

# class RescaleT(object):

# 	def __init__(self,output_size):
# 		assert isinstance(output_size,(int,tuple))
# 		self.output_size = output_size

# 	def __call__(self,sample):
# 		imidx, image, label = sample['imidx'], sample['image'],sample['label']

# 		h, w = image.shape[:2]

# 		if isinstance(self.output_size,int):
# 			if h > w:
# 				new_h, new_w = self.output_size*h/w,self.output_size
# 			else:
# 				new_h, new_w = self.output_size,self.output_size*w/h
# 		else:
# 			new_h, new_w = self.output_size

# 		new_h, new_w = int(new_h), int(new_w)

# 		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
# 		# img = transform.resize(image,(new_h,new_w),mode='constant')
# 		# lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

# 		image = transform.resize(image,(self.output_size,self.output_size),mode='constant')
# 		label = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)

# 		# print('image.shape_RescaleT', image.shape)
# 		# print('max_Rescale', np.max(image[:,:,3]))

# 		return {'imidx':imidx, 'image':image,'label':label}

class ColorJitter(object):

	def __init__(self,brightness,contrast,saturation,hue):
		# assert isinstance(output_size,(int,tuple))
		# self.output_size = output_size
		self.brightness = brightness
		self.contrast = contrast
		self.saturation = saturation
		self.hue = hue

	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'],sample['label']

		# h, w = image.shape[:2]

		prior = image[:,:,3:]
		image = image[:,:,:3]
		image = Image.fromarray(np.uint8(image))
		# print('max_color_jitter', np.max(image))
		# image = Image.fromarray(np.uint8(image))
		image = transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation, hue=self.hue)(image)
		image = np.array(image) 

		if USE_PRIOR:
			image = np.concatenate((image, prior), axis=-1)

		# print('image.shape_ColorJitter', image.shape)
		# print('max_ColorJitter', np.max(image[:,:,3]))

		return {'imidx':imidx, 'image':image,'label':label}

class RescaleT(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'],sample['label']

		prior = image[:,:,3:]
		image = image[:,:,:3]

		# print('image_shape', image.shape)
		# print('prior_shape', prior.shape)
		# print('label_shape', label.shape)

		image = Image.fromarray(np.uint8(image))
		if USE_PRIOR:
			prior = Image.fromarray(np.uint8(prior[:,:,0]))
		label = Image.fromarray(np.uint8(label[:,:,0]))
		# label = Image.fromarray((label[:,:,0]).astype(np.uint8))


		image = transforms.Resize(self.output_size, interpolation=2)(image)
		if USE_PRIOR:
			prior = transforms.Resize(self.output_size, interpolation=2)(prior)
		label = transforms.Resize(self.output_size, interpolation=2)(label)

		# image = torch.cat((image, prior), axis=-1)
		image = np.array(image)
		if USE_PRIOR:
			prior = np.array(prior)
			prior = prior[:,:,np.newaxis]
		label = np.array(label)
		label = label[:,:,np.newaxis]

		if USE_PRIOR:
			image = np.concatenate((image, prior), axis=-1)

		return {'imidx':imidx, 'image':image,'label':label}

class RandomCrop(object):

	def __init__(self,output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size
	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'], sample['label']

		prior = image[:,:,3:]
		image = image[:,:,:3]

		image = Image.fromarray(np.uint8(image))
		prior = Image.fromarray(np.uint8(prior[:,:,0]))
		label = Image.fromarray(np.uint8(label[:,:,0]))

		# print('max_color_jitter', np.max(image))
		image = transforms.RandomCrop(self.output_size)(image)
		prior = transforms.RandomCrop(self.output_size)(prior)
		label = transforms.RandomCrop(self.output_size)(label)

		image = np.array(image)
		prior = np.array(prior)
		prior = prior[:,:,np.newaxis]
		label = np.array(label)
		label = label[:,:,np.newaxis]

		# image = np.array(image) 
		image = np.concatenate((image, prior), axis=-1)
		
		# image = torch.cat((image, prior), axis=-1)

		# if random.random() >= 0.5:
		# 	image = image[::-1]
		# 	label = label[::-1]

		# h, w = image.shape[:2]
		# new_h, new_w = self.output_size

		# top = np.random.randint(0, h - new_h)
		# left = np.random.randint(0, w - new_w)

		# image = image[top: top + new_h, left: left + new_w]
		# label = label[top: top + new_h, left: left + new_w]

		# print('shape_RandomCrop', image.shape)
		# print('max_RandomCrop', np.max(image[:,:,3]))

		return {'imidx':imidx,'image':image, 'label':label}

class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):

		imidx, image, label = sample['imidx'], sample['image'], sample['label']

		tmpImg = np.zeros((image.shape[0],image.shape[1],3))
		tmpLbl = np.zeros(label.shape)

		image = image/np.max(image)
		if(np.max(label)<1e-6):
			label = label
		else:
			label = label/np.max(label)

		if image.shape[2]==1:
			tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
		else:
			tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
			tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

		tmpLbl[:,:,0] = label[:,:,0]

		# change the r,g,b to b,r,g from [0,255] to [0,1]
		#transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
		tmpImg = tmpImg.transpose((2, 0, 1))
		tmpLbl = label.transpose((2, 0, 1))

		return {'imidx':torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}

# class ToTensorLab(object):
# 	"""Convert ndarrays in sample to Tensors."""
# 	def __init__(self,flag=0):
# 		self.flag = flag

# 	def __call__(self, sample):

# 		imidx, image, label =sample['imidx'], sample['image'], sample['label']

# 		# print('image_ToTensorLab', image.shape)

# 		tmpLbl = np.zeros(label.shape)

# 		if(np.max(label)<1e-6):
# 			label = label
# 		else:
# 			label = label/np.max(label)

# 		# change the color space
# 		if self.flag == 2: # with rgb and Lab colors
# 			tmpImg = np.zeros((image.shape[0],image.shape[1],6))
# 			tmpImgt = np.zeros((image.shape[0],image.shape[1],3))
# 			if image.shape[2]==1:
# 				tmpImgt[:,:,0] = image[:,:,0]
# 				tmpImgt[:,:,1] = image[:,:,0]
# 				tmpImgt[:,:,2] = image[:,:,0]
# 			else:
# 				tmpImgt = image
# 			tmpImgtl = color.rgb2lab(tmpImgt)

# 			# nomalize image to range [0,1]
# 			tmpImg[:,:,0] = (tmpImgt[:,:,0]-np.min(tmpImgt[:,:,0]))/(np.max(tmpImgt[:,:,0])-np.min(tmpImgt[:,:,0]))
# 			tmpImg[:,:,1] = (tmpImgt[:,:,1]-np.min(tmpImgt[:,:,1]))/(np.max(tmpImgt[:,:,1])-np.min(tmpImgt[:,:,1]))
# 			tmpImg[:,:,2] = (tmpImgt[:,:,2]-np.min(tmpImgt[:,:,2]))/(np.max(tmpImgt[:,:,2])-np.min(tmpImgt[:,:,2]))
# 			tmpImg[:,:,3] = (tmpImgtl[:,:,0]-np.min(tmpImgtl[:,:,0]))/(np.max(tmpImgtl[:,:,0])-np.min(tmpImgtl[:,:,0]))
# 			tmpImg[:,:,4] = (tmpImgtl[:,:,1]-np.min(tmpImgtl[:,:,1]))/(np.max(tmpImgtl[:,:,1])-np.min(tmpImgtl[:,:,1]))
# 			tmpImg[:,:,5] = (tmpImgtl[:,:,2]-np.min(tmpImgtl[:,:,2]))/(np.max(tmpImgtl[:,:,2])-np.min(tmpImgtl[:,:,2]))

# 			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

# 			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
# 			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
# 			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
# 			tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3]))/np.std(tmpImg[:,:,3])
# 			tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4]))/np.std(tmpImg[:,:,4])
# 			tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5]))/np.std(tmpImg[:,:,5])

# 		elif self.flag == 1: #with Lab color
# 			tmpImg = np.zeros((image.shape[0],image.shape[1],3))

# 			if image.shape[2]==1:
# 				tmpImg[:,:,0] = image[:,:,0]
# 				tmpImg[:,:,1] = image[:,:,0]
# 				tmpImg[:,:,2] = image[:,:,0]
# 			else:
# 				tmpImg = image

# 			tmpImg = color.rgb2lab(tmpImg)

# 			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

# 			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.min(tmpImg[:,:,0]))/(np.max(tmpImg[:,:,0])-np.min(tmpImg[:,:,0]))
# 			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.min(tmpImg[:,:,1]))/(np.max(tmpImg[:,:,1])-np.min(tmpImg[:,:,1]))
# 			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.min(tmpImg[:,:,2]))/(np.max(tmpImg[:,:,2])-np.min(tmpImg[:,:,2]))

# 			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
# 			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
# 			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])

# 		else: # with rgb color
# 			tmpImg = np.zeros((image.shape[0],image.shape[1],4))
# 			image = image/np.max(image)
# 			# if image.shape[2]==1:
# 			# 	tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
# 			# 	tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
# 			# 	tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
# 			# 	tmpImg[:,:,3] = image[:,:,1]
# 			# else:
# 			# 	tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
# 			# 	tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
# 			# 	tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225
# 			# 	tmpImg[:,:,3] = image[:,:,3]

# 			if image.shape[2]==1:
# 				tmpImg[:,:,0] = image[:,:,0]
# 				tmpImg[:,:,1] = image[:,:,0]
# 				tmpImg[:,:,2] = image[:,:,0]
# 				tmpImg[:,:,3] = image[:,:,1]
# 			else:
# 				tmpImg[:,:,0] = image[:,:,0]
# 				tmpImg[:,:,1] = image[:,:,1]
# 				tmpImg[:,:,2] = image[:,:,2]
# 				tmpImg[:,:,3] = image[:,:,3]
				
# 			# print('image_max', np.max(image[:,:,3]))
# 			# print('tmpImg_max', np.max(tmpImg[:,:,3]))

# 		tmpLbl[:,:,0] = label[:,:,0]

# 		# change the r,g,b to b,r,g from [0,255] to [0,1]
# 		#transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
# 		tmpImg = tmpImg.transpose((2, 0, 1))
# 		tmpLbl = label.transpose((2, 0, 1))

# 		# print('tmpImg_shape', tmpImg.shape)
# 		# print('tmpLbl_shape', tmpLbl.shape)

# 		return {'imidx':torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}

class ToTensorLab(object):
	"""Convert ndarrays in sample to Tensors."""
	def __init__(self,flag=0):
		self.flag = flag

	def __call__(self, sample):

		imidx, image, label = sample['imidx'], sample['image'], sample['label']

		# print('image_ToTensorLab', image.shape)

		tmpLbl = np.zeros(label.shape)

		if(np.max(label)<1e-6):
			label = label
		else:
			label = label/np.max(label)

		if USE_PRIOR:
			tmpImg = np.zeros((image.shape[0],image.shape[1],4))
		else:
			tmpImg = np.zeros((image.shape[0],image.shape[1],3))

		image = image/np.max(image)
		# label = image/np.max(label)

		if image.shape[2]==1:
			tmpImg[:,:,0] = image[:,:,0]
			tmpImg[:,:,1] = image[:,:,0]
			tmpImg[:,:,2] = image[:,:,0]
			if USE_PRIOR:
				tmpImg[:,:,3] = image[:,:,1]
		else:
			tmpImg[:,:,0] = image[:,:,0]
			tmpImg[:,:,1] = image[:,:,1]
			tmpImg[:,:,2] = image[:,:,2]
			if USE_PRIOR:
				tmpImg[:,:,3] = image[:,:,3]
			
		# print('image_max', np.max(image[:,:,3]))
		# print('tmpImg_max', np.max(tmpImg[:,:,3]))

		tmpLbl[:,:,0] = label[:,:,0]

		tmpImg = np.moveaxis(tmpImg, 2, 0)
		tmpLbl = np.moveaxis(tmpLbl, 2, 0)

		# change the r,g,b to b,r,g from [0,255] to [0,1]
		# tmpImg = tmpImg.transpose((2, 0, 1))
		# tmpLbl = label.transpose((2, 0, 1))

		# print('tmpImg_shape', tmpImg.shape)
		# print('tmpLbl_shape', tmpLbl.shape)

		return {'imidx':torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}


class SalObjDataset(Dataset):
	def __init__(self,img_name_list,lbl_name_list,transform=None):
		# self.root_dir = root_dir
		# self.image_name_list = glob.glob(image_dir+'*.png')
		# self.label_name_list = glob.glob(label_dir+'*.png')
		self.image_name_list = img_name_list
		self.label_name_list = lbl_name_list
		self.transform = transform

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self,idx):

		# image = Image.open(self.image_name_list[idx])#io.imread(self.image_name_list[idx])
		# label = Image.open(self.label_name_list[idx])#io.imread(self.label_name_list[idx])

		image = io.imread(self.image_name_list[idx])
		imname = self.image_name_list[idx]
		imidx = np.array([idx])

		if(0==len(self.label_name_list)):
			label_3 = np.zeros(image.shape)
		else:
			label_3 = io.imread(self.label_name_list[idx])

		label = np.zeros(label_3.shape[0:2])
		if(3==len(label_3.shape)):
			label = label_3[:,:,0]
		elif(2==len(label_3.shape)):
			label = label_3

		if(3==len(image.shape) and 2==len(label.shape)):
			label = label[:,:,np.newaxis]
		elif(2==len(image.shape) and 2==len(label.shape)):
			image = image[:,:,np.newaxis]
			label = label[:,:,np.newaxis]

		sample = {'imidx':imidx, 'image':image, 'label':label}

		if self.transform:
			sample = self.transform(sample)

		return sample
