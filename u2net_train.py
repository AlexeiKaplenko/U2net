import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
import os
import cv2

# from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from data_loader import Augment_prior
from data_loader import ColorJitter

from model import U2NET
from model import U2NETP

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

	loss0 = bce_loss(d0,labels_v)
	loss1 = bce_loss(d1,labels_v)
	loss2 = bce_loss(d2,labels_v)
	loss3 = bce_loss(d3,labels_v)
	loss4 = bce_loss(d4,labels_v)
	loss5 = bce_loss(d5,labels_v)
	loss6 = bce_loss(d6,labels_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	# print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data[0],loss1.data[0],loss2.data[0],loss3.data[0],loss4.data[0],loss5.data[0],loss6.data[0]))
	print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data,loss1.data,loss2.data,loss3.data,loss4.data,loss5.data,loss6.data))


	return loss0, loss


# ------- 2. set the directory of training dataset --------

model_name = 'u2netp' #'u2net' #'u2netp'
# model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth') # pretrained model dir

data_dir = os.path.join(os.getcwd(), 'train_data' + os.sep)
# tra_image_dir = os.path.join('DUTS', 'DUTS-TR', 'DUTS-TR', 'im_aug' + os.sep)
# tra_label_dir = os.path.join('DUTS', 'DUTS-TR', 'DUTS-TR', 'gt_aug' + os.sep)

tra_image_dir = os.path.join('FINAL3_combined' + os.sep)
# tra_label_dir = os.path.join('FINAL3_MATTE_predicted' + os.sep)
tra_label_dir = os.path.join('FINAL3_MATTE_predicted' + os.sep)


# tra_image_dir = os.path.join('test1_image' + os.sep)
# tra_label_dir = os.path.join('test1_matte' + os.sep)

print("tra_image_dir", tra_image_dir)
print("tra_label_dir", tra_label_dir)

# image_ext = '.jpg'
image_ext = '.png'
label_ext = '.png'

# model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)
# model_dir = os.path.join('/home/xkaple00/JUPYTER_SHARED/Digis/Background_removal/U-2-Net/saved_models/u2net/u2net.pth')
# model_dir = "./saved_models/u2netp/202000.pth"
model_dir = "./saved_models/u2netp/itr_8000_train_0.273093_tar_0.022038.pth"

# model_dir = os.path.join('/home/xkaple00/JUPYTER_SHARED/Digis/Background_removal/U-2-Net/saved_models/u2netp/u2netp.pth')

# model_dir = os.path.join('/home/xkaple00/JUPYTER_SHARED/Digis/Background_removal/U-2-Net/saved_models/u2net/u2net_bce_itr_2000_train_2.724735_tar_0.386822.pth')
# model_dir = os.path.join('/home/xkaple00/JUPYTER_SHARED/Digis/Background_removal/U-2-Net/saved_models/u2netp/11400.pth')


epoch_num = 100000
batch_size_train = 16
batch_size_val = 1
train_num = 0
val_num = 0

print('datadir', data_dir + tra_image_dir + '*' + image_ext)
# print('labeldir', data_dir + tra_label_dir + imidx + label_ext)

tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)

tra_lbl_name_list = []
for img_path in tra_img_name_list:
	img_name = img_path.split(os.sep)[-1]

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        Augment_prior(0.5),
        RescaleT((320,320)),
        # RandomCrop(288), #cause of misalignment of label and input
        ColorJitter(brightness=(0.9,1.1),contrast=(0.9,1.1),saturation=(0.9,1.1),hue=0.1),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=32)

# ------- 3. define model --------
# define the net
if(model_name=='u2net'):
    net = U2NET(4, 1)
    net.load_state_dict(torch.load(model_dir), strict = False)
elif(model_name=='u2netp'):
    net = U2NETP(4,1)
    net.load_state_dict(torch.load(model_dir), strict = False)

if torch.cuda.is_available():
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0) # lr = 0.001

# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
save_freq = 2000 # save the model every 2000 iterations

for epoch in range(0, epoch_num):
    net.train()

    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels = data['image'], data['label']

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # print('inputs_size', inputs.size())
        # print('labels_size', labels.size())

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

        loss.backward()
        optimizer.step()

        # # print statistics
        # running_loss += loss.data[0]
        # running_tar_loss += loss2.data[0]
        
        running_loss += loss.data
        running_tar_loss += loss2.data
        

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
        epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

        if ite_num % save_freq == 0:

            torch.save(net.state_dict(), "saved_models/u2netp/itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # resume train

            # middle_output = (d0[0][0] + d1[0][0] + d2[0][0] + d3[0][0] + d4[0][0] + d5[0][0] + d6[0][0]) / 7 * 255
            
            middle_output = d0[0][0] * 255

            middle_output = middle_output.cpu().detach().numpy()
            middle_input = inputs.cpu().detach().numpy()[0][:3]
            middle_input = np.moveaxis(middle_input, 0, 2) * 255
            middle_prior = inputs.cpu().detach().numpy()[0][3] * 255

            middle_label = labels.cpu().detach().numpy()[0][0] * 255

            # print('middle_input', middle_input.shape)
            # print('middle_prior', middle_prior.shape)


            # cv2.imwrite(model_dir + "saved_models/u2netp/_output_bce_itr_%d_train_%3f_tar_%3f.png" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val), middle_output)
            # cv2.imwrite(model_dir + "_input_bce_itr_%d_train_%3f_tar_%3f.png" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val), cv2.cvtColor(middle_input, cv2.COLOR_BGR2RGB))
            # cv2.imwrite(model_dir + "_prior_bce_itr_%d_train_%3f_tar_%3f.png" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val), middle_prior)
            # cv2.imwrite(model_dir + "_difference_prior_output_bce_itr_%d_train_%3f_tar_%3f.png" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val), middle_prior - middle_output)
            # cv2.imwrite(model_dir + "_difference_prior_label_bce_itr_%d_train_%3f_tar_%3f.png" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val), middle_prior - middle_label)
            # cv2.imwrite(model_dir + "_difference_output_label_bce_itr_%d_train_%3f_tar_%3f.png" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val), middle_output - middle_label)

            cv2.imwrite("saved_models/u2netp/output_itr_%d.png" % (ite_num), middle_output)
            cv2.imwrite("saved_models/u2netp/input_itr_%d.png" % (ite_num), cv2.cvtColor(middle_input, cv2.COLOR_BGR2RGB))
            cv2.imwrite("saved_models/u2netp/prior_itr_%d.png" % (ite_num), middle_prior)
            cv2.imwrite("saved_models/u2netp/difference_prior_output_itr_%d.png" % (ite_num), middle_prior - middle_output)
            cv2.imwrite("saved_models/u2netp/difference_prior_label_itr_%d.png" % (ite_num), middle_prior - middle_label)
            cv2.imwrite("saved_models/u2netp/difference_output_label_itr_%d.png" % (ite_num), middle_output - middle_label)

            ite_num4val = 0

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss

if __name__ == "__main__":
    main()
