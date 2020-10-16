import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob
import cv2

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from data_loader import Augment_prior
from data_loader import ColorJitter



from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def main():
    # filename_list = glob.glob('/home/xkaple00/JUPYTER_SHARED/Digis/Background_removal/U-2-Net/train_data/FINAL3_combined')
    # --------- 1. get image path and name ---------
    model_name='u2net'#u2netp
    # model_dir = os.path.join('/home/xkaple00/JUPYTER_SHARED/Digis/Background_removal/U-2-Net/saved_models/u2netp/166000_train_0.3457.pth')    model_dir = os.path.join('/home/xkaple00/JUPYTER_SHARED/Digis/Background_removal/U-2-Net/saved_models/u2netp/166000_train_0.3457.pth')
    # model_dir = os.path.join('/home/xkaple00/JUPYTER_SHARED/Digis/Background_removal/U-2-Net/saved_models/u2netp/prior_0.5_affine_0.05_loss_0.4585.pth')

    model_dir = os.path.join('/home/xkaple00/JUPYTER_SHARED/Digis/Background_removal/U-2-Net/saved_models/u2netp/best4_large.pth')
    # model_dir = os.path.join('/home/xkaple00/JUPYTER_SHARED/Digis/Background_removal/U-2-Net/saved_models/u2netp/best3_large.pth')

    # model_dir = '/home/xkaple00/JUPYTER_SHARED/Digis/Background_removal/U-2-Net/saved_models/u2netp/112000.pth'

    # model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')
    # model_dir = os.path.join('/home/xkaple00/JUPYTER_SHARED/Digis/Background_removal/U-2-Net/saved_models/u2netp/u2netp.pth')

    # image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images')
    # image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images_mine', 'IMG_9793')

    # image_dir = '/home/xkaple00/JUPYTER_SHARED/Digis/Background_removal/U-2-Net/train_data/FINAL3_combined_resized_320'
    # label_dir = '/home/xkaple00/JUPYTER_SHARED/Digis/Background_removal/U-2-Net/train_data/FINAL3_MATTE_predicted'
    # prior_dir = '/home/xkaple00/JUPYTER_SHARED/Digis/Background_removal/U-2-Net/train_data/FINAL3_MATTE_predicted_2'

    # image_dir = '/home/xkaple00/JUPYTER_SHARED/Digis/Background_removal/dataset/Digis1/Extraction/0.1Ct small/spin/0.1ct small__2020-06-07-15-39-06'
    # image_dir = '/home/xkaple00/JUPYTER_SHARED/Digis/Background_removal/dataset/Digis1/Extraction/0.1ct/spins/0.1ct__2020-04-17-17-15-47'
    
    # image_dir = '/home/xkaple00/JUPYTER_SHARED/Digis/Background_removal/dataset/Digis1/Extraction/images'
    # prior_dir = '/home/xkaple00/JUPYTER_SHARED/Digis/Background_removal/U-2-Net/test_data/MATTE4_predicted'
    

    # image_dir = '/home/xkaple00/JUPYTER_SHARED/Digis/Background_removal/U-2-Net/train_data/FINAL5_combined'
    # label_dir = '/home/xkaple00/JUPYTER_SHARED/Digis/Background_removal/U-2-Net/train_data/FINAL5_MATTE'
    # prior_dir = '/home/xkaple00/JUPYTER_SHARED/Digis/Background_removal/U-2-Net/train_data/FINAL5_MATTE'

    # image_dir = '/home/xkaple00/JUPYTER_SHARED/Digis/Background_removal/U-2-Net/test_data/Digis1_images1_resized'

    # image_dir = '/home/xkaple00/JUPYTER_SHARED/Digis/Background_removal/U-2-Net/test_data/FINAL5/input'
    # prior_dir = '/home/xkaple00/JUPYTER_SHARED/Digis/Background_removal/U-2-Net/test_data/FINAL5/prior'

    # image_dir = '/home/xkaple00/JUPYTER_SHARED/Digis/Background_removal/Motion-Guided-Attention/dataset/DAVIS/JPEGImages/480p/ring'

    # image_dir = '/home/xkaple00/JUPYTER_SHARED/Digis/Background_removal/pytorch-pwc/INPUT/test3'
    # prior_dir = '/home/xkaple00/JUPYTER_SHARED/Digis/Background_removal/U-2-Net/test_data/FINAL5' 

    image_dir = '/home/xkaple00/JUPYTER_SHARED/Digis/Background_removal/dataset/Digis1/Extraction/images'


    # image_dir = '/home/xkaple00/JUPYTER_SHARED/Digis/Background_removal/dataset/video2frames/test1'
    # prior_dir = '/home/xkaple00/JUPYTER_SHARED/Digis/Background_removal/U-2-Net/test_data/FINAL5/prior_video_pred'

    # image_dir = '/home/xkaple00/JUPYTER_SHARED/Digis/Background_removal/U-2-Net/test_data/FINAL5_input1'
    # prior_dir = '/home/xkaple00/JUPYTER_SHARED/Digis/Background_removal/U-2-Net/test_data/FINAL5_pred1'

    # prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep)
    output_dir = os.path.join(os.getcwd(), 'test_data', 'FINAL5' + os.sep)
    
    # output_dir = 'test_data' + os.sep + 'FINAL3_MATTE_predicted_2' + os.sep
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    img_name_list = sorted(glob.glob(image_dir + os.sep + '*'))

    # lbl_name_list = sorted(glob.glob(label_dir + os.sep + '*'))
    lbl_name_list = [] 

    # pri_name_list = sorted(glob.glob(prior_dir + os.sep + '*'))
    pri_name_list = []
    # print('len(lbl_name_list)', len(lbl_name_list))

    # print(img_name_list)

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = lbl_name_list,
                                        pri_name_list = pri_name_list,
                                        # transform=transforms.Compose([
                                        #                             Augment_prior(0.5),
                                        #                             RescaleT((320, 320)),
                                        #                             ToTensorLab(flag=0)])
                                        # )

                                        transform=transforms.Compose([
                                            Augment_prior(1.),
                                            RescaleT((320,320)),
                                            # RandomCrop(288), #cause of misalignment of label and input
                                            # ColorJitter(brightness=(0.9,1.1),contrast=(0.9,1.1),saturation=(0.9,1.1),hue=0.1),
                                            ToTensorLab(flag=0)]))

    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(4,1)
        net.load_state_dict(torch.load(model_dir))

    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(4,1)
        net.load_state_dict(torch.load(model_dir))

    if torch.cuda.is_available():
        net.cuda()
    
    net.eval()

    # net = net.load_state_dict(torch.load(model_dir))

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:", img_name_list[i_test].split(os.sep)[-1])

        print('img_name_list', img_name_list[i_test])

        hires_image = cv2.imread(img_name_list[i_test])
        hires_image = cv2.cvtColor(hires_image, cv2.COLOR_BGR2RGB)

        height, width = hires_image.shape[:2]

        inputs_test = data_test['image']
        label = data_test['label']
        label = label.cpu().detach().numpy()[0][0] * 255
        print('label', label.shape)

        print('inputs_test.shape', inputs_test.shape)
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d0,d1,d2,d3,d4,d5,d6 = net(inputs_test)

        # normalization
        pred = d0[0,0,:,:] * 255
        pred = pred.cpu().detach().numpy()
        # pred = cv2.resize(pred, (height, width), cv2.INTER_LINEAR)
        pred = cv2.resize(pred, (width, height), cv2.INTER_LINEAR)

        print('pred_shape', pred.shape)

        inputs = inputs_test.cpu().detach().numpy()[0]
        input_image = inputs[:3] * 255

        if inputs.shape[0] == 4:
            prior = inputs[3] * 255

        input_image = np.moveaxis(input_image, 0, 2)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)


        final_image = cv2.cvtColor(hires_image, cv2.COLOR_RGB2RGBA)
        print('final_image_shape', final_image.shape)
        final_image[:,:,3] = pred

        image_pil = Image.fromarray(np.uint8(final_image))
        # image_pil.save(os.path.join(output_dir, img_name_list[i_test].split(os.sep)[-1])+'.png')

        
        # #Save Input and final output
        image_pil.save(os.path.join(output_dir, str(i_test)+'_final_output.png'))
        hires_image = cv2.cvtColor(hires_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(output_dir, str(i_test)+'_inputs.png'), hires_image)

        # #save CV2 image
        # cv2.imwrite(os.path.join(output_dir, '{:03d}.png'.format(i_test+1)), pred)
        # cv2.imwrite(os.path.join(output_dir, str(i_test)+'_pred.png'), pred)
        # cv2.imwrite(os.path.join(output_dir, str(i_test)+'_inputs.png'), input_image)

        # cv2.imwrite(os.path.join(output_dir, str(i_test)+'_prior.png'), prior) 

        # # Prepare data for large network
        # if not os.path.exists(output_dir + '/' + 'prior'):
        #      os.makedirs(output_dir + '/' + 'prior', exist_ok=True)
        # if not os.path.exists(output_dir + '/' + 'input'):
        #      os.makedirs(output_dir + '/' + 'input', exist_ok=True)
        # if not os.path.exists(output_dir + '/' + 'label'):
        #      os.makedirs(output_dir + '/' + 'label', exist_ok=True)

        # cv2.imwrite(os.path.join(output_dir + '/' + 'prior', str(i_test)+'.png'), pred)
        # cv2.imwrite(os.path.join(output_dir + '/' + 'input', str(i_test)+'.png'), input_image)
        # cv2.imwrite(os.path.join(output_dir + '/' + 'label', str(i_test)+'.png'), label)

        # # Prepare data for large network video
        # if not os.path.exists(output_dir + '/' + 'prior_video'):
        #      os.makedirs(output_dir + '/' + 'prior_video', exist_ok=True)
        # if not os.path.exists(output_dir + '/' + 'input_video'):
        #      os.makedirs(output_dir + '/' + 'input_video', exist_ok=True)
        # if not os.path.exists(output_dir + '/' + 'label_video'):
        #      os.makedirs(output_dir + '/' + 'label_video', exist_ok=True)

        # if i_test == 0:
        #     cv2.imwrite(os.path.join(output_dir + '/' + 'prior_video', str(i_test)+'.png'), np.zeros(pred.shape))

        # cv2.imwrite(os.path.join(output_dir + '/' + 'prior_video', str(i_test+1)+'.png'), pred)
        # cv2.imwrite(os.path.join(output_dir + '/' + 'input_video', str(i_test)+'.png'), input_image)
        # # cv2.imwrite(os.path.join(output_dir + '/' + 'label_video', str(i_test)+'.png'), label)

        del d0,d1,d2,d3,d4,d5,d6

if __name__ == "__main__":
    main()
