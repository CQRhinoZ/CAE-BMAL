import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from glob import glob
import datetime
import custom_transforms as tr
import random
import argparse
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import custom_transforms as tr
from PIL import Image

import tqdm

if not os.path.exists('./dc_img/Domain4'):
    os.mkdir('./dc_img/Domain4')
if not os.path.exists('./wae_img/Domain4'):
    os.mkdir('./wae_img/Domain4')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
        '--datasetS', type=str, default='Domain4', help='test folder id contain images ROIs to test'
    )
parser.add_argument(
    '--data-dir',
    default='/ai/sjq/dataset/Fundus',
    help='data root path'
)
args = parser.parse_args()

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 796, 796)
    return x


num_epochs = 100           # Domain 1,2: 200  Domain 3,4: 100
batch_size = 40            # Domain 1,2: 128  Domain 3,4: 40
learning_rate = 1e-3

img_transform = transforms.Compose([
    #tr.ResizeImg(796),
    #tr.RandomScaleCrop(512),#512
    tr.CenterCrop(796),
    # tr.CenterCrop(800),
    # tr.RandomRotate(),
    # tr.RandomFlip(),
    # tr.elastic_transform(),
    # tr.add_salt_pepper_noise(),
    # tr.adjust_light(),
    # tr.eraser(),
    tr.Normalize_tf(),
    tr.ToTensor()
])

class FundusSegmentation(Dataset):
    """
    Fundus segmentation dataset
    including 5 domain dataset
    one for test others for training
    """
    def __init__(self,
                 base_dir=Path.db_root_dir('fundus'),
                 dataset='refuge',
                 split='train/ROIs',
                 testid=None,
                 transform=None
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        # super().__init__()
        self._base_dir = base_dir
        self.image_list = []
        self.split = split

        self.image_pool = []
        self.label_pool = []
        self.img_name_pool = []
        SEED = 1212
        random.seed(SEED)

        self._image_dir = os.path.join(self._base_dir, dataset, split, 'image')
        print(self._image_dir)
        imagelist = glob(self._image_dir + "/*.png")
        for image_path in imagelist:
            gt_path = image_path.replace('image', 'mask')
            self.image_list.append({'image': image_path, 'label': gt_path, 'id': testid})

        self.transform = transform
        self._read_img_into_memory()
        #self._read_img_list_into_memory()
        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.image_list)))


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        _img = self.image_pool[index]
        #_target = self.label_pool[index]
        _img_name = self.img_name_pool[index]
        #anco_sample = {'image': _img, 'label': _target, 'img_name': _img_name}
        anco_sample = {'image': _img, 'img_name': _img_name}
        if self.transform is not None:
            anco_sample = self.transform(anco_sample)

        return anco_sample

    def _read_img_into_memory(self):
        img_num = len(self.image_list)
        for index in range(img_num):
            self.image_pool.append(Image.open(self.image_list[index]['image']).convert('RGB'))
            #_target = Image.open(self.image_list[index]['label'])
            # if _target.mode is 'RGB':
            #     _target = _target.convert('L')
            # self.label_pool.append(_target)
            _img_name = self.image_list[index]['image'].split('/')[-1]
            self.img_name_pool.append(_img_name)
        return self.img_name_pool



    def __str__(self):
        return 'Fundus(split=' + str(self.split) + ')'

domain = FundusSegmentation(base_dir=args.data_dir, dataset=args.datasetS, split='train/ROIs',
                                                         transform=img_transform)
domain_loaderS = DataLoader(domain, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
# Domain1 Domain2 num_workers=2, pin_memory=True
# Domain3 num_workers=0, pin_memory=False


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 320, 3, stride=3, padding=1),  # b, 16, 10, 10  # 50 99 320 320
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(320, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 320, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(320, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
starttime = datetime.datetime.now()

for epoch in range(num_epochs):
    batch_flagv = -1
    output_array = []
    output_list = []
    for batch_idx, sampleS in tqdm.tqdm(
            enumerate(domain_loaderS), total=len(domain_loaderS),
            desc='Train epoch=%d' % num_epochs, ncols=80, leave=False):
        img = sampleS['image'].cuda()
        # img = imageS
        # img = img.view(img.size(0), -1)
        # img = Variable(img)
        img_name = sampleS['img_name']
        img = (img.cuda() if torch.cuda.is_available() else img)
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_flagv != batch_idx:
            output_array.append(output)
            output_list.append(img_name)
            batch_flagv = batch_idx

    #print(len(output_array))
    # ===================log========================
    endtime = datetime.datetime.now()
    print('epoch [{}/{}], loss:{:.4f}, time:{:.2f}s'.format(epoch + 1, num_epochs, loss.item(),
                                                            (endtime - starttime).seconds))

    if (epoch + 1) % 50 == 0:#or epoch == 0:
        for i in range(len(output_array)):
            pic = to_img(output_array[i].cpu().data)
            save_image(pic, './dc_img/Domain3/image_{}_{}.png'.format(epoch + 1, i))
            if epoch + 1 == num_epochs:
                name_list = output_list[i]
                pics = pic.permute(0, 2, 3, 1)
                k = pics.cpu().detach().numpy()
                for i in range(pic.size(0)):
                    res = k[i]  # Get the picture of one step in the batch
                    res = res * 255
                    image = Image.fromarray(np.uint8(res)).convert('RGB')
                    # Store results by time naming
                    #timestamp = datetime.datetime.now().strftime("%M-%S")
                    savepath = './wae_img/Domain4/' + 're_' + name_list[i]
                    image.save(savepath)
torch.save(model.state_dict(), './logs/conv_autoencoder_Domain4.pth')
