# -*- coding: utf-8 -*-

import collections
import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
print(torch.version.cuda) #10.1
import time

from tensorflow.keras.preprocessing import image
from PIL import Image

t0 = time.time()
##############################################################################
"""args for models"""

args = {}
args['dim_h'] = 64          # factor controlling size of hidden layers
args['n_channel'] = 3       # number of channels in the input data 

args['n_z'] = 600     # number of dimensions in latent space. 

args['sigma'] = 1.0        # variance in n_z
args['lambda'] = 0.01      # hyper param for weight of discriminator loss
args['lr'] = 0.0002        # learning rate for Adam optimizer .000
args['epochs'] = 50         # how many epochs to run for
args['batch_size'] = 100   # batch size for SGD
args['save'] = True        # save weights at each epoch of training if True
args['train'] = True       # train networks if True, else load networks from

args['dataset'] = 'mnist' #'fmnist' # specify which dataset to use

##############################################################################



## create encoder model and decoder model
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']
        
        # convolutional filters, work excellent with image data
        self.conv = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
            #nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            #nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            #nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(self.dim_h * 8, self.dim_h * 16, 4, 2, 1, bias=False),
            
            nn.BatchNorm2d(self.dim_h * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h * 16, self.dim_h * 32, 4, 1, 0, bias=False)
        )
        # final layer is fully connected
        self.fc = nn.Linear(self.dim_h * (2 ** 5), self.n_z)
        

    def forward(self, x):
        print('enc')
        #print('input ',x.size()) #torch.Size([100, 3,32,32])
        x = self.conv(x)
        #print('conv ',x.size())
        x = x.squeeze()
        #print('aft squeeze ',x.size()) #torch.Size([128, 320])
        #aft squeeze  torch.Size([100, 320])
        x = self.fc(x)
        #print('out ',x.size()) #torch.Size([128, 20])
        #out  torch.Size([100, 300])
        return x


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']

        # first layer is fully connected
        self.fc = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 16 * 4 * 4),
            nn.ReLU())

        # deconvolutional filters, essentially inverse of convolutional filters
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 16, self.dim_h * 8, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 2, self.dim_h, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h, 3, 4, 2, 1),
            #nn.Sigmoid())
            nn.Tanh())

    def forward(self, x):
        print('dec')
        print('input ',x.size())
        x = self.fc(x)
        print('fc ',x.size())
        x = x.view(-1, self.dim_h *16, 4, 4)
        print('view ', x.size())
        x = self.deconv(x)
        print('deconv ', x.size())
        return x

##############################################################################

def biased_get_class1(c):
    
    xbeg = dec_x[dec_y == c]
    ybeg = dec_y[dec_y == c]
    
    return xbeg, ybeg
    #return xclass, yclass


def G_SM1(X, y,n_to_sample,cl):

    
    # fitting the model
    n_neigh = 5 + 1
    nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
    nn.fit(X)
    dist, ind = nn.kneighbors(X)

    # generating samples
    base_indices = np.random.choice(list(range(len(X))),n_to_sample)
    neighbor_indices = np.random.choice(list(range(1, n_neigh)),n_to_sample)

    X_base = X[base_indices]
    X_neighbor = X[ind[base_indices, neighbor_indices]]

    samples = X_base + np.multiply(np.random.rand(n_to_sample,1),
            X_neighbor - X_base)

    #use 10 as label because 0 to 9 real classes and 1 fake/smoted = 10
    return samples, [cl]*n_to_sample

#############################################################################
np.printoptions(precision=5,suppress=True)

dtrnimg = './shaver_shell_train'

ids = os.listdir(dtrnimg)
idtri_f = [os.path.join(dtrnimg, image_id) for image_id in ids]

actual_img_files = []
for path in idtri_f:
    try:
        image.load_img(path)
        if not('good' in path):
            actual_img_files.append(path)
    except:
        print('Ignoring invalid path: ', path)
        
idtri_f = actual_img_files
print('len idtri_f:', len(idtri_f))

#path on the computer where the models are stored
modpth = '../input/deepsmotedecsnapshot/models/crs5/'

encf = []
decf = []
for p in [0]:
    enc = modpth + f'/bst_enc{p}.pth'
    dec = modpth + f'/bst_dec{p}.pth'
    encf.append(enc)
    decf.append(dec)
    #print(enc)
    #print(dec)
    #print()

for m in range(0,1):
        
    images = []
    labels = []
    for im_i in range(len(idtri_f)):
        trnimgfile = idtri_f[im_i]
    
        img_orig = image.load_img(trnimgfile, target_size=(128, 128))
        dec_x = image.img_to_array(img_orig).astype(np.uint8)
        dec_x = np.moveaxis(dec_x, -1, 0)
        images.append(dec_x)
     
        if 'good' in trnimgfile:   
            dec_y = 1
        elif 'bad' in trnimgfile:
            dec_y = 0
        else:
            dec_y = 2
     
        dec_y = np.array(dec_y)
        labels.append(dec_y)
                            
    dec_x = np.array(images)
    dec_y = np.array(labels) 

    print('train imgs before reshape ',dec_x.shape) #(44993, 3072) 45500, 3072)
    print('train labels ',dec_y.shape) #(44993,) (45500,)

    classes = ('bad', 'good')
    
    #generate some images 
    train_on_gpu = torch.cuda.is_available()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    path_enc = encf[m]
    path_dec = decf[m]

    encoder = Encoder(args)
    encoder.load_state_dict(torch.load(path_enc), strict=False)
    encoder = encoder.to(device)

    decoder = Decoder(args)
    decoder.load_state_dict(torch.load(path_dec), strict=False)
    decoder = decoder.to(device)

    encoder.eval()
    decoder.eval()

    imbal = [100, 300]

    resx = []
    resy = []

    for i in range(0,1):
        xclass, yclass = biased_get_class1(i)
        print('xclass', xclass.shape) #(500, 3, 32, 32)
        print('yclass', yclass.shape) #(500,)
            
        #encode xclass to feature space
        xclass = torch.Tensor(xclass)
        xclass = xclass.to(device)
        xclass = encoder(xclass)
        print(xclass.shape) #torch.Size([500, 600])
            
        xclass = xclass.detach().cpu().numpy()
        n = imbal[1] - imbal[i]
        xsamp, ysamp = G_SM1(xclass,yclass,n,i)
        print('xsamp ',xsamp.shape) #(4500, 600)
        print('ysamp', len(ysamp)) #4500
        
        batch_size = 100
        for index in range(0,len(ysamp),batch_size):
            top = min(index+batch_size,len(ysamp))
            x_batch=xsamp[index:top, :]

            y_batch=ysamp[index:top]
        
            y_batch = np.array(y_batch)
    
            """to generate samples for resnet"""   
            x_batch = torch.Tensor(x_batch)
            x_batch = x_batch.to(device)
            #print(xsamp.size()) #torch.Size([10, 600, 1, 1])
            ximg = decoder(x_batch)

            ximn = ximg.detach().cpu().numpy()
            print(ximn.shape) #(4500, 3, 32, 32)
            #ximn = np.expand_dims(ximn,axis=1)
            print(ximn.shape) #(4500, 3, 32, 32)
            for im_idx, im in enumerate(ximn):
                label = y_batch[im_idx]
                ifile = './aug_data/'
                if label == 0:
                    ifile = ifile + 'bad/'
                else:
                    ifile = ifile + 'good/'
                ifile = ifile + f'im{index}_{im_idx}.png'
                out_im = torch.tensor(im[0]).mul_(255).clamp_(0, 255).to('cpu', torch.uint8).numpy()
                out_im = Image.fromarray(out_im)
                out_im.save(ifile)


t1 = time.time()
print('final time(min): {:.2f}'.format((t1 - t0)/60))
































