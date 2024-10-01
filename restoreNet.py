import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.conv(x)
        return x

class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )
    def forward(self, x):
        x = self.mpconv(x)
        return x


class Encode(nn.Module):
    def __init__(self, in_channels):
        super(Encode, self).__init__()
        self.inc = InConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)
        
        
    def forward(self, x , depth):
        skipList = []

        x1 = self.inc(x)
        size = ( x1.shape[2] , x1.shape[3])
        resiz = torchvision.transforms.Resize(size,max_size=None, antialias=None)
        dep = resiz(depth)
        skipList.append(torch.cat([x1 , dep] ,  dim  = 1))

        x2 = self.down1(x1)
        size = ( x2.shape[2] , x2.shape[3])
        resiz = torchvision.transforms.Resize(size,max_size=None, antialias=None)
        dep = resiz(depth)
        skipList.append(torch.cat([x2 , dep] ,  dim  = 1))

        x3 = self.down2(x2)
        size = ( x3.shape[2] , x3.shape[3])
        resiz = torchvision.transforms.Resize(size,max_size=None, antialias=None)
        dep = resiz(depth)
        skipList.append(torch.cat([x3 , dep] ,  dim  = 1))

        x4 = self.down3(x3)
        return x4,skipList

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x):
        x = self.conv(x)
        return x

class Decode(nn.Module):
	def __init__( self):
		super(Decode,self).__init__()
		self.layer0 = nn.Conv2d(256,256,3,1)
		self.layer1 = Up(515,128)
		self.layer2 = Up(259,64)
		self.layer3 = Up(131,64)
		self.layer5 = DoubleConv(64,32)
		self.layer6 = DoubleConv(32,3)

	def forward(self,x,skipList):
		length = len(skipList)
		x = self.layer0(x)
		x = self.layer1(x,skipList[length - 1])
		x = self.layer2(x,skipList[length - 2])
		x = self.layer3(x,skipList[length - 3])
		x = self.layer5(x)
		x = self.layer6(x)
		return x

class generator(nn.Module):
	def __init__(self):
		super(generator,self).__init__()
		self.imageEncoder = Encode(3)
		self.decoder = Decode()
        
	def forward(self, x , depth):
		imageX, skipList = self.imageEncoder(x , depth)
		out = self.decoder(imageX, skipList)
		return out

class Discriminator(nn.Module):
    """ A 4-layer Markovian discriminator as described in the paper
    """
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            #Returns downsampling layers of each discriminator block
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if bn: layers.append(nn.BatchNorm2d(out_filters, momentum=0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 32, bn=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, 4, padding=1, bias=False),nn.Sigmoid()
        )

    def forward(self, img_A):
        # Concatenate image and condition image by channels to produce input
        #img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_A)

class GramMatrix(nn.Module):
    def forward(self, input):
        #print(input.size())
        a = 1
        b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, lays):
        super(StyleLoss, self).__init__()
        self.criterion = nn.L1Loss()
        self.lays = lays
        self.gram = GramMatrix()

    def forward(self, x_vgg1, x_vgg2):
        loss = 0
       # print(x_vgg1.size())
        for lay in self.lays:
            #print((x_vgg1[lay].size()))
            gram1 = self.gram.forward(x_vgg1[lay])
            gram2 = self.gram.forward(x_vgg2[lay])
            loss += 0.5 * torch.mean(torch.abs(gram1 - gram2))
        return loss

class VGG19_PercepLoss(nn.Module):
    """ Calculates perceptual loss in vgg19 space
    """
    def __init__(self, _pretrained_=True):
        super(VGG19_PercepLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=_pretrained_).features
        for param in self.vgg.parameters():
            param.requires_grad_(False)

    def get_features(self, image, layers=None):
        if layers is None: 
            layers = {'30': 'conv5_2'} # may add other layers
        features = {}
        x = image
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def forward(self, pred, true, layer='conv5_2'):
        true_f = self.get_features(true)
        pred_f = self.get_features(pred)
        return torch.mean((true_f[layer]-pred_f[layer])**2)



def test():
	inp = torch.rand((20,3,48,64))
	depth = torch.rand((20,3,48,64))
	genModel = generator()
	x = genModel(inp,depth)
	print('Generator OutShape = ' + str(x.shape))
 
test()

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)




#my stuff
class simpleNet(nn.Module):
    def __init__(self):
        super(simpleNet, self).__init__()
        self.rel = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 16,3 ,padding = 1)
        self.conv2 = nn.Conv2d(16, 32, 3 , padding = 1)
        self.conv3 = nn.Conv2d(32, 64, 3 , padding = 1)
        self.conv4 = nn.Conv2d(64, 32, 3 , padding = 1)
        self.conv5 = nn.Conv2d(32, 16,3, padding = 1)
        self.conv6 = nn.Conv2d(16, 3, 3 , padding = 1)

    def forward(self, x , y):
        y = x
        x = self.rel(self.conv1(x))
        x = self.rel(self.conv2(x))
        x = self.rel(self.conv3(x))
        x = self.rel(self.conv4(x))
        x = self.rel(self.conv5(x))
        x = self.rel(self.conv6(x))
        return x



