# import torch
# import torch.nn as nn
# from torch.autograd import Variable
import jittor 
from jittor import nn

import functools
import numpy as np

from models import vggJittor
from .unet import UNet

# from .caffenet import *
# from .inception import *
###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # m.weight.data.normal_(0.0, 0.02)
        m.weight.gauss_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        # m.weight.data.normal_(1.0, 0.02)
        m.weight.gauss_(0.0, 0.02)
        m.bias.constant_(value=0.0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    # elif norm_type == 'spectral':
    #     norm_layer = jittor.nn.utils.spectral_norm
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

# def define_G(input_nc, output_nc, ngf, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1,
#              n_blocks_local=3, norm='instance', use_sigmoid=True, activation=nn.ReLU(), gpu_ids=[]):
#     norm_layer = get_norm_layer(norm_type=norm)
#     netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer, use_sigmoid=use_sigmoid, activation=activation)
#     if len(gpu_ids) > 0:
#         assert(torch.cuda.is_available())
#         netG.cuda(gpu_ids[0])
#     netG.apply(weights_init)
#     return netG
def define_openpose(gpu_ids):

    net_openpose = CaffeNet('./models/pose_deploy.prototxt')
    net_openpose.load_weights('./models/pose_iter_584000.caffemodel')
    # if len(gpu_ids) > 0:
    #     assert(torch.cuda.is_available())   
    #     net_openpose.cuda(gpu_ids[0])
    return net_openpose

def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, 
             n_blocks_local=3, norm='instance', gpu_ids=[],activation=nn.ReLU()):    
    norm_layer = get_norm_layer(norm_type=norm)     
    if netG == 'global':    
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer,padding_type='zero')       
    elif netG == 'local':        
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, 
                                  n_local_enhancers, n_blocks_local, norm_layer)
    elif netG == 'encoder':
        netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    elif netG == 'mask':
        netG=MaskGenerator(input_nc,output_nc, ngf,n_downsample_global, n_blocks_global, norm_layer,activation,gpu_ids)
    elif netG == 'unet':
        netG = UnetGenerator(input_nc, output_nc, n_downsample_global, ngf,norm_layer, False)
    elif netG == 'unet2':
        netG = UnetGenerator2(input_nc, output_nc, n_downsample_global, ngf,norm_layer, False)
    else:
        raise('generator not implemented!')
    # print(netG)
    # if len(gpu_ids) > 0:
    #     assert(torch.cuda.is_available())   
    #     netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG




def define_U(input_nc, output_nc, depth, gpu_ids=[]):
    netU = UNet(input_nc, output_nc, depth=depth, padding=True, batch_norm=True)
    # if len(gpu_ids) > 0:
    #     assert(torch.cuda.is_available())
    #     netU.cuda(gpu_ids[0])
    return netU

def define_E(input_nc, output_nc, ngf, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1,
             n_blocks_local=3, norm='instance', gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    netE = Encoder(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    # netE = nn.Conv2d(input_nc, ngf, 1, 1)
    # if len(gpu_ids) > 0:
    #     assert(torch.cuda.is_available())
    #     netE.cuda(gpu_ids[0])
    netE.apply(weights_init)
    return netE

def define_De(input_nc, output_nc, ngf, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1,
             n_blocks_local=3, norm='instance', activation=nn.ReLU(), mask=False, use_sigmoid=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    netDe = Decoder(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer, activation=activation, mask=mask, use_sigmoid=use_sigmoid)
    # if len(gpu_ids) > 0:
    #     assert(torch.cuda.is_available())
    #     netDe.cuda(gpu_ids[0])
    netDe.apply(weights_init)
    return netDe

def define_A(input_nc, output_nc, ngf, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1,
             n_blocks_local=3, norm='instance', gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    netA = MaskGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    # if len(gpu_ids) > 0:
    #     assert(torch.cuda.is_available())
    #     netA.cuda(gpu_ids[0])
    netA.apply(weights_init)
    return netA

def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    # netD = nn.Conv2d(input_nc, 1, 1, 1)
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)
    # if len(gpu_ids) > 0:
    #     assert(torch.cuda.is_available())
    #     netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                # real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                # self.real_label_var = Variable(real_tensor, requires_grad=False)
                self.real_label_var = jittor.zeros(input.size(), dtype=jittor.float32).stop_grad() + self.real_label
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                # fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                # self.fake_label_var = Variable(fake_tensor, requires_grad=False)
                self.fake_label_var = jittor.zeros(input.size(), dtype=jittor.float32).stop_grad() + self.fake_label
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):

        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def execute(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())   
        return loss


class Vgg19(jittor.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()

        vgg_pretrained_features = vggJittor.vgg19(pretrained=True).features
        self.slice1 = jittor.nn.Sequential()
        self.slice2 = jittor.nn.Sequential()
        self.slice3 = jittor.nn.Sequential()
        self.slice4 = jittor.nn.Sequential()
        self.slice5 = jittor.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def execute(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

##############################################################################
# Generator
##############################################################################
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model
        model_global = [model_global[i] for i in range(len(model_global)-3)] # get rid of final convolution layers
        self.model = nn.Sequential(*model_global)

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                                norm_layer(ngf_global), nn.ReLU(),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
                                norm_layer(ngf_global * 2), nn.ReLU()]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
                               norm_layer(ngf_global), nn.ReLU()]

            ### final convolution
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]

            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def execute(self, input):
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', use_sigmoid=True, activation=nn.ReLU()):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()

        model1 = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model1 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model1 += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample
        model2 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            '''
            model2 += [nn.Upsample(scale_factor = 2, mode='bilinear'),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(ngf * mult, int(ngf * mult / 2),kernel_size=3, stride=1, padding=0),
                      norm_layer(int(ngf * mult / 2)), activation]
            '''
            model2 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                        norm_layer(int(ngf * mult / 2)), activation]
        if use_sigmoid:
            model2 += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Sigmoid()]
        else:
            model2 += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)

    def execute(self, input):
        f = self.model1(input)
        x = self.model2(f)
        return x

class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def execute(self, input):
        return self.model(input)

class UnetGenerator2(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator2, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock2(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock2(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock2(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock2(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock2(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock2(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
        self.norm = nn.InstanceNorm2d(input_nc)
        self.model = unet_block

    def execute(self, x, y, flag=True):
        return self.model(self.norm(x), self.norm(y), flag)










# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=5,
                             stride=2, padding=2, bias=use_bias)
        # downrelu = nn.LeakyReLU(0.2, True)
        downrelu = nn.LeakyReLU(0.2)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU()
        upnorm = norm_layer(outer_nc)

        if outermost:
            upsamp = nn.Upsample(scale_factor = 2, mode='bilinear') #, align_corners=True)
            # upsamp_act = nn.ReflectionPad2d(1)
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=5, padding=2)

            # upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1)
            down = [downconv]
            up = [uprelu, upsamp, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            # upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1, bias=use_bias)
            upsamp = nn.Upsample(scale_factor = 2, mode='bilinear') #, align_corners=True)
            # upsamp_act = nn.ReflectionPad2d(1)
            upconv = nn.Conv2d(inner_nc, outer_nc, kernel_size=5, padding=2)

            down = [downrelu, downconv]
            up = [uprelu, upsamp, upconv, upnorm]
            model = down + up
        else:
            # upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1, bias=use_bias)
            upsamp = nn.Upsample(scale_factor = 2, mode='bilinear') #, align_corners=True)
            # upsamp_act = nn.ReflectionPad2d(1)
            upconv = nn.Conv2d(inner_nc*2, outer_nc, kernel_size=5, padding=2)

            down = [downrelu, downconv, downnorm]
            up = [uprelu, upsamp, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def execute(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return jittor.concat([x, self.model(x)], 1)        


class UnetSkipConnectionBlock2(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock2, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU()
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            # model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            # model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            # if use_dropout:
            #     model = down + [submodule] + up + [nn.Dropout(0.5)]
            # else:
            #     model = down + [submodule] + up
        self.up = nn.Sequential(*up)
        self.down = nn.Sequential(*down)
        self.submodule = submodule
        # self.model = nn.Sequential(*model)

    def execute(self, x, y = None, flag = True):
        if self.outermost:
            x_down = self.down(x)
            y_down = self.down(y)
            if flag:
                return self.up(self.submodule(x_down, y_down, flag = flag))
            else:
                return self.up(self.submodule(x_down + y_down, flag = flag))
        elif self.innermost:
            if flag:
                x_down = self.down(x)
                y_down = self.down(y)
                temp =  self.up(x_down + y_down)
                return jittor.concat([x + y, temp], 1)
            else:
                x_down = self.down(x)
                temp =  self.up(x_down)
                return jittor.concat([x, temp], 1)
        else:
            if flag:
                x_down = self.down(x)
                y_down = self.down(y)
                temp = self.up(self.submodule(x_down, y_down, flag = flag))
                return jittor.concat([x + y, temp], 1)
            else:
                x_down = self.down(x)
                temp = self.up(self.submodule(x_down, flag = flag))
                return jittor.concat([x, temp], 1)


























class MaskGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(MaskGenerator, self).__init__()
        activation = nn.LeakyReLU(0.2)

        model1 = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model1 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model1 += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample
        model2 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model2 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                        norm_layer(int(ngf * mult / 2)), activation]
        model2 += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)

    def execute(self, input):
        x = self.model1(input)
        x = self.model2(x)
        return x









# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def execute(self, x):
        out = x + self.conv_block(x)
        return out

class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(Encoder, self).__init__()
        activation = nn.ReLU()

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        self.model = nn.Sequential(*model)

    def execute(self, input):
        x = self.model(input)
        return x

class Decoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation=nn.ReLU(), mask=False, use_sigmoid=False):
        assert(n_blocks >= 0)
        super(Decoder, self).__init__()

        model2 = []
        mult = 2**n_downsampling



        for i in range(n_blocks):
            # model2 += [InceptionA(ngf * mult,pool_features=ngf * mult-224)]
             model2 += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            if mask:
                model2 += [nn.Upsample(scale_factor = 2, mode='bilinear'), #, align_corners=True),
                          nn.ReflectionPad2d(1),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),kernel_size=3, stride=1, padding=0),
                          norm_layer(int(ngf * mult / 2)), 
                          activation]
            else:
                model2 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                            norm_layer(int(ngf * mult / 2)), activation]

        model2 += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]

        self.model2 = nn.Sequential(*model2)

    def execute(self, label, template):
        x = self.model2(label + template)
        return x

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def singleD_execute(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def execute(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_execute(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
                norm_layer(nf),
                nn.LeakyReLU(0.2)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def execute(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


################### use attention ############################
class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,ngf):
        super(Attention_block,self).__init__()
        self.activation = nn.ReLU()

        self.W_g = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(F_g, ngf, kernel_size=3,stride=1,padding=0,bias=True),
            nn.InstanceNorm2d(ngf),
            self.activation
            )
        
        self.W_x = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(F_l, ngf, kernel_size=3,stride=1,padding=0,bias=True),
            nn.InstanceNorm2d(ngf),
            self.activation
            )

        model = [nn.Conv2d(ngf*2, 1, kernel_size=3, stride=1, padding=1), # conv
                nn.InstanceNorm2d(1),
                nn.Sigmoid()
                ]

        self.psi = nn.Sequential(*model)
        
    def execute(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # with torch.no_grad():
        # psi = g1+x1
        psi = jittor.concat((g1, x1), dim=1)
        psi = self.psi(psi)

        return x*psi + g*(1-psi), psi