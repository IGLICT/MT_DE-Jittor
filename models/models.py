### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
# import torch
import jittor

def create_gan_model(opt):
    from .gan import GAN, Inference
    if opt.isTrain:
        model = GAN()
    else:
        model = Inference()
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))
    # if opt.isTrain and len(opt.gpu_ids):
    #     model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    return model

def create_cross_attention(opt, netG, netD):
    from .cross_attention import CrossAttention, Inference
    if opt.isTrain:
        model = CrossAttention()
    else:
        model = Inference()
    model.initialize(opt, netG, netD)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))
    # if opt.isTrain and len(opt.gpu_ids):
    #     model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    return model
