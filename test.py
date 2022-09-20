import os
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_gan_model
# from models.models import create_cross_attention
import util.util as util
from util.visualizer import Visualizer
from util import html
# from util.cat3 import cat3

# import torch
# from torch.autograd import Variable
# import torchvision.transforms as transforms

from PIL import Image
import scipy.io as sio

import jittor
from jittor import nn
import jittor.transform as transforms
jittor.flags.use_cuda = 1

opt = TestOptions().parse(save=False)
opt.nThreads = 0   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.sb = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# # test
# if not opt.engine and not opt.onnx:
#     gan = create_gan_model(opt)
#     # cross_attention = create_cross_attention(opt, gan.module.netG, gan.module.netD)
#     if opt.data_type == 16:
#         gan.half()
#         # cross_attention.half()
#     elif opt.data_type == 8:
#         gan.type(torch.uint8)
#         # cross_attention.type(torch.uint8)
# else:
#     from run_engine import run_trt_engine, run_onnx
gan = create_gan_model(opt)
    
score=[]
score2=[]
for i, data in enumerate(dataset):

    if not opt.use_attention:
        src_label = []
        trg_label = []

        if i >= opt.how_many:
            break
        for i in range(opt.num_of_frame):
            if opt.input_type == 0:
                    src_label += [jittor.concat((data['src_openpose'][i], data['src_densepose'][i]), dim=1)]
                    trg_label += [jittor.concat((data['trg_openpose'][i], data['trg_densepose'][i]), dim=1)]
            elif opt.input_type == 1:
                src_label += [data['src_openpose'][i]]
                trg_label += [data['trg_openpose'][i]]

            elif opt.input_type == 2:
                src_label += [data['src_densepose'][i]]
    else:
        with jittor.no_grad():
            src_label, src_at_mask = gan.forward_attention(data['src_openpose'], data['src_densepose'])
    # print(src_label[-1].shape)
    xx = Image.new('RGB', (src_label[-1].shape[3], src_label[-1].shape[2]), (128,128,128))

    transform_list = []
    transform_list += [transforms.ToTensor()]
    # transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
    #                                             (0.5, 0.5, 0.5))]
    transform_list += [transforms.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    temp = transforms.Compose(transform_list)
    xx = temp(xx)
    xx = jittor.unsqueeze(xx,0) #xx.unsqueeze(0) #torch.unsqueeze(xx,0)
    # print(template.shape)
    template = data['trg_template']
    src2trg, trg2src, src2trg_blend, trg2src_blend = gan.inference(src_label, trg_label, data['src_img'], data['trg_img'], data['src_template'], template)
    # print(s[0][-1].cpu().numpy())
    # score.append(s)
    # score2.append(s2)
    print("inference over")

    visuals = OrderedDict([('src_img', util.tensor2im(data['src_img'][-1][0])),
                           ('src2trg', util.tensor2im(src2trg[0])),
                           ('trg2src', util.tensor2im(trg2src[0])),
                           ('src2trg_blend', util.tensor2im(src2trg_blend[0])),
                           ('trg2src_blend', util.tensor2im(trg2src_blend[0]))
                           ])
    img_path = data['path']
    print('process image... %s' % img_path)
    import ntpath
    short_path = ntpath.basename(img_path[0])
    name = os.path.splitext(short_path)[0]
    image_dir = webpage.get_image_dir()
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    

    for label, image_numpy in visuals.items():
        if not os.path.exists(os.path.join(image_dir, label)):
            os.makedirs(os.path.join(image_dir, label))
        image_name = '%s.jpg' % (name)
        save_path = os.path.join(image_dir, label, image_name)
        util.save_image(image_numpy, save_path)
    # sio.savemat(image_dir+"/"+name+".mat",{'score':s[0][-1].cpu().numpy(),'score2':s[1][-1].cpu().numpy()})
webpage.save()
