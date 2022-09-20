import os
import numpy as np

# import torch
# from torch.autograd import Variable

import jittor


from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
# from PerceptualSimilarity.models import dist_model as dm
from .cat3 import cat3
from .TVLoss import TVLoss
from PIL import Image
import logging

def get_blending_param(log_file_path, Num=100):
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    def extract_value(s):
        index1 = s.find("[")
        index2 = s.find("]")
        value = s[index1+1:index2]
        return float(value)
    blend_param = 0.0
    if len(lines) < Num:
        blend_param = 0.5
    else:
        for line in lines[-Num:]:
            blend_param += extract_value(line)
        blend_param /= Num
    return blend_param

class GAN(BaseModel):
    def name(self):
        return 'GAN'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        # if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
        #     torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain

        ### add logger for recording
        Log_Format = "%(levelname)s %(asctime)s - %(message)s"
        self.logger = logging.getLogger("SYT")
        log_file_path = os.path.join(opt.checkpoints_dir, opt.name, 'blending_log.txt')
        if self.isTrain and os.path.exists(log_file_path):
            input("The blending paramter file has existed! Are you sure to remove it?")
            os.remove(log_file_path)
        if self.isTrain:
            handler = logging.FileHandler(log_file_path)
            handler.setFormatter(logging.Formatter(Log_Format))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)
        self.step_cnt = 0

        ##### define networks
        # Generator network
        # print(opt.input_nc * opt.num_of_frame)
        if opt.input_type == 0:
            if opt.use_attention:
                print('use attention !!!!!!!!!!!!!!!!!!')
                print('use attention !!!!!!!!!!!!!!!!!!')
                print('use attention !!!!!!!!!!!!!!!!!!')
                input_att_nc = opt.input_nc
                self.net_Attention = networks.Attention_block(input_att_nc, input_att_nc, input_att_nc)
                if len(opt.gpu_ids) > 0:
                    self.net_Attention.cuda(opt.gpu_ids[0])
                input_nc = input_att_nc * opt.num_of_frame

            else:
                input_nc = opt.input_nc*2
                input_nc = input_nc * opt.num_of_frame
        else:
            input_nc = opt.input_nc * opt.num_of_frame

        self.netE_label = networks.define_E(input_nc, opt.output_nc, opt.ngf,
                                      opt.n_downsample_global, opt.n_blocks_global//2, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)
        self.netE_template_src = networks.define_E(opt.output_nc, opt.output_nc, opt.ngf,
                                      opt.n_downsample_global, opt.n_blocks_global//2, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)
        self.netE_template_trg = networks.define_E(opt.output_nc, opt.output_nc, opt.ngf,
                                      opt.n_downsample_global, opt.n_blocks_global//2, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)

        if opt.oned:
            self.netDe = networks.define_De(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.n_downsample_global, opt.n_blocks_global - opt.n_blocks_global//2, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids, mask=True)
        # # print(self.netE_label, self.netDe)
        else:
            self.netDe_src = networks.define_De(opt.input_nc, opt.output_nc, opt.ngf,
                                          opt.n_downsample_global, opt.n_blocks_global - opt.n_blocks_global//2, opt.n_local_enhancers,
                                          opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids, mask=True)
            self.netDe_trg = networks.define_De(opt.input_nc, opt.output_nc, opt.ngf,
                                          opt.n_downsample_global, opt.n_blocks_global - opt.n_blocks_global//2, opt.n_local_enhancers,
                                          opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids, mask=True)
        # self.netA = networks.define_U(opt.output_nc * 3, 3, opt.n_downsample_global, gpu_ids=self.gpu_ids)
        # print(self.netE_template_src)
        if self.opt.usemulti:
            input_nc = 3 * self.opt.num_of_frame
        else:
            input_nc = 3
        if self.opt.useA2:
            neta='unet2'
        else:
            neta='unet'


        self.netA = networks.define_G(input_nc, 3, opt.ngf, neta, opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
            opt.n_blocks_local,gpu_ids=self.gpu_ids)

        if self.isTrain:
            #self.blendingParam = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float32).cuda())
            self.blendingParam = jittor.nn.Parameter(jittor.array([0.5]))
        else:
            self.blendingParam = get_blending_param(log_file_path)
            print("load learnable blending param: %f" % self.blendingParam)

        # Discriminator network
        # if self.isTrain:

        netD_input_nc = opt.input_nc * opt.num_of_frame + opt.output_nc
        if self.opt.input_type == 0:
            if opt.use_attention:
                netD_input_nc = input_att_nc * opt.num_of_frame + opt.output_nc
            else:
                netD_input_nc = opt.input_nc * opt.num_of_frame * 2 + opt.output_nc

        if self.isTrain:
            self.netD_src = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, 'instance', opt.no_lsgan,
                                        opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
            self.netD_trg = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, 'instance', opt.no_lsgan,
                                        opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

            self.netD_A = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, 'instance', opt.no_lsgan,
                                        opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        if self.opt.useopenpose:
            print('use openpose!!!!')
            print('use openpose!!!!')
            print('use openpose!!!!')
            self.net_openpose = networks.define_openpose(self.gpu_ids)
        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netE_label, self.name() + 'E_label', opt.which_epoch, pretrained_path)
            self.load_network(self.netA, self.name() + 'A', opt.which_epoch, pretrained_path)
            self.load_network(self.netE_template_src, self.name() + 'E_template_src', opt.which_epoch, pretrained_path)
            self.load_network(self.netE_template_trg, self.name() + 'E_template_trg', opt.which_epoch, pretrained_path)
            if self.opt.use_attention:
                self.load_network(self.net_Attention, self.name() + 'Attention', opt.which_epoch, pretrained_path)
            if self.opt.oned:
                self.load_network(self.netDe, self.name() + 'De', opt.which_epoch, pretrained_path)
            else:
                self.load_network(self.netDe_src, self.name() + 'De_src', opt.which_epoch, pretrained_path)
                self.load_network(self.netDe_trg, self.name() + 'De_trg', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD_src, self.name() + 'D_src', opt.which_epoch, pretrained_path)
                self.load_network(self.netD_trg, self.name() + 'D_trg', opt.which_epoch, pretrained_path)
                self.load_network(self.netD_A, self.name() + 'D_A', opt.which_epoch, pretrained_path)
        if self.opt.verbose:
            print('---------- Networks initialized -------------')
        # print(">>>>>>>>>>>>>>>>> self.netE_label")
        # networks.print_network(self.netE_label)
        # print(">>>>>>>>>>>>>>>>> self.netE_template_src")
        # networks.print_network(self.netE_template_src)
        # print(">>>>>>>>>>>>>>>>> self.netE_template_trg")
        # networks.print_network(self.netE_template_trg)
        # print(">>>>>>>>>>>>>>>>> self.netDe_src")
        # networks.print_network(self.netDe_src)
        # print(">>>>>>>>>>>>>>>>> self.netDe_trg")
        # networks.print_network(self.netDe_trg)
        # print(">>>>>>>>>>>>>>>>> self.netD_src")
        # networks.print_network(self.netD_src)
        # print(">>>>>>>>>>>>>>>>> self.netD_trg")
        # networks.print_network(self.netD_trg)
        # import ipdb; ipdb.set_trace()

        # set loss functions and optimizers
        if self.isTrain:
            # define loss functions
            
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = jittor.nn.L1Loss()
            if opt.useL1:
                print('use L1 !!!!!!!!!!!!!!!!!!')
                print('use L1 !!!!!!!!!!!!!!!!!!')
                print('use L1 !!!!!!!!!!!!!!!!!!')
                self.criterionVGG = jittor.nn.L1Loss()
            else:
                self.criterionVGG = networks.VGGLoss(self.gpu_ids[0])
            # self.criterionPS = dm.DistModel()
            # self.criterionPS.initialize(model='net-lin', net='alex', use_gpu=True)
            self.criterionSpike = jittor.nn.L1Loss()
            self.criterionTV = TVLoss()

            # Names so we can breakout loss
            self.loss_names = ['D_fake_src', 'D_real_src', 'G_GAN_src', 'G_GAN_Feat_src', 'G_VGG_src',
                'D_fake_trg', 'D_real_trg', 'G_GAN_trg', 'G_GAN_Feat_trg', 'G_VGG_trg',
                'A', 'A_VGG', 'A_Feat','openpose', 'A_real', 'A_fake','tv', 'blend_reg']

            # initialize optimizers
            # optimizer G
            # params = list(self.netE_label.parameters())
            params = list()
            if self.opt.use_attention:
                params += list(self.net_Attention.parameters())
            params += list(self.netE_template_src.parameters())
            if self.opt.oned:
                params += list(self.netDe.parameters())
            else:
                params += list(self.netDe_src.parameters())
            # self.optimizer_G_src = jittor.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            params += list(self.netE_label.parameters())
            if self.opt.use_attention:
                params += list(self.net_Attention.parameters())
            params += list(self.netE_template_trg.parameters())
            if self.opt.oned:
                params += list(self.netDe.parameters())
            else:
                params += list(self.netDe_trg.parameters())
            self.optimizer_G_trg = jittor.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D
            if self.isTrain:
                params = list(self.netD_src.parameters())
                self.optimizer_D_src = jittor.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

                params = list(self.netD_trg.parameters())
                self.optimizer_D_trg = jittor.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

                params = list(self.netD_A.parameters())
                self.optimizer_D_A = jittor.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))


            params = list(self.netA.parameters())
            params += [self.blendingParam]
            self.optimizer_A = jittor.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input(self, inputs):
        for i, input in enumerate(inputs):
            if isinstance(input,list):
                for j,x in enumerate(input):
                    # input[j] = Variable(x.data.cuda())
                    input[j] = x#.data
            else:
                # inputs[i] = Variable(input.data.cuda())
                inputs[i] = input#.data #.cuda()

        return inputs

    def discriminate(self, netD, label, image):
        input_concat = jittor.concat((label.detach(), image.detach()), dim=1)
        return netD.execute(input_concat)

    def mean(self, a, b):
        # return (a + b)*0.5-(torch.sum((a + b)*0.5,(1,2)) / 512 / 512)
        # logging.debug("[SYT] blending param is %s" % (self.blendingParam.data))
        if self.isTrain and self.step_cnt % 1000 == 0:
            self.logger.debug("STEP: %d, blending param is %s" % (self.step_cnt, self.blendingParam.data))
        self.step_cnt += 1
        return a * self.blendingParam + b * (1 - self.blendingParam)
    
    def std(self, a, b):
        mean = self.mean(a, b)
        std = self.mean((a - mean) ** 2, (b - mean) ** 2) ** 0.5
        return std

    def execute_attention(self, openposes, denseposes):
        label = []
        assert len(openposes) == len(denseposes)
        openposes, denseposes = self.encode_input([openposes, denseposes])
        for i in range(self.opt.num_of_frame):
            pose, mask = self.net_Attention(openposes[i], denseposes[i])
            label += [pose]
        return label, mask

    def execute(self,
    src_label,
    src_image,
    src_template,
    trg_label,
    trg_image,
    trg_template,
    stage=1):

        # Encode Inputs
        if not self.opt.use_attention:
            src_label, src_image, src_template, trg_label, trg_image, trg_template = self.encode_input([src_label, src_image, src_template, trg_label, trg_image, trg_template])
        else:
            src_image, src_template, trg_image, trg_template = self.encode_input([src_image, src_template, trg_image, trg_template])
        
        # 3*num_of_frame
        src_label_concat = jittor.concat(src_label, dim=1)
        trg_label_concat = jittor.concat(trg_label, dim=1)
        # Fake Generation
        src_label_encoded = self.netE_label(src_label_concat)
        trg_label_encoded = self.netE_label(trg_label_concat)
        # 3*1
        src_template_encoded = self.netE_template_src(src_template)
        trg_template_encoded = self.netE_template_trg(trg_template)

        if not self.opt.use_flownet:
            if self.opt.oned:
                src_fake = self.netDe(src_label_encoded, src_template_encoded)

            # trg_input = jittor.concat((trg_label_encoded, trg_template_encoded), dim=1)
                trg_fake = self.netDe(trg_label_encoded, trg_template_encoded)
            else:
                src_fake = self.netDe_src(src_label_encoded, src_template_encoded)
                trg_fake = self.netDe_trg(trg_label_encoded, trg_template_encoded)


        with jittor.no_grad():
            if self.opt.usemulti:
                # logging.debug("[SYT] generate multi-images ... ")
                src2trg = []
                trg2src = []
                # src_label [-2, -1, 0]
                for i in range(self.opt.num_of_frame):
                    src_before_label = []
                    trg_before_label = []
                    # loop to get label to generate
                    for j in range(i+1):
                        src_before_label += [src_label[j]]
                        trg_before_label += [trg_label[j]]

                    src_before_label = src_before_label[::-1]
                    trg_before_label = trg_before_label[::-1]

                    for j in range(self.opt.num_of_frame - i - 1):
                        before_img = jittor.zeros(src_before_label[-1].size())#.cuda()

                        src_before_label +=[before_img]
                        trg_before_label +=[before_img]

                    src_before_label = src_before_label[::-1]
                    trg_before_label = trg_before_label[::-1]

                    src_before_label_concat = jittor.concat(src_before_label, dim=1)
                    trg_before_label_concat = jittor.concat(trg_before_label, dim=1)

                    src_before_label_encoded = self.netE_label(src_before_label_concat)
                    trg_before_label_encoded = self.netE_label(trg_before_label_concat)

                    if self.opt.oned:
                        src2trg += [self.netDe(src_before_label_encoded, trg_template_encoded)]
                        trg2src += [self.netDe(trg_before_label_encoded, src_template_encoded)]
                    else:
                        src2trg += [self.netDe_trg(src_before_label_encoded, trg_template_encoded)]
                        trg2src += [self.netDe_src(trg_before_label_encoded, src_template_encoded)]

                src2trg = jittor.concat(src2trg, dim=1)
                trg2src = jittor.concat(trg2src, dim=1)

            else:
                if self.opt.oned:
                    src2trg = self.netDe(src_label_encoded, trg_template_encoded)
                    trg2src = self.netDe(trg_label_encoded, src_template_encoded)
                else:
                    src2trg = self.netDe_trg(src_label_encoded, trg_template_encoded)
                    trg2src = self.netDe_src(trg_label_encoded, src_template_encoded)

        # Fake Detection and Loss
        pred_fake_src = self.discriminate(
            self.netD_src,
            src_label_concat,
            src_fake)
        loss_D_fake_src = self.criterionGAN(pred_fake_src, False)

        # Real Detection and Loss
        pred_real_src = self.discriminate(
            self.netD_src,
            src_label_concat,
            src_image[-1])
        loss_D_real_src = self.criterionGAN(pred_real_src, True)

        # GAN loss (Fake Passability Loss)
        input_fake_src = jittor.concat((
            src_label_concat,
            src_fake), dim=1)
        pred_pass_src = self.netD_src.execute(input_fake_src)
        loss_G_GAN_src = self.criterionGAN(pred_pass_src, True)

        pred_fake_trg = self.discriminate(
            self.netD_trg,
            trg_label_concat,
            trg_fake)
        loss_D_fake_trg = self.criterionGAN(pred_fake_trg, False)
        # Real Detection and Loss
        pred_real_trg = self.discriminate(
            self.netD_trg,
            trg_label_concat,
            trg_image[-1])
        loss_D_real_trg = self.criterionGAN(pred_real_trg, True)

        # GAN loss (Fake Passability Loss)
        input_fake_trg = jittor.concat((
            trg_label_concat,
            trg_fake), dim=1)
        pred_pass_trg = self.netD_trg.execute(input_fake_trg)
        loss_G_GAN_trg = self.criterionGAN(pred_pass_trg, True)

        loss_G_GAN_Feat_src = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake_src[i])-1):
                    loss_G_GAN_Feat_src += D_weights * feat_weights * \
                        self.criterionFeat(pred_pass_src[i][j], pred_real_src[i][j].detach()) * self.opt.lambda_gan_feat

        loss_G_GAN_Feat_trg = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake_trg[i])-1):
                    loss_G_GAN_Feat_trg += D_weights * feat_weights * \
                        self.criterionFeat(pred_pass_trg[i][j], pred_real_trg[i][j].detach()) * self.opt.lambda_gan_feat

        loss_G_VGG_src = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG_src = self.criterionVGG(src_fake, src_image[-1]) * self.opt.lambda_feat

        loss_G_VGG_trg = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG_trg = self.criterionVGG(trg_fake, trg_image[-1]) * self.opt.lambda_feat

        loss_A = 0
        loss_A_VGG = 0
        loss_A_Feat = 0
        loss_openpose = 0
        loss_op = 0
        loss_tv = 0

        src2trg_mask = jittor.zeros_like(src2trg)
        trg2src_mask = jittor.zeros_like(trg2src)
        trg_cycle = trg2src
        src2trg_f = src2trg
        if stage == 2:
            if self.opt.useA2:
                if self.opt.usemulti:
                    src2trg_mask = self.netA(jittor.concat(src_image, dim=1),src2trg.detach(), False)
                    trg2src_mask = self.netA(trg2src.detach(),jittor.concat(trg_image, dim=1), False)
                else:
                    src2trg_mask = self.netA(src_image[-1],src2trg.detach(), False)
                    trg2src_mask = self.netA(trg2src.detach(),trg_image[-1], False)
            else:
                if self.opt.usemulti:
                    src2trg_mask = self.netA(self.mean(jittor.concat(src_image, dim=1), src2trg.detach())) 
                                                # self.std(src_image, src2trg.detach()), 
                                            #jittor.concat(src_label,dim=1)), dim=1))
                    trg2src_mask = self.netA(self.mean(trg2src.detach(), jittor.concat(trg_image, dim=1)))
                                                # self.std(trg2src.detach(), trg_image), 
                                            # jittor.concat(trg_label,dim=1)), dim=1))
                else:
                    #print(src_image[-1].shape,src2trg.shape)
                    src2trg_mask = self.netA(self.mean(src_image[-1], src2trg.detach()))
                    trg2src_mask = self.netA(self.mean(trg2src.detach(), trg_image[-1]))

        loss_A_fake = 0
        if stage == 2:
            if self.opt.useres:
                pred_fake_A = self.discriminate(
                    self.netD_A,
                    trg_label_concat,
                    trg2src_mask+trg_fake.detach())
            else:
                pred_fake_A = self.discriminate(
                    self.netD_A,
                    trg_label_concat,
                    trg2src_mask)
            loss_A_fake = self.criterionGAN(pred_fake_A, False) 

        loss_A_real = 0
        if stage == 2:
            pred_real_A = self.discriminate(
                self.netD_A,
                trg_label_concat,
                trg_image[-1])
            loss_A_real = self.criterionGAN(pred_real_A, True)

        loss_A = 0
        if stage == 2:
            if self.opt.useres:
                # pred_A_src2trg = self.netD_trg(jittor.concat((src_label_concat, src2trg_mask+src2trg.detach()), dim=1))
                pred_pass_trg2src = self.netD_A(jittor.concat((trg_label_concat.detach(), trg2src_mask+trg_fake.detach()), dim=1))
            else:
                # pred_A_src2trg = self.netD_trg(jittor.concat((src_label_concat, src2trg_mask), dim=1))
                pred_pass_trg2src = self.netD_A(jittor.concat((trg_label_concat.detach(), trg2src_mask), dim=1))
            loss_A = self.criterionGAN(pred_pass_trg2src, True)

        loss_A_VGG = 0
        if stage == 2:
            if not self.opt.no_vgg_loss:
                if self.opt.useres:
                    # print('use res!!!!!!!!!!!')
                    # print('use res!!!!!!!!!!!')
                    # print('use res!!!!!!!!!!!')
                    
                    loss_A_VGG = self.criterionVGG(trg2src_mask + trg_fake.detach(), trg_image[-1]) * self.opt.lambda_feat
                    #loss_A_VGG = self.criterionVGG(trg2src_mask + trg_fake.detach(), trg_image[-1]) * self.opt.lambda_feat
                else:
                    #loss_A_VGG = self.criterionVGG(trg2src_mask, trg_image[-1]) * self.opt.lambda_feat
                    loss_A_VGG = self.criterionVGG(trg2src_mask, trg_image[-1]) * self.opt.lambda_feat
                    # loss_A_VGG = torch.nn.L1Loss().execute(trg2src_mask, trg_image[-1]) * self.opt.lambda_feat

        loss_A_Feat = 0
        if stage == 2:
            if not self.opt.no_ganFeat_loss:
                feat_weights = 4.0 / (self.opt.n_layers_D + 1)
                D_weights = 1.0 / self.opt.num_D
                for i in range(self.opt.num_D):
                    for j in range(len(pred_fake_trg[i])-1):
                        loss_A_Feat += D_weights * feat_weights * \
                            self.criterionFeat(pred_pass_trg2src[i][j], pred_real_A[i][j].detach()) * self.opt.lambda_gan_feat

        loss_blend_reg = 0
        if stage == 2:
            loss_blend_reg = (self.blendingParam - 0.5) ** 2 * 100
        
        if self.opt.useopenpose:
            G_heatmap = self.net_openpose(src2trg_mask)
            B_heatmap = self.net_openpose(src_image)
            loss_openpose = jittor.nn.L1Loss().execute(G_heatmap['net_output'],B_heatmap['net_output'].detach())*100

        
        if self.opt.usetv:
            loss_tv = self.opt.lambda_tv * self.criterionTV(trg2src_mask)


        return [loss_D_fake_src, loss_D_real_src, loss_G_GAN_src, loss_G_GAN_Feat_src, loss_G_VGG_src,
                loss_D_fake_trg, loss_D_real_trg, loss_G_GAN_trg, loss_G_GAN_Feat_trg, loss_G_VGG_trg,
                loss_A, loss_A_VGG, loss_A_Feat,loss_openpose, loss_A_fake, loss_A_real,loss_tv, loss_blend_reg], src_fake, trg_fake, src2trg_mask, trg2src_mask, src2trg, trg2src,trg_cycle, src2trg_f

    def inference(self, src_label, trg_label, src_image, trg_image, src_template, trg_template):
        # Encode Inputs
        if not self.opt.use_attention:
            src_label, src_image, trg_image, trg_template, trg_label, src_template = self.encode_input([src_label, src_image, trg_image, trg_template, trg_label, src_template])
        else:
            src_image, trg_template = self.encode_input([src_image, trg_template])
            self.net_Attention.eval()

        src_label_concat = jittor.concat(src_label, dim=1)
        trg_label_concat = jittor.concat(trg_label, dim=1)

        self.netE_label.eval()
        self.netE_template_trg.eval()
        self.netE_template_src.eval()
        if self.opt.oned:
            self.netDe.eval()
        else:
            self.netDe_src.eval()
            self.netDe_trg.eval()

        self.netA.eval()
        # self.netD_trg.eval()
        # self.netD_A.eval()
        # Fake Generation
        with jittor.no_grad():
            src_label_encoded = self.netE_label(src_label_concat)
            trg_label_encoded = self.netE_label(trg_label_concat)
            trg_template_encoded = self.netE_template_trg(trg_template)
            src_template_encoded = self.netE_template_src(src_template)

            if self.opt.usemulti:

           # if self.opt.usemulti:
                src2trg = []
                trg2src = []
                # src_label [-2, -1, 0]
                for i in range(self.opt.num_of_frame):
                    src_before_label = []
                    trg_before_label = []
                    # loop to get label to generate
                    for j in range(i+1):
                        src_before_label += [src_label[j]]
                        trg_before_label += [trg_label[j]]

                    src_before_label = src_before_label[::-1]
                    trg_before_label = trg_before_label[::-1]

                    for j in range(self.opt.num_of_frame - i - 1):
                        before_img = jittor.zeros(src_before_label[-1].size()) #.cuda()

                        src_before_label +=[before_img]
                        trg_before_label +=[before_img]

                    src_before_label = src_before_label[::-1]
                    trg_before_label = trg_before_label[::-1]

                    src_before_label_concat = jittor.concat(src_before_label, dim=1)
                    trg_before_label_concat = jittor.concat(trg_before_label, dim=1)

                    src_before_label_encoded = self.netE_label(src_before_label_concat)
                    trg_before_label_encoded = self.netE_label(trg_before_label_concat)

                    if self.opt.oned:
                        src2trg += [self.netDe(src_before_label_encoded, trg_template_encoded)]
                        trg2src += [self.netDe(trg_before_label_encoded, src_template_encoded)]
                    else:
                        src2trg += [self.netDe_trg(src_before_label_encoded, trg_template_encoded)]
                        trg2src += [self.netDe_src(trg_before_label_encoded, src_template_encoded)]
                src2trg = jittor.concat(src2trg, dim=1)
                trg2src = jittor.concat(trg2src, dim=1)


            else:
                if self.opt.oned:
                    src2trg = self.netDe(src_label_encoded, trg_template_encoded)
                else:
                    src2trg = self.netDe_trg(src_label_encoded, trg_template_encoded)
            if self.opt.useA2:
                if self.opt.usemulti:
                    src2trg_mask = self.netA(jittor.concat(src_image,dim=1),src2trg.detach(), False)
                else:
                    src2trg_mask = self.netA(src_image[-1],src2trg.detach(), False)
            else:
                if self.opt.usemulti:
                    src2trg_mask = self.netA(self.mean(jittor.concat(src_image,dim=1), src2trg.detach()))
                    trg2src_mask = self.netA(self.mean(trg2src.detach() ,jittor.concat(trg_image,dim=1)))

                else:
                    src2trg_mask = self.netA(self.mean(src_image[-1], src2trg.detach()))

            # score = self.netD_trg(jittor.concat((src_label_concat,src2trg_mask), dim=1))
            # score2 = self.netD_A(jittor.concat((src_label_concat,src2trg_mask), dim=1))

        if self.opt.useres:
            if self.opt.usemulti:
                return src2trg, src2trg_mask++src2trg[:,-3::,:,:], score, score2
            else:
                return src2trg, src2trg_mask+src2trg, score, score2
        else:
            return src2trg[:,-3::,:,:],trg2src[:,-3::,:,:], src2trg_mask, trg2src_mask

    def save(self, which_epoch):
        self.save_network(self.netE_label, self.name() + 'E_label', which_epoch, self.gpu_ids)
        self.save_network(self.netA, self.name() + 'A', which_epoch, self.gpu_ids)
        self.save_network(self.netE_template_src, self.name() + 'E_template_src', which_epoch, self.gpu_ids)
        self.save_network(self.netE_template_trg, self.name() + 'E_template_trg', which_epoch, self.gpu_ids)
        if self.opt.use_attention:
            self.save_network(self.net_Attention, self.name() + 'Attention', which_epoch, self.gpu_ids)
        if self.opt.oned:
            self.save_network(self.netDe, self.name() + 'De', which_epoch, self.gpu_ids)
        else:
            self.save_network(self.netDe_src, self.name() + 'De_src', which_epoch, self.gpu_ids)
            self.save_network(self.netDe_trg, self.name() + 'De_trg', which_epoch, self.gpu_ids)
        self.save_network(self.netD_src, self.name() + 'D_src', which_epoch, self.gpu_ids)
        self.save_network(self.netD_trg, self.name() + 'D_trg', which_epoch, self.gpu_ids)
        self.save_network(self.netD_A, self.name() + 'D_A', which_epoch, self.gpu_ids)

class Inference(GAN):
    def execute(self, src_label, src_image, trg_template):
        return self.inference(src_label, src_image, trg_template)