import time
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_gan_model
import util.util as util
from util.visualizer import Visualizer
import os
import numpy as np
# import torch
# from torch.autograd import Variable

import logging
# logging.basicConfig(level=logging.DEBUG)
# pil_logger = logging.getLogger('PIL')
# pil_logger.setLevel(logging.INFO)
# torch.manual_seed(1)
# torch.cuda.manual_seed(1)
import jittor
from jittor import nn
import jittor.transform as transform
jittor.flags.use_cuda = 1

np.random.seed(1)
opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

def update_learning_rate(opt):
    opt.lr = opt.lr_cycle * 0.8
    print("learning rate has been updated!!!!!!!!!!")

gan = create_gan_model(opt)

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)
visualizer = Visualizer(opt)

total_steps = (start_epoch-1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq
for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    if epoch < 2:
        stage = 1
    else:
        stage = 2 ### multistage ###
        # update_learning_rate(opt)
    print(stage, '!!!!!!!!!!!!!!!!!!!!')
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size

    for i, data in enumerate(dataset):
        # data_time = (time.time() - epoch_start_time) / (i+1)
        # print("load data costing %s" % data_time)
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta
        if not opt.use_attention:
            src_label = []
            trg_label = []
            for i in range(opt.num_of_frame):
                # src_label += [data['src_densepose'][i]]
                # trg_label += [data['trg_densepose'][i]]
                if opt.input_type == 0:
                    src_label += [jittor.concat((data['src_openpose'][i], data['src_densepose'][i]), dim=1)]
                    trg_label += [jittor.concat((data['trg_openpose'][i], data['trg_densepose'][i]), dim=1)]
                elif opt.input_type == 1:
                    src_label += [data['src_openpose'][i]]
                    trg_label += [data['trg_openpose'][i]]
                elif opt.input_type == 2:
                    src_label += [data['src_densepose'][i]]
                    trg_label += [data['trg_densepose'][i]]
        else:
            src_label, src_at_mask = gan.forward_attention(data['src_openpose'], data['src_densepose'])
            trg_label, trg_at_mask = gan.forward_attention(data['trg_openpose'], data['trg_densepose'])
        ############## Forward Pass ######################
        losses, src_fake, trg_fake, src2trg_mask, trg2src_mask, src2trg, trg2src, trg_cycle, src2trg_f = gan(
            src_label,  # label
            data['src_img'],
            data['src_template'],
            trg_label,  # label
            data['trg_img'],
            data['trg_template'], stage)
        # sum per device losses
        losses = [jittor.mean(x) if not isinstance(x, int) else x for x in losses]
        losses_dict = dict(zip(gan.loss_names, losses))

        loss_D_src = losses_dict['D_fake_src'] + losses_dict['D_real_src']
        loss_G_src = losses_dict['G_GAN_src'] + losses_dict['G_GAN_Feat_src'] + losses_dict['G_VGG_src']

        loss_D_trg = losses_dict['D_fake_trg'] + losses_dict['D_real_trg']
        loss_G_trg = losses_dict['G_GAN_trg'] + losses_dict['G_GAN_Feat_trg'] + losses_dict['G_VGG_trg'] + loss_G_src

        loss_D_A = losses_dict['A_real'] + losses_dict['A_fake']
        loss_A =  losses_dict['A'] + losses_dict['A_VGG'] + losses_dict['A_Feat'] + losses_dict['blend_reg']
        loss_openpose = losses_dict['openpose']
        loss_A = loss_A + loss_openpose + losses_dict['tv']
        ############### Backward Pass ####################

        # update generator weights
        # gan.optimizer_G_src.zero_grad()
        # gan.optimizer_G_src.backward(loss_G_src)
        # gan.optimizer_G_src.step()

        gan.optimizer_G_trg.zero_grad()
        gan.optimizer_G_trg.backward(loss_G_trg)
        gan.optimizer_G_trg.step()

        gan.optimizer_D_src.zero_grad()
        gan.optimizer_D_src.backward(loss_D_src)
        gan.optimizer_D_src.step()

        gan.optimizer_D_trg.zero_grad()
        gan.optimizer_D_trg.backward(loss_D_trg)
        gan.optimizer_D_trg.step()

        if stage == 2:
            gan.optimizer_A.zero_grad()
            gan.optimizer_A.backward(loss_A)
            gan.optimizer_A.step()

            gan.optimizer_D_A.zero_grad()
            gan.optimizer_D_A.backward(loss_D_A)
            gan.optimizer_D_A.step()


        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors_gan = {k: v.data.item() if not isinstance(v, int) else v for k, v in losses_dict.items()}
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors_gan, t)
            visualizer.plot_current_errors(errors_gan, total_steps)
            visualizer.print_line('')

        ### display output images

        if save_fake:
            visuals = OrderedDict([('trg_img', util.tensor2im(data['trg_img'][-1][0])),
                                   ('src_img', util.tensor2im(data['src_img'][-1][0])),
                                   ('trg_label', util.tensor2im(trg_label[-1].data[0][0:3])),
                                   ('src_label', util.tensor2im(src_label[-1].data[0][0:3])),
                                #  ('trg_densepose', util.tensor2im(trg_label[-1].data[0][3:6])),
                                #  ('src_densepose', util.tensor2im(src_label[-1].data[0][3:6])),
                                #    ('trg_cycle', util.tensor2im(trg_cycle.data[0])),
                                #    ('src2trg_f', util.tensor2im(src2trg_f.data[0])),
                                   ('trg_fake', util.tensor2im(trg_fake.data[0])),
                                   ('src_fake', util.tensor2im(src_fake.data[0])),
                                   ('src2trg', util.tensor2im(src2trg[:,-3:,...].data[0])),
                                   ('trg2src', util.tensor2im(trg2src[:,-3:,...].data[0])),
                                   ('src2trg_mask', util.tensor2im(src2trg_mask.data[0])),
                                   ('trg2src_mask', util.tensor2im(trg2src_mask.data[0])),
                                   ])
            if opt.input_type == 0:
                visuals['trg_openpose'] = util.tensor2im(trg_label[-1].data[0][3:6])
                visuals['src_openpose'] = util.tensor2im(src_label[-1].data[0][3:6])
            if opt.use_attention:
                visuals['src_at_mask'] = util.tensor2im(src_at_mask.data[0])
                visuals['trg_at_mask'] = util.tensor2im(trg_at_mask.data[0])
            visualizer.display_current_results(visuals, epoch, total_steps)
            ### show weights ###
            # visualizer.plot_current_weights(gan, total_steps)

        ### save 2000 model
        if total_steps == 2000:
            print('saving the 2000 model (epoch %d, total_steps %d)' % (epoch, total_steps))
            gan.save('2000')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            gan.save('latest')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break

    # end of epoch
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        gan.save('latest')
        gan.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')
