import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from data.keypoint2img import read_keypoints
from PIL import Image

from random import randint


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        ### input src labels (label maps)

        self.num_of_frame = opt.num_of_frame


        dir_src_openpose = '_src_openpose'
        self.dir_src_openpose = os.path.join(self.root, opt.phase + dir_src_openpose)
        self.src_openpose_paths = sorted(make_dataset(self.dir_src_openpose))

        dir_trg_openpose = '_trg_openpose'
        self.dir_trg_openpose = os.path.join(self.root, opt.phase + dir_trg_openpose)
        self.trg_openpose_paths = sorted(make_dataset(self.dir_trg_openpose))

        dir_src_densepose = '_src_densepose'
        self.dir_src_densepose = os.path.join(self.root, opt.phase + dir_src_densepose)
        self.src_densepose_paths = sorted(make_dataset(self.dir_src_densepose))

        dir_trg_densepose = '_trg_densepose'
        self.dir_trg_densepose = os.path.join(self.root, opt.phase + dir_trg_densepose)
        self.trg_densepose_paths = sorted(make_dataset(self.dir_trg_densepose))

        dir_src_img = '_src_img'
        self.dir_src_img = os.path.join(self.root, opt.phase + dir_src_img)
        self.src_img_paths = sorted(make_dataset(self.dir_src_img))

        dir_trg_img = '_trg_img'
        self.dir_trg_img = os.path.join(self.root, opt.phase + dir_trg_img)
        self.trg_img_paths = sorted(make_dataset(self.dir_trg_img))

        self.trg_size = len(self.trg_img_paths)
        self.src_size = len(self.src_img_paths)
        self.dataset_size = self.trg_size

    def __getitem__(self, index):
        src_index = index % self.src_size

        src_img_path = self.src_img_paths[src_index]
        src_img = [Image.open(src_img_path).convert('RGB')]

        src_template_path = self.src_img_paths[0]
        src_template_img = Image.open(src_template_path).convert('RGB')

        src_openpose_path = self.src_openpose_paths[src_index]
        src_openpose = [Image.open(src_openpose_path).convert('RGB')]
        # src_openpose = self.get_image(src_openpose_path, src_img.size, params, input_type='openpose')
        src_openpose = Image.fromarray(read_keypoints(src_openpose_path, src_img.size, random_drop_prob=0, remove_face_labels=False, basic_point_only=False))
        
        src_densepose_path = self.src_densepose_paths[src_index]
        src_densepose = [Image.open(src_densepose_path).convert('RGB')]


        trg_img_path = self.trg_img_paths[index]
        trg_img = [Image.open(trg_img_path).convert('RGB')]

        trg_template_path = self.trg_img_paths[0]
        trg_template_img = Image.open(trg_template_path).convert('RGB')

        trg_openpose_path = self.trg_openpose_paths[index]
        # trg_openpose = [Image.open(trg_openpose_path).convert('RGB')]
        trg_openpose = Image.fromarray(read_keypoints(trg_openpose_path, src_img.size, random_drop_prob=0, remove_face_labels=False, basic_point_only=False))

        trg_densepose_path = self.trg_densepose_paths[index]
        trg_densepose = [Image.open(trg_densepose_path).convert('RGB')]

        # print(src_img[-1].size)

        params = get_params(self.opt, trg_img[0].size)
        transform = get_transform(self.opt, params)

        if self.num_of_frame > 1:
            for i in range(self.num_of_frame-1):
                if src_index - i - 1 < 0:
                    before_img = Image.new('RGB', (src_img[-1].size[0], src_img[-1].size[1]), (128,128,128))
                    before_openpose = Image.new('RGB', (src_img[-1].size[0], src_img[-1].size[1]), (128,128,128))
                    before_densepose = Image.new('RGB', (src_img[-1].size[0], src_img[-1].size[1]), (128,128,128))
                else:
                    before_img = self.src_img_paths[src_index - i - 1]
                    before_img = Image.open(before_img).convert('RGB')

                    before_openpose = self.src_openpose_paths[src_index - i - 1]
                    before_openpose = Image.open(before_openpose).convert('RGB')
                    
                    before_densepose = self.src_densepose_paths[src_index - i - 1]
                    before_densepose = Image.open(before_densepose).convert('RGB')

                src_img +=[before_img]
                src_densepose += [before_densepose]
                src_openpose += [before_openpose]


                if index - i - 1 < 0:
                    before_img = Image.new('RGB', (src_img[-1].size[0], src_img[-1].size[1]), (128,128,128))
                    before_openpose = Image.new('RGB', (src_img[-1].size[0], src_img[-1].size[1]), (128,128,128))
                    before_densepose = Image.new('RGB', (src_img[-1].size[0], src_img[-1].size[1]), (128,128,128))
                else:
                    before = self.trg_img_paths[index - i - 1]
                    before_img = Image.open(before).convert('RGB')
                    before_openpose = self.trg_openpose_paths[index - i - 1]
                    before_openpose = Image.open(before_openpose).convert('RGB')
                    
                    before_densepose = self.trg_densepose_paths[index - i - 1]
                    before_densepose = Image.open(before_densepose).convert('RGB')
                trg_img +=[before_img]
                trg_densepose += [before_densepose]
                trg_openpose += [before_openpose]




        src_img = src_img[::-1]
        src_densepose = src_densepose[::-1]
        src_openpose = src_openpose[::-1]

        trg_img = trg_img[::-1]
        trg_densepose = trg_densepose[::-1]
        trg_openpose = trg_openpose[::-1]

        src_img = [transform(i) for i in src_img]
        src_openpose = [transform(i) for i in src_openpose]
        src_densepose = [transform(i) for i in src_densepose]

        trg_img = [transform(i) for i in trg_img]
        trg_densepose = [transform(i) for i in trg_densepose]
        trg_openpose = [transform(i) for i in trg_openpose]


        # # src_img_tensor = transform(src_img)
        # src_openpose_tensor = transform(src_openpose)
        # src_densepose_tensor = transform(src_densepose)
        src_template =  transform(src_template_img)

        # # trg_img_tensor = transform(trg_img)
        # trg_openpose_tensor = transform(trg_openpose)
        # trg_densepose_tensor = transform(trg_densepose)
        trg_template =  transform(trg_template_img)
        

        input_dict = {'src_img': src_img, 'trg_img': trg_img,
                      'src_openpose': src_openpose, 'trg_openpose': trg_openpose,
                      'src_densepose': src_densepose, 'trg_densepose': trg_densepose,
                      'trg_template': trg_template, 'src_template': src_template,'path': src_img_path}

        return input_dict

    def __len__(self):
        return len(self.trg_img_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'
