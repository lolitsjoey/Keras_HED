import os
import numpy as np
from PIL import Image

def read_file_list(filelist):

    pfile = open(filelist)
    filenames = pfile.readlines()
    pfile.close()

    filenames = [f.strip() for f in filenames]

    return filenames

def split_pair_names(filenames, base_dir):

    filenames = [c.split(' ') for c in filenames]
    filenames = [(os.path.join(base_dir, c[0]), os.path.join(base_dir, c[1])) for c in filenames]

    return filenames


class DataParser():
    def __init__(self, batch_size_train, lst_file, parent_folder, train_folder):
        self.train_file = lst_file
        self.train_data_dir = parent_folder
        self.training_pairs = read_file_list(self.train_file)
        train_test_dir = split_pair_names(self.training_pairs, self.train_data_dir)
        self.samples = [(train_test_dir[0][0] + '/' + x, train_test_dir[0][1] + '/' + x) for x in os.listdir(train_test_dir[0][0])]
        
        
        self.all_ids = os.listdir(train_folder)
        np.random.shuffle(self.all_ids)
        
        train_split = 0.8
        self.training_ids = self.all_ids[:int(train_split * len(self.all_ids))]
        self.validation_ids = self.all_ids[int(train_split * len(self.all_ids)):]
        
        self.batch_size_train = batch_size_train
        assert len(self.training_ids) % batch_size_train == 0
        self.steps_per_epoch = len(self.training_ids)/batch_size_train
        
        assert len(self.validation_ids) % (batch_size_train*2) == 0
        self.validation_steps = len(self.validation_ids)/(batch_size_train*2)
        
        self.image_width = 480
        self.image_height = 480
        self.target_regression = True

    def get_training_batch(self):
    
        batch_ids = np.random.choice(self.training_ids, self.batch_size_train)
    
        return self.get_batch(batch_ids)
    
    def get_validation_batch(self):
    
        batch_ids = np.random.choice(self.validation_ids, self.batch_size_train*2)
    
        return self.get_batch(batch_ids)
    
    def get_all_train(self):
        
        ims, ems, _ = self.get_batch(self.training_ids)

        self.R = ims[..., 0].mean()
        self.G = ims[..., 1].mean()
        self.B = ims[..., 2].mean()

        ims[..., 0] -= self.R
        ims[..., 1] -= self.G
        ims[..., 2] -= self.B

        return ims. ems
    
    def get_all_valid(self, batch):
        ims,ems,_ = self.get_batch(self.validation_ids)

        ims[..., 0] -= self.R
        ims[..., 1] -= self.G
        ims[..., 2] -= self.B

        return ims. ems
    
    
    def get_batch(self, batch):

        filenames = []
        images = []
        edgemaps = []

        for idx, b in enumerate(batch):

            im = Image.open(self.samples[b][0])
            em = Image.open(self.samples[b][1])

            im = im.resize((self.image_width, self.image_height))
            em = em.resize((self.image_width, self.image_height))

            im = np.array(im, dtype=np.float32)
            im = im[..., ::-1] # RGB 2 BGR
            im[..., 0] -= 103.939
            im[..., 1] -= 116.779
            im[..., 2] -= 123.68

            # Labels needs to be 1 or 0 (edge pixel or not)
            # or can use regression targets as done by the author
            # https://github.com/s9xie/hed/blob/9e74dd710773d8d8a469ad905c76f4a7fa08f945/src/caffe/layers/image_labelmap_data_layer.cpp#L213

            em = np.array(em.convert('L'), dtype=np.float32)

            if self.target_regression:
                bin_em = em / 255.0
            else:
                bin_em = np.zeros_like(em)
                bin_em[np.where(em)] = 1

            # Some edge maps have 3 channels some dont
            bin_em = bin_em if bin_em.ndim == 2 else bin_em[:, :, 0]
            # To fit [batch_size, H, W, 1] output of the network
            bin_em = np.expand_dims(bin_em, 2)

            images.append(im)
            edgemaps.append(bin_em)
            filenames.append(self.samples[b])

        images   = np.asarray(images)
        edgemaps = np.asarray(edgemaps)

        return images, edgemaps, filenames
    
    
    
