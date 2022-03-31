
import glob
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms
from parameters import par
from helper import normalize_angle_delta


def get_data_info(folder_list, seq_len_range, overlap, sample_times=1):
    X_path, Y = [], []
    X_len = []
    for folder in folder_list:
        poses = np.load('{}{}.npy'.format(par.pose_path, folder))
        fpaths = glob.glob('{}{}/*.png'.format(par.image_path, folder))
        fpaths.sort()

        length_frames = len(fpaths)
        min_len, max_len = seq_len_range[0], seq_len_range[1]
        for i in range(sample_times):
            start = 0
            while True:
                n = np.random.random_integers(min_len, max_len)
                if start + n < length_frames:
                    x_seg = fpaths[start:start+n]
                    X_path.append(x_seg)
                    Y.append(poses[start:start+n])
                else:
                    break

                start += n - overlap
                X_len.append(len(x_seg))
        print('Folder {} finish'.format(folder))

    # 转化为 pandas 数据结构
    data = {'seq_len': X_len, 'image_path': X_path, 'pose': Y}
    df = pd.DataFrame(data, columns=['seq_len', 'image_path', 'pose'])
    df = df.sort_values(by=['seq_len'], ascending=False)  # 对数据帧排序
    return df


class SortedRandomBatchSampler(Sampler):
    def __init__(self, info_dataframe, batch_size, drop_last):
        self.df = info_dataframe
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.unique_seq_lens = sorted(self.df.iloc[:].seq_len.unique(), reverse=True)

        # 计算 len ( batches 数, 而不是 samples 数 )
        self.len = 0
        for v in self.unique_seq_lens:
            n_sample = len(self.df.loc[self.df.seq_len == v])
            n_batch = int(n_sample / self.batch_size)
            if not self.drop_last and n_sample % self.batch_size != 0:
                n_batch += 1
            self.len += n_batch

    def __iter__(self):

        # 计算每个 group 中的 sameples 数
        list_batch_indexes = []
        start_idx = 0
        for v in self.unique_seq_lens:
            n_sample = len(self.df.loc[self.df.seq_len == v])
            n_batch = int(n_sample / self.batch_size)
            if not self.drop_last and n_sample % self.batch_size != 0:
                n_batch += 1
            rand_idxs = (start_idx + torch.randperm(n_sample)).tolist()
            tmp = [rand_idxs[s*self.batch_size: s*self.batch_size+self.batch_size] for s in range(0, n_batch)]
            list_batch_indexes += tmp
            start_idx += n_sample
        return iter(list_batch_indexes)

    def __len__(self):
        return self.len


class ImageSequenceDataset(Dataset):
    def __init__(self, info_dataframe, resize_mode='crop', new_sizeize=None, img_mean=None, img_std=(1,1,1)):
        # Transformsrescale
        transform_ops = []
        if resize_mode == 'crop':
            transform_ops.append(transforms.CenterCrop((new_sizeize[0], new_sizeize[1])))
        elif resize_mode == 'rescale':
            transform_ops.append(transforms.Resize((new_sizeize[0], new_sizeize[1])))
        transform_ops.append(transforms.ToTensor())
        # transform_ops.append(transforms.Normalize(mean=img_mean, std=img_std))
        self.transformer = transforms.Compose(transform_ops)
        self.normalizer = transforms.Normalize(mean=img_mean, std=img_std)

        self.data_info = info_dataframe
        self.seq_len_list = list(self.data_info.seq_len)
        self.image_arr = np.asarray(self.data_info.image_path)  # image paths
        self.groundtruth_arr = np.asarray(self.data_info.pose)

    def __getitem__(self, index):
        raw_groundtruth = np.hsplit(self.groundtruth_arr[index], np.array([6]))
        groundtruth_sequence = raw_groundtruth[0]
        groundtruth_rotation = raw_groundtruth[1][0].reshape((3, 3)).T  # opposite rotation of the first frame
        groundtruth_sequence = torch.FloatTensor(groundtruth_sequence)

        groundtruth_sequence[1:] = groundtruth_sequence[1:] - groundtruth_sequence[0]  # get relative pose w.r.t. the first frame in the sequence

        # here we rotate the sequence relative to the first frame
        for gt_seq in groundtruth_sequence[1:]:
            location = torch.FloatTensor(groundtruth_rotation.dot(gt_seq[3:].numpy()))
            gt_seq[3:] = location[:]

        # get relative pose w.r.t. previous frame
        groundtruth_sequence[2:] = groundtruth_sequence[2:] - groundtruth_sequence[1:-1]

        # here we consider cases when rotation angles over Y axis go through PI -PI discontinuity
        for gt_seq in groundtruth_sequence[1:]:
            gt_seq[0] = normalize_angle_delta(gt_seq[0])

        image_path_sequence = self.image_arr[index]
        sequence_len = torch.tensor(self.seq_len_list[index])  # sequence_len = torch.tensor(len(image_path_sequence))

        image_sequence = []
        for img_path in image_path_sequence:
            img_as_img = Image.open(img_path)
            img_as_tensor = self.transformer(img_as_img)
            img_as_tensor = self.normalizer(img_as_tensor)
            img_as_tensor = img_as_tensor.unsqueeze(0)
            image_sequence.append(img_as_tensor)
        image_sequence = torch.cat(image_sequence, 0)
        return (sequence_len, image_sequence, groundtruth_sequence)

    def __len__(self):
        return len(self.data_info.index)

def get_data_info2(folder_list, seq_len_range, overlap):
    X_path, Y = [], []
    X_len = []
    for folder in folder_list:
        poses = np.load('{}{}.npy'.format(par.pose_path, folder))
        fpaths = glob.glob('{}{}/*.png'.format(par.image_path, folder))
        fpaths.sort()

        start_frames = [0]
        for st in start_frames:
            seq_len = seq_len_range[0]
            n_frames = len(fpaths) - st
            jump = seq_len - overlap
            res = n_frames % seq_len
            if res != 0:
                n_frames = n_frames - res
            x_segs = [fpaths[i:i + seq_len] for i in range(st, n_frames, jump)]
            y_segs = [poses[i:i + seq_len] for i in range(st, n_frames, jump)]
            Y += y_segs
            X_path += x_segs
            X_len += [len(xs) for xs in x_segs]

    data = {'seq_len': X_len, 'image_path': X_path, 'pose': Y}
    df = pd.DataFrame(data, columns=['seq_len', 'image_path', 'pose'])
    df = df.sort_values(by=['seq_len'], ascending=False)  # 对数据帧排序
    return df
