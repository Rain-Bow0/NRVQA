import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn import functional as Func
import skvideo.io
from PIL import Image
import os
import h5py
import numpy as np
import random
from argparse import ArgumentParser


class VideoDataset(Dataset):
    """Read data from the original dataset for   """

    def __init__(self, video_dir, video_name, score, videos_format='RGB', wid=None, heigh=None):

        super(VideoDataset, self).__init__()
        self.videos_dir = video_dir
        self.video_names = video_name
        self.score = score
        self.format = videos_format
        self.width = wid
        self.height = heigh

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx]

        assert self.format == 'YUV420' or self.format == 'RGB'

        if self.format == 'YUV420':
            video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_name), self.height, self.width,
                                          inputdict={'-pix_fmt': 'yuvj420p'})
        else:
            video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_name))
        video_score = self.score[idx]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        video_length = video_data.shape[0]
        video_channel = video_data.shape[3]
        video_height = video_data.shape[1]
        video_width = video_data.shape[2]
        transformed_video = torch.zeros([video_length, video_channel, video_height, video_width])
        for frame_idx in range(video_length):
            frame = video_data[frame_idx]
            frame = Image.fromarray(frame)
            frame = transform(frame)
            transformed_video[frame_idx] = frame

        sample = {'video': transformed_video,
                  'score': video_score}

        return sample


class ResNet50(nn.Module):

    def __init__(self):
        super(ResNet50, self).__init__()
        self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        # features@: 7->res5c
        for ii, model in enumerate(self.features):
            x = model(x)
            # [64, 2048, 17, 30]
            if ii == 7:
                features_mean = Func.adaptive_avg_pool2d(x, 1)

                features_std = global_std_pool2d(x)
                return features_mean, features_std


class Densenet121(nn.Module):
    def __init__(self):
        super(Densenet121, self).__init__()
        self.features = nn.Sequential(*list(models.densenet121(pretrained=True).features))
        self.len = len(self.features)
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        for i, model in enumerate(self.features):
            x = model(x)
            if i == self.len - 1:
                feature_mean = Func.adaptive_avg_pool2d(x, 1)
                feature_std = global_std_pool2d(x)
                return feature_mean, feature_std


def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)


def get_features(video_data, frame_batch_size=64):
    """feature extraction"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor = ResNet50().to(device)
    extractor.eval()
    extractor1 = Densenet121().to(device)
    extractor1.eval()
    output1 = torch.Tensor().to(device)
    output2 = torch.Tensor().to(device)
    output3 = torch.Tensor().to(device)
    output4 = torch.Tensor().to(device)

    video_length = video_data.shape[0]
    with torch.no_grad():

        for i in range(0, video_length, frame_batch_size):
            if i + frame_batch_size > video_length:
                batch = video_data[i: video_length].to(device)
            else:
                batch = video_data[i: i + frame_batch_size].to(device)

            features_mean, features_std = extractor(batch)
            output1 = torch.cat((output1, features_mean), 0)
            output2 = torch.cat((output2, features_std), 0)
            features_mean, features_std = extractor1(batch)
            output3 = torch.cat((output3, features_mean), 0)
            output4 = torch.cat((output4, features_std), 0)

        output = torch.cat((output1, output2, output3, output4), 1).squeeze()

    return output


if __name__ == "__main__":
    parser = ArgumentParser(description='"Extracting Content-Aware Perceptual Features using Pre-Trained ResNet-50')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--frame_batch_size', type=int, default=64,
                        help='frame batch size for feature extraction (default: 64)')
    parser.add_argument('--gpu', type=int, default=7,
                        help='frame batch size for feature extraction (default: 7)')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    videos_dir = r'videos/KoNViD_1k_videos'  # videos dir
    features_dir = 'CNN_features_KoNViD-1k/'  # features dir
    datainfo = r'KoNViD-1kinfo.mat'
    # database info: video_names, scores; video format, width, height, index,
    # ref_ids, max_len, etc.

    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    # read the .mat file
    Info = h5py.File(datainfo, 'r')
    video_names = [Info[Info['video_names'][0, :][i]][()].tobytes()[::2].decode() for i in
                   range(len(Info['video_names'][0, :]))]
    scores = Info['scores'][0, :]

    video_format = Info['video_format'][()].tobytes()[::2].decode()
    width = int(Info['width'][0])
    height = int(Info['height'][0])

    dataset = VideoDataset(videos_dir, video_names, scores, video_format, width, height)
    print(type(dataset))

    exit()
    # len(dataset) = 1200
    # dataset[0][video].shape = [192, 3, 540, 960]
    # dataset[0][scores] = **

    # 防止中途停止的方法
    if not os.path.exists('stop.txt'):
        with open('stop.txt', 'w') as f:
            f.write('-1')

    with open('stop.txt', 'r') as f:
        t = int(f.read())
    a = [9, 23, 27, 34, 53, 55, 114, 134, 135, 146, 152, 176, 187, 201, 215, 252, 263, 268, 306, 324, 328, 347, 356, 388, 427, 490, 491, 494, 499, 524, 641, 678, 687, 698, 709, 736, 741, 749, 770, 776, 778, 802, 812, 841, 866, 877, 893, 894, 896, 906, 939, 940, 946, 971, 977, 1024, 1049, 1057, 1065, 1078, 1079, 1085, 1102, 1116, 1122, 1128, 1143, 1161, 1168, 1170, 1174, 1176, 1177, 1179, 1183, 1187, 1189, 1191]


    for i in range(t + 1, len(dataset)):
        if i not in a:
            continue
        print(i)
        current_data = dataset[i]
        current_video = current_data['video']
        current_score = current_data['score']
        # print('Video {}: length {}'.format(i, current_video.shape[0]))
        features = get_features(current_video, args.frame_batch_size)
        # features.shape = [192, 4096]
        np.save(features_dir + str(i) + '_resnet-50_res5c', features.to('cpu').numpy())
        print(features.shape)
        np.save(features_dir + str(i) + '_score', current_score)
        with open('stop.txt', 'w') as f:
            f.write(str(i))
