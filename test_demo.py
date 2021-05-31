"""Test Demo for Quality Assessment of In-the-Wild Videos, ACM MM 2019"""
#
# Author: Dingquan Li
# Email: dingquanli AT pku DOT edu DOT cn
# Date: 2018/3/27
#
import torch
from torchvision import transforms
import skvideo.io
from PIL import Image
import numpy as np
import os
from regression import VQA
from CNNfeatures import get_features
from argparse import ArgumentParser
import time


if __name__ == "__main__":
    parser = ArgumentParser(description='"Test Demo of VQA')
    parser.add_argument('--model_path', default='models/1', type=str,
                        help='model path (default: models/1)')
    parser.add_argument('--video_path', default='test4.mp4', type=str,
                        help='video path (default: test4.mp4)')
    parser.add_argument('--video_format', default='RGB', type=str,
                        help='video format: RGB or YUV420 (default: RGB)')
    parser.add_argument('--video_width', type=int, default=None,
                        help='video width')
    parser.add_argument('--video_height', type=int, default=None,
                        help='video height')

    parser.add_argument('--frame_batch_size', type=int, default=32,
                        help='frame batch size for feature extraction (default: 32)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start = time.time()

    features_dir = 'cnn/'
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    if os.path.exists(features_dir + args.video_path + '.npy'):
        features = np.load(features_dir + args.video_path + '.npy')
        features = torch.from_numpy(features)
    else:
        # data preparation
        assert args.video_format == 'YUV420' or args.video_format == 'RGB'
        if args.video_format == 'YUV420':
            video_data = skvideo.io.vread(args.video_path, args.video_height, args.video_width,
                                          inputdict={'-pix_fmt': 'yuvj420p'})
        else:
            video_data = skvideo.io.vread(args.video_path)

        video_length = video_data.shape[0]
        video_channel = video_data.shape[3]
        video_height = video_data.shape[1]
        video_width = video_data.shape[2]
        transformed_video = torch.zeros([video_length, video_channel, video_height, video_width])
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        for frame_idx in range(video_length):
            frame = video_data[frame_idx]
            frame = Image.fromarray(frame)
            frame = transform(frame)
            transformed_video[frame_idx] = frame
        print('Video length: {}'.format(transformed_video.shape[0]))
        features = get_features(transformed_video, frame_batch_size=args.frame_batch_size)
        np.save(features_dir + args.video_path, features.to('cpu').numpy())

    features = torch.unsqueeze(features, 0)  # batch size 1



    model = VQA()
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))  #
    model.to(device)


    model.eval()
    # feature extraction


    with torch.no_grad():
        input_length = features.shape[1] * torch.ones(1, 1)
        outputs = model(features, input_length)
        y_pred = outputs[0][0].to('cpu').numpy()
        y_pred *= 100.0
        print("Predicted quality: {:.2f}".format(y_pred))

    end = time.time()

    print('Time: {} s'.format(end-start))
