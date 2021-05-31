from argparse import ArgumentParser
import os
import h5py
import torch
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random
from scipy import stats
from tensorboardX import SummaryWriter
import datetime


class VQADataset(Dataset):
    def __init__(self, features_dir='CNN_features_KoNViD-1k/', index=None, max_len=240, feat_dim=4096 + 2048, scale=1):
        super(VQADataset, self).__init__()
        self.features = np.zeros((len(index), max_len, feat_dim))
        self.length = np.zeros((len(index), 1))
        self.mos = np.zeros((len(index), 1))
        for i in range(len(index)):
            features = np.load(features_dir + str(index[i]) + '_resnet-50_res5c.npy')
            self.length[i] = features.shape[0]

            self.features[i, :features.shape[0], :] = features
            self.mos[i] = np.load(features_dir + str(index[i]) + '_score.npy')  #
        self.scale = scale
        self.label = self.mos / self.scale  # label normalization

    def __len__(self):
        return len(self.mos)

    def __getitem__(self, idx):
        sample = self.features[idx], self.length[idx], self.label[idx]
        return sample


class ANN(nn.Module):
    def __init__(self, input_size=4096 + 2048, reduced_size=128, n_ANNlayers=1, dropout_p=0.5):
        super(ANN, self).__init__()
        self.n_ANNlayers = n_ANNlayers
        self.fc0 = nn.Linear(input_size, 1024)
        self.fc1 = nn.Linear(1024, reduced_size)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(reduced_size, reduced_size)  #

    def forward(self, input):
        input = self.fc0(input)
        input = self.fc1(input)
        for i in range(self.n_ANNlayers - 1):  # nonlinear
            input = self.fc(self.dropout(F.relu(input)))
        return input



def TP(q, tau=12, beta=0.5):
    """subjectively-inspired temporal pooling"""
    q = torch.unsqueeze(torch.t(q), 0)
    qm = -float('inf') * torch.ones((1, 1, tau - 1)).to(q.device)
    qp = 10000.0 * torch.ones((1, 1, tau - 1)).to(q.device)  #
    l = -F.max_pool1d(torch.cat((qm, -q), 2), tau, stride=1)
    m = F.avg_pool1d(torch.cat((q * torch.exp(-q), qp * torch.exp(-qp)), 2), tau, stride=1)
    n = F.avg_pool1d(torch.cat((torch.exp(-q), torch.exp(-qp)), 2), tau, stride=1)
    m = m / n
    return beta * m + (1 - beta) * l


class VQA(nn.Module):
    def __init__(self, input_size=4096+2048, reduced_size=128, hidden_size=32):
        super(VQA, self).__init__()
        self.hidden_size = hidden_size
        self.ann = ANN(input_size, reduced_size, 1)
        self.rnn = nn.LSTM(reduced_size, hidden_size, batch_first=True)
        self.q = nn.Linear(hidden_size, 1)

    def forward(self, input, input_length):
        input = self.ann(input)  # dimension reduction
        outputs, _ = self.rnn(input, self._get_initial_state(input.size(0), input.device))
        q = self.q(outputs)  # frame quality
        score = torch.zeros_like(input_length, device=q.device)  #
        for i in range(input_length.shape[0]):
            qi = q[i, :np.int(input_length[i].numpy())]
            qi = TP(qi)
            score[i] = torch.mean(qi)  # video overall quality
        return score

    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h0, c0


if __name__ == "__main__":

    parser = ArgumentParser(description='"VQA: Quality Assessment of In-the-Wild Videos')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='learning rate (default: 0.00001)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')

    parser.add_argument('--model', default='VQA', type=str,
                        help='model name (default: VQA)')
    parser.add_argument('--exp_id', default=0, type=int,
                        help='exp id for train-val-test splits (default: 0)')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='test ratio (default: 0.2)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='val ratio (default: 0.2)')

    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='weight decay (default: 0.0005)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='the number of gpu (default: 7)')

    parser.add_argument("--log_dir", type=str, default="logs",
                        help="log directory for Tensorboard log output")

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    args.decay_interval = int(args.epochs / 20)
    args.decay_ratio = 0.9

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(args.exp_id)
        torch.backends.cudnn.deterministic = True  # 设置是否复现
        torch.backends.cudnn.benchmark = True  # 由于图片尺寸一样，设置成True能加速
    else:
        device = torch.device('cpu')
        torch.manual_seed(args.exp_id)

    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    features_dir = 'CNN_features_KoNViD-1k/'  # features dir
    datainfo = 'KoNViD-1kinfo.mat'  # database info: video_names, scores;
    # video format, width, height, index, ref_ids, max_len, etc.

    print('EXP ID: {}'.format(args.exp_id))

    Info = h5py.File(datainfo, 'r')  # index, ref_ids
    max_len = int(Info['max_len'][0])

    index = [i for i in range(1200)]
    random.shuffle(index)

    train_index = index[0: int(1200 * (1 - args.test_ratio - args.val_ratio))]
    val_index = index[int(1200 * (1 - args.test_ratio - args.val_ratio)): int(1200 * (1 - args.test_ratio))]
    test_index = index[int(1200 * (1 - args.test_ratio)): 1200]

    scale = Info['scores'][0, :].max()  # label normalization factor 4.64

    train_dataset = VQADataset(features_dir, train_index, max_len, scale=scale)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = VQADataset(features_dir, val_index, max_len, scale=scale)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset)

    if args.test_ratio > 0:
        test_dataset = VQADataset(features_dir, test_index, max_len, scale=scale)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset)

    model = VQA().to(device)

    if not os.path.exists('models'):
        os.makedirs('models')
    trained_model_file = 'models/{}-{}'.format(args.model, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
    if not os.path.exists('results'):
        os.makedirs('results')
    save_result_file = 'results/{}-EXP{}'.format(args.model, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))

    writer = SummaryWriter(log_dir='{}/EXP{}-{}-{}-{}-{}-{}'
                           .format(args.log_dir, args.exp_id, args.model,
                                   args.lr, args.batch_size, args.epochs,
                                   datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")))

    criterion = nn.L1Loss()  # L1 loss
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_interval, gamma=args.decay_ratio)
    best_val_criterion = -1  # SROCC min
    for epoch in range(args.epochs):
        # Train
        print(epoch)
        model.train()
        L = 0
        for i, (features, length, label) in enumerate(train_loader):
            features = features.to(device).float()
            label = label.to(device).float()
            optimizer.zero_grad()  #
            outputs = model(features, length.float())
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            L = L + loss.item()
        train_loss = L / (i + 1)

        model.eval()
        # Val
        y_pred = np.zeros(len(val_index))
        y_val = np.zeros(len(val_index))
        L = 0
        with torch.no_grad():
            for i, (features, length, label) in enumerate(val_loader):
                y_val[i] = scale * label.item()  #
                features = features.to(device).float()
                label = label.to(device).float()
                outputs = model(features, length.float())
                y_pred[i] = scale * outputs.item()
                loss = criterion(outputs, label)
                L = L + loss.item()
        val_loss = L / (i + 1)
        val_PLCC = stats.pearsonr(y_pred, y_val)[0]
        val_SROCC = stats.spearmanr(y_pred, y_val)[0]
        val_RMSE = np.sqrt(((y_pred - y_val) ** 2).mean())
        val_KROCC = stats.stats.kendalltau(y_pred, y_val)[0]

        # Test
        if args.test_ratio > 0:
            y_pred = np.zeros(len(test_index))
            y_test = np.zeros(len(test_index))
            L = 0
            with torch.no_grad():
                for i, (features, length, label) in enumerate(test_loader):
                    y_test[i] = scale * label.item()  #
                    features = features.to(device).float()
                    label = label.to(device).float()
                    outputs = model(features, length.float())
                    y_pred[i] = scale * outputs.item()
                    loss = criterion(outputs, label)
                    L = L + loss.item()
            test_loss = L / (i + 1)
            PLCC = stats.pearsonr(y_pred, y_test)[0]
            SROCC = stats.spearmanr(y_pred, y_test)[0]
            RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())
            KROCC = stats.stats.kendalltau(y_pred, y_test)[0]
        # 可视化数据

        writer.add_scalar("loss/train", train_loss, epoch)  #
        writer.add_scalar("loss/val", val_loss, epoch)  #
        writer.add_scalar("SROCC/val", val_SROCC, epoch)  #
        writer.add_scalar("KROCC/val", val_KROCC, epoch)  #
        writer.add_scalar("PLCC/val", val_PLCC, epoch)  #
        writer.add_scalar("RMSE/val", val_RMSE, epoch)  #
        if args.test_ratio > 0:
            writer.add_scalar("loss/test", test_loss, epoch)  #
            writer.add_scalar("SROCC/test", SROCC, epoch)  #
            writer.add_scalar("KROCC/test", KROCC, epoch)  #
            writer.add_scalar("PLCC/test", PLCC, epoch)  #
            writer.add_scalar("RMSE/test", RMSE, epoch)  #

        # Update the model with the best val_SROCC
        if val_SROCC > best_val_criterion:
            print("EXP ID={}: Update best model using best_val_criterion in epoch {}".format(args.exp_id, epoch))
            print("Val results: val loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                  .format(val_loss, val_SROCC, val_KROCC, val_PLCC, val_RMSE))
            if args.test_ratio > 0:
                print("Test results: test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                      .format(test_loss, SROCC, KROCC, PLCC, RMSE))

            torch.save(model.state_dict(), trained_model_file)
            best_val_criterion = val_SROCC  # update best val SROCC

    # Test
    if args.test_ratio > 0:
        model.load_state_dict(torch.load(trained_model_file))  #
        model.eval()
        with torch.no_grad():
            y_pred = np.zeros(len(test_index))
            y_test = np.zeros(len(test_index))
            L = 0
            for i, (features, length, label) in enumerate(test_loader):
                y_test[i] = scale * label.item()  #
                features = features.to(device).float()
                label = label.to(device).float()
                outputs = model(features, length.float())
                y_pred[i] = scale * outputs.item()
                loss = criterion(outputs, label)
                L = L + loss.item()
        test_loss = L / (i + 1)
        PLCC = stats.pearsonr(y_pred, y_test)[0]
        SROCC = stats.spearmanr(y_pred, y_test)[0]
        RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())
        KROCC = stats.stats.kendalltau(y_pred, y_test)[0]
        print("Test results: test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
              .format(test_loss, SROCC, KROCC, PLCC, RMSE))
        np.save(save_result_file, (y_pred, y_test, test_loss, SROCC, KROCC, PLCC, RMSE, test_index))
