import numpy as np
import scipy.io
from torch.autograd import Variable
import torch
import time
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce
from eegnet_model import *
from atcnet_model import *
from ot_pseudolabel import relation_OT

class ExperimentProcessor:
    def __init__(self, subject_number):
        super(ExperimentProcessor, self).__init__()
        self.dataset = 'hgd'
        self.class_num = 2
        self.channel_num = 3
        self.lr = 0.001
        self.epoch_number = 60
        self.batch_size = 288
        self.test_batch_size = 288
        self.subject_number = subject_number
        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor
        self.reg_1=0.01
        self.reg_2=1.
        self.criterion_cls = torch.nn.CrossEntropyLoss(reduction='none').cuda()
        if self.dataset == '2a':
            self.class_num = 4
            self.channel_num = 22
            self.root = './datasets/2a/'
        elif self.dataset == '2b':
            self.class_num = 2
            self.channel_num = 3
            self.root = './datasets/2b/B0'
        elif self.dataset == 'hgd':
            self.class_num = 4
            self.channel_num = 45
            self.root = './datasets/hgd/'
        self.net = ATCNetModule(self.class_num, self.channel_num)
        self.net.load_state_dict(torch.load('./ckpts/hgd/{}_best.pth'.format(self.subject_number))) # ablation study
        self.net.cuda()

        self.model_name = type(self.net).__name__
        # self.net.load_state_dict(torch.load('./saved_model/{}/{}/model_without{}.pth'.format(self.dataset,self.model_name, self.subject_number)))

    def get_train_test_data(self):

        # train data
        train_data = []
        train_label = []
        
        # import pdb
        # pdb.set_trace()
        for i in range(1, 15):
            if i == self.subject_number:
                continue
            total_data = scipy.io.loadmat(self.root + '%d.mat' % i)
            train_data.append(total_data['traindata'])
            train_label.append(total_data['trainlabel'])
        train_data = np.concatenate(train_data, axis=0)
        train_label = np.concatenate(train_label, axis=0)

        train_data = np.expand_dims(train_data, axis=1)
        train_label = np.transpose(train_label)
        train_label = train_label[0]
        # test data
        test_tmp = scipy.io.loadmat(self.root + '%d.mat' % self.subject_number)
        test_data = test_tmp['testdata']
        test_label = test_tmp['testlabel']
        test_data = np.expand_dims(test_data, axis=1)
        test_label = np.transpose(test_label)
        test_label = test_label[0]

        target_mean = np.mean(train_data)
        target_std = np.std(train_data)
        train_data = (train_data - target_mean) / target_std
        test_data = (test_data - target_mean) / target_std

        return train_data, train_label, test_data, test_label

    def train(self):
        train_data, train_label, test_data, test_label = self.get_train_test_data()

        train_data = torch.from_numpy(train_data)
        train_label = torch.from_numpy(train_label)
        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label)
        train_data = Variable(train_data.type(self.Tensor))
        train_label = Variable(train_label.type(self.LongTensor))
        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))
        train_dataset = TensorDataset(train_data, train_label)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        test_dataset = TensorDataset(test_data, test_label)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)
        # Optimizers
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)

        best_accuracy = 0
        average_accuracy = 0

        for epoch in range(self.epoch_number):
            torch.cuda.synchronize()
            epoch_start = time.time()
            self.net.eval()
            current_ration = 0.5 + 0.5 / self.epoch_number * (epoch + 1)
            feats, outputs = self.net(test_data)
            t_predict, conf, selected_mask = relation_OT(feats.detach(), outputs.detach() / 5, ratio=current_ration, reg_1=self.reg_1, reg2=self.reg_2)
            t_predict = t_predict[selected_mask]
            conf = conf[selected_mask]
            pseudo_acc = float((t_predict == test_label[selected_mask]).cpu().numpy().astype(int).sum()) / float(test_label[selected_mask].size(0))
            print('epoch:{}, subject number:{}, pseudo_label_accu:{}'.format(epoch, self.subject_number, pseudo_acc))
            adapt_label = t_predict.cpu().type(self.LongTensor)
            conf = Variable(conf).cpu()
            adapt_label_dataset = TensorDataset(test_data[selected_mask], adapt_label, conf)
            adapt_loader = DataLoader(dataset=adapt_label_dataset, batch_size=test_data[selected_mask].size(0) // 32, shuffle=True)
            # adapt_loader = DataLoader(dataset=adapt_label_dataset, batch_size=9, shuffle=True)

            self.net.train()
            train_times = 0
            train_acc = 0
            train_loss = 0
            for _ in range(10):
                for i, (t_data, t_label, t_conf) in enumerate(adapt_loader):
                    outputs = self.net(t_data)
                    # conf = torch.ones_like(conf) # ablation study
                    loss_ada = self.criterion_cls(outputs, t_label) * t_conf.cuda()
                    
                    try:
                        src_data, src_label = next(train_loader)
                    except:
                        train_loader = iter(DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True))
                        src_data, src_label = next(train_loader)
                    outputs_src = self.net(src_data)
                    loss_src = self.criterion_cls(outputs_src, src_label)
                    loss = loss_ada.mean() + loss_src.mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    t_predict = torch.max(outputs, 1)[1]
                    train_acc += float((t_predict == t_label).cpu().numpy().astype(int).sum()) / float(t_label.size(0))
                    train_loss += loss.item()
                    train_times += 1
            train_loss = train_loss / train_times
            train_acc = train_acc / train_times
            torch.cuda.synchronize()
            epoch_end = time.time()
            epoch_elapsed = epoch_end - epoch_start
            epoch_times.append(epoch_elapsed)
            print(f"Epoch {epoch} elapsed time: {epoch_elapsed:.3f} seconds")
            # test
            self.net.eval()
            test_times = 0
            test_acc = 0
            test_loss = 0
            for i, (t_data, t_label) in enumerate(test_loader):
                feats, outputs = self.net(t_data)
                t_predict, conf, _ = relation_OT(feats.detach(), outputs.detach() / 10, ratio=1.0, reg_1=self.reg_1, reg2=self.reg_2)
                t_predict = torch.max(outputs, 1)[1]
                test_acc += float((t_predict == t_label).cpu().numpy().astype(int).sum()) / float(t_label.size(0))
                test_times += 1

            test_loss = test_loss / test_times
            test_acc = test_acc / test_times
            print("subject: %d, epoch: %d, train loss: %.5f, train accuracy: %.3f,"
                 "test accuracy is %.3f" % (self.subject_number, epoch, train_loss, train_acc, test_acc))
            average_accuracy += test_acc
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                torch.save(self.net.state_dict(), './ckpts/finetune_{}_{}_best.pth'.format(self.dataset,self.subject_number))

        average_accuracy = average_accuracy / self.epoch_number
        return best_accuracy, average_accuracy


def main():
    result = []
    for i in range(1, 15):
        exp = ExperimentProcessor(i)
        best_accuracy, average_accuracy = exp.train()
        result.append(best_accuracy)
    print(result)
    print(sum(result) / len(result))
    with open("1.txt", "w") as file:
        file.write(f"Result: {result}\n")
        file.write(f"Average: {sum(result) / len(result)}\n")


if __name__ == "__main__":
    main()
