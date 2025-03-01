from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import psutil  # 用于获取 CPU 内存
import os
import pandas as pd
warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        self.args=args
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def manual_mse_loss(self, predictions, targets, tarpre):
        # 计算预测值与目标值之间的差异
        diff = predictions - targets
        diff1 = targets-tarpre
        diff2 = predictions-tarpre
        # 计算平方差
        squared_diff = diff ** 2
        squared_diff1 = diff1 ** 2
        squared_diff2 = diff2 ** 2

        # 计算均方误差
        mse = squared_diff.mean()
        # mse = mse + squared_diff2.mean()
        # mse = mse + squared_diff1.mean()
        mse = mse + squared_diff1.mean() + squared_diff2.mean()
        return mse

    # L2 正则化函数
    def l2_regularization(self,model, lambda_l2):
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        return lambda_l2 * l2_norm

    def get_memory_usage(self):
        # 获取 CPU 内存占用
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 ** 2)  # 返回内存占用（MB）

    def get_gpu_memory_usage(self):
        # 获取当前 GPU 显存占用
        return torch.cuda.memory_allocated() / (1024 ** 2)  # 返回显存占用（MB）
    def vali(self, vali_data, vali_loader, criterion):

        total_loss = []
        lambda_l2=0.000015
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                # print(batch_x)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs,y_out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_y)[0]
                        else:
                            outputs,y_out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_y)
                else:
                    if self.args.output_attention:
                        outputs,y_out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_y)[0]
                    else:
                        outputs,y_out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_y)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                y_out = y_out[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                y_out = y_out.detach().cpu()

                loss = criterion(pred, true)
                # loss = self.manual_mse_loss(pred, true,y_out)
                # l2_loss = self.l2_regularization(self.model, lambda_l2)
                # loss = criterion(pred, true) + l2_loss

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):


        train_data, train_loader = self._get_data(flag='train')

        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')


        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        lambda_l2 = 0.000015

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):

            iter_count = 0
            train_loss = []

            # 记录开始的内存占用
            start_memory = self.get_memory_usage()
            start_gpu = self.get_gpu_memory_usage()

            self.model.train()
            epoch_time = time.time()


            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:#false
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs,y_out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_y)[0]
                        else:
                            outputs,y_out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_y)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        y_out = y_out[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        # loss = self.manual_mse_loss(outputs,batch_y,y_out)
                        # l2_loss = self.l2_regularization(self.model, lambda_l2)
                        # loss = criterion(outputs, batch_y)+l2_loss
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:#false
                        outputs,y_out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_y)[0]
                    else:
                        outputs,y_out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_y)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    y_out = y_out[:, -self.args.pred_len:, f_dim:].to(self.device)
                    # l2_loss = self.l2_regularization(self.model, lambda_l2)
                    # loss = criterion(outputs, batch_y) + l2_loss
                    loss = criterion(outputs, batch_y)
                    # loss = self.manual_mse_loss(outputs, batch_y, y_out)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            # epoch_time = time.time() - start_time
            end_memory = self.get_memory_usage()
            end_gpu = self.get_gpu_memory_usage()
            memory_used = end_memory - start_memory
            gpu_used = end_gpu - start_gpu

            print("Epoch: {} cost time: {} memory_used: {} memory_gpu:{}".format(epoch + 1,
                                                                                         time.time() - epoch_time,
                                                                                         memory_used, gpu_used))
            train_loss = np.average(train_loss)
            print("==========")
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            print("-------------")
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    # def test(self, setting, test=0):
    #     test_data, test_loader = self._get_data(flag='test')
    #     if test:
    #         print('loading model')
    #         self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
    #
    #     preds = []
    #     trues = []
    #     folder_path = './test_results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    #
    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float().to(self.device)
    #
    #             if 'PEMS' in self.args.data or 'Solar' in self.args.data:
    #                 batch_x_mark = None
    #                 batch_y_mark = None
    #             else:
    #                 batch_x_mark = batch_x_mark.float().to(self.device)
    #                 batch_y_mark = batch_y_mark.float().to(self.device)
    #
    #             # decoder input
    #             dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
    #             dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
    #             # encoder - decoder
    #             if self.args.use_amp:
    #                 with torch.cuda.amp.autocast():
    #                     if self.args.output_attention:
    #                         outputs,y_out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_y)[0]
    #                     else:
    #                         outputs,y_out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_y)
    #             else:
    #                 if self.args.output_attention:
    #                     outputs,y_out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_y)[0]
    #
    #                 else:
    #                     outputs,y_out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_y)
    #
    #             f_dim = -1 if self.args.features == 'MS' else 0
    #             outputs = outputs[:, -self.args.pred_len:, f_dim:]
    #             batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
    #             y_out = y_out[:, -self.args.pred_len:, f_dim:].to(self.device)
    #             outputs = outputs.detach().cpu().numpy()
    #             batch_y = batch_y.detach().cpu().numpy()
    #             y_out = y_out.detach().cpu().numpy()
    #             if test_data.scale and self.args.inverse:
    #                 shape = outputs.shape
    #                 outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
    #                 batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
    #
    #             pred = outputs
    #             true = batch_y
    #
    #             preds.append(pred)
    #             trues.append(true)
    #             if i % 20 == 0:
    #                 input = batch_x.detach().cpu().numpy()
    #                 if test_data.scale and self.args.inverse:
    #                     shape = input.shape
    #                     input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
    #                 gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
    #                 pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
    #                 visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
    #
    #     preds = np.array(preds)
    #     trues = np.array(trues)
    #     print('test shape:', preds.shape, trues.shape)
    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    #     trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    #     print('test shape:', preds.shape, trues.shape)
    #
    #     # result save
    #     folder_path = './results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    #
    #     mae, mse, rmse, mape, mspe = metric(preds, trues)
    #     print('mse:{}, mae:{}'.format(mse, mae))
    #     f = open("result_long_term_forecast.txt", 'a')
    #     f.write(setting + "  \n")
    #     f.write('mse:{}, mae:{}'.format(mse, mae))
    #     f.write('\n')
    #     f.write('\n')
    #     f.close()
    #
    #     np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
    #     np.save(folder_path + 'pred.npy', preds)
    #     np.save(folder_path + 'true.npy', trues)
    #
    #     return


    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results_hunhe/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs, y_out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)[0]
                        else:
                            outputs, y_out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                else:
                    if self.args.output_attention:
                        outputs, y_out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)[0]
                    else:
                        outputs, y_out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                y_out = y_out[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                y_out = y_out.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd1 = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd1, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)



        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        with open("result_long_term_forecast.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write(f'mse:{mse}, mae:{mae}\n\n')

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_y)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_y)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_y)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_y)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return