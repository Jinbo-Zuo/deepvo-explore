
import os
import pandas as pd
import torch
from parameters import par
from data_operate import get_data_info, SortedRandomBatchSampler, ImageSequenceDataset
from model import DeepVO
from torch.utils.data import DataLoader

if __name__ == '__main__':

    # Prepare Data
    if os.path.isfile(par.train_data_info_path) and os.path.isfile(par.valid_data_info_path):
        train_df = pd.read_pickle(par.train_data_info_path)
        valid_df = pd.read_pickle(par.valid_data_info_path)
    else:
        train_df = get_data_info(par.train_part, par.seq_len, 1, par.sample_times)
        valid_df = get_data_info(par.valid_part, par.seq_len, 1, par.sample_times)
        train_df.to_pickle(par.train_data_info_path)
        valid_df.to_pickle(par.valid_data_info_path)

    train_sampler = SortedRandomBatchSampler(train_df, par.batch_size, True)
    train_dataset = ImageSequenceDataset(train_df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds)
    train_dl = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=par.n_processors, pin_memory=par.pin_mem)

    valid_sampler = SortedRandomBatchSampler(valid_df, par.batch_size, True)
    valid_dataset = ImageSequenceDataset(valid_df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds)
    valid_dl = DataLoader(valid_dataset, batch_sampler=valid_sampler, num_workers=par.n_processors, pin_memory=par.pin_mem)

    M_deepvo = DeepVO(par.img_h, par.img_w, par.batch_norm)
    M_deepvo = M_deepvo.cuda()

    # Create optimizer
    # optimizer = torch.optim.Adam(M_deepvo.parameters(), lr=0.001, betas=(0.9, 0.999))
    optimizer = torch.optim.Adagrad(M_deepvo.parameters(), lr=par.lr)

    # 如果是接着之前的模型继续训练
    if par.resume_train:
        M_deepvo.load_state_dict(torch.load(par.load_model_path))
        optimizer.load_state_dict(torch.load(par.load_optimizer_path))

    # Train
    min_loss_t = 1e10
    min_loss_v = 1e10

    for ep in range(par.epochs):
        print('--' * 50)

        # Train
        M_deepvo.train()
        loss_mean = 0
        t_loss_list = []

        for _, t_x, t_y in train_dl:
            t_x = t_x.cuda(non_blocking=par.pin_mem)
            t_y = t_y.cuda(non_blocking=par.pin_mem)
            ls = M_deepvo.step(t_x, t_y, optimizer).data.cpu().numpy()
            t_loss_list.append(float(ls))
            loss_mean += float(ls)

        loss_mean /= len(train_dl)

        # Validation
        M_deepvo.eval()
        loss_mean_valid = 0
        v_loss_list = []

        for _, v_x, v_y in valid_dl:
            v_x = v_x.cuda(non_blocking=par.pin_mem)
            v_y = v_y.cuda(non_blocking=par.pin_mem)
            v_ls = M_deepvo.get_loss(v_x, v_y).data.cpu().numpy()
            v_loss_list.append(float(v_ls))
            loss_mean_valid += float(v_ls)

        loss_mean_valid /= len(valid_dl)

        # Save model
        if loss_mean_valid < min_loss_v:
            min_loss_v = loss_mean_valid
            print('Save model at ep {}, mean of valid loss: {}'.format(ep + 1, loss_mean_valid))
            torch.save(M_deepvo.state_dict(), par.save_model_path + '.valid')
            torch.save(optimizer.state_dict(), par.save_optimzer_path + '.valid')

        if loss_mean < min_loss_t:
            min_loss_t = loss_mean
            print('Save model at ep {}, mean of train loss: {}'.format(ep + 1, loss_mean))
            torch.save(M_deepvo.state_dict(), par.save_model_path+'.train')
            torch.save(optimizer.state_dict(), par.save_optimzer_path+'.train')

        save_error_path = 'error/error.txt'
        f = open(save_error_path, 'a')
        f.write(str(loss_mean_valid))
        f.write(',')
        f.write(str(loss_mean))
        f.write('\n')
        f.close()
