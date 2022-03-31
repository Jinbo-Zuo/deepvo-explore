
from parameters import par
from model import DeepVO
import torch
from data_operate import get_data_info2, ImageSequenceDataset
from torch.utils.data import DataLoader
import numpy as np
from helper import eulerAnglesToRotationMatrix

if __name__ == '__main__':

    # path
    load_model_path = par.load_model_path
    save_path = 'result/'
    videos_to_test = ['01', '04', '05', '06']

    M_deepvo = DeepVO(par.img_h, par.img_w, par.batch_norm)
    M_deepvo = M_deepvo.cuda()
    M_deepvo.load_state_dict(torch.load(load_model_path))

    seq_len = int((par.seq_len[0] + par.seq_len[1]) / 2)
    overlap = seq_len - 1

    for test_video in videos_to_test:
        df = get_data_info2([test_video], [seq_len, seq_len], overlap)
        df = df.loc[df.seq_len == seq_len]
        dataset = ImageSequenceDataset(df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds)
        df.to_csv('test_df.csv')
        dataloader = DataLoader(dataset, batch_size=par.batch_size, shuffle=False, num_workers=1)

        gt_pose = np.load('{}{}.npy'.format(par.pose_path, test_video))  # (n_images, 6)

        # 进入正题
        M_deepvo.eval()
        has_predict = False
        answer = [[0.0] * 6, ]
        n_batch = len(dataloader)

        for i, batch in enumerate(dataloader):
            _, x, y = batch
            x = x.cuda()
            y = y.cuda()
            batch_predict_pose = M_deepvo.forward(x)

            batch_predict_pose = batch_predict_pose.data.cpu().numpy()

            if i == 0:
                for pose in batch_predict_pose[0]:
                    for i in range(len(pose)):
                        pose[i] += answer[-1][i]
                    answer.append(pose.tolist())
                batch_predict_pose = batch_predict_pose[1:]

            for predict_pose_seq in batch_predict_pose:
                ang = eulerAnglesToRotationMatrix([0, answer[-1][0], 0])
                location = ang.dot(predict_pose_seq[-1][3:])
                predict_pose_seq[-1][3:] = location[:]

                last_pose = predict_pose_seq[-1]
                for i in range(len(last_pose)):
                    last_pose[i] += answer[-1][i]
                last_pose[0] = (last_pose[0] + np.pi) % (2 * np.pi) - np.pi
                answer.append(last_pose.tolist())

        # save
        with open('{}/result_{}.txt'.format(save_path, test_video), 'w') as f:
            for pose in answer:
                if type(pose) == list:
                    f.write(', '.join([str(p) for p in pose]))
                else:
                    f.write(str(pose))
                f.write('\n')

        gt_pose = np.load('{}{}.npy'.format(par.pose_path, test_video))
        loss = 0
        for t in range(len(gt_pose)):
            angle_loss = np.sum((answer[t][:3] - gt_pose[t, :3]) ** 2)
            translation_loss = np.sum((answer[t][3:] - gt_pose[t, 3:6]) ** 2)
            loss = (100 * angle_loss + translation_loss)
        loss /= len(gt_pose)
        print('Loss = ', loss)