
import os

class Parameters():
    def __init__(self):

        # Training parameters
        self.epochs = 100
        self.batch_size = 6
        self.dropout = 0.5
        self.lr = 0.001
        self.resume_train = False
        self.pin_mem = True
        self.n_processors = 8

        # Model parameters
        self.rnn_hidden_size = 1000
        self.rnn_dropout_out = 0.5
        self.rnn_dropout_between = 0  # 0: no dropout
        self.batch_norm = True

        # Path
        self.data_path = 'D:/7.file/6.python_code/3.deepvo/KITTI'
        # self.data_path = 'D:/7.file/6.python_code/3.deepvo/EuRoc'
        self.image_path = self.data_path + '/images/'
        self.pose_path = self.data_path + '/pose/'
        self.save_model_path = 'models/deepvo.model'
        self.save_optimzer_path = 'models/deepvo.optimizer'
        self.load_model_path = 'models/deepvo.model.train'
        self.load_optimizer_path = 'models/deepvo.optimizer'
        self.train_part = ['00', '01', '02', '03']
        self.valid_part = ['04', '05', '06']

        # Data info path
        self.train_data_info_path = 'datainfo/deepvo_train.pickle'
        self.valid_data_info_path = 'datainfo/deepvo_valid.pickle'

        # Image parameters
        self.img_w = 511  # 1226 / 608 / 1241 / 408
        self.img_h = 154  # 370 / 184 / 376 / 123
        self.seq_len = (5, 7)
        self.sample_times = 3
        self.img_means = (-0.15181537060486192, -0.13357731187006083, -0.14182149327187585)
        self.img_stds = (0.31740946637377987, 0.3198330615266129, 0.3237293402692505)
        self.resize_mode = 'rescale'  # 'crop' / 'rescale' / None

        if not os.path.isdir(os.path.dirname(self.save_model_path)):
            os.makedirs(os.path.dirname(self.save_model_path))
        if not os.path.isdir(os.path.dirname(self.save_optimzer_path)):
            os.makedirs(os.path.dirname(self.save_optimzer_path))
        if not os.path.isdir(os.path.dirname(self.train_data_info_path)):
            os.makedirs(os.path.dirname(self.train_data_info_path))

par = Parameters()