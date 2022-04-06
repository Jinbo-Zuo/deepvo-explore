# deepvo-explore

# 1. 运行程序
- 首先下在当前目录下新建文件夹 "/KITTI"
  - 下载KITTI图像数据集保存在" /KITTI/images" 下
  - 下载KITTI数据集对应的位姿文件，保存在 "/KITTI/pose" 下
    - (注：本项目中使用的位姿文件经过预处理，为6维储存欧拉角+平移 和 9维储存旋转矩阵R)

- 在 "parameters.py" 中修改路径与参数
  - 修改 "self.data_path" 为KITTI数据集保存路径
  - 根据性能和需求修改参数（包括Training parameters, Model parameters 以及 Image parameters）
    - batch size 和 图像尺寸 (img_w, img_h) 应根据显卡性能选择
    - pin_mem 应根据内存 RAM 大小选择
    - resume_train 决定基于之前的模型继续训练 or 从零训练

- 运行 "main.py"

# 2. 生成位姿数据
- 运行 "generate_pose.py"
  - 结果保存在 "/result" 下
  - 位姿文件可以基于 evo 等工具进行可视化

# 3. 其他文件说明
- data_operate.py : 定义 PyTorch dataset 和 sampler
- parameters.py : 参数保存
- model.py : 模型定义

# 4. 用到的包
- numpy
- pandas
- pytorch
- os
- torchvision

