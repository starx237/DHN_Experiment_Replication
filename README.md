# DHN_Experiment_Replication

## 一、本地复现流程

### 1.1 环境准备

#### 1.1.1 安装 Miniconda（如未安装）

从 https://docs.conda.io/en/latest/miniconda.html 下载并安装 Miniconda（Python 3.10+ 版本）。

#### 1.1.2 创建 Conda 环境

```bash
conda create -n DHN python=3.10
conda activate DHN

# 安装 PyTorch
# CUDA 13.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
# 或南大镜像源
pip install torch torchvision torchaudio --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu130
# 或 CPU 版本
pip install torch torchvision torchaudio

# 安装其他依赖
pip install numpy scipy matplotlib absl-py ml-collections tensorboard
```

### 1.2 数据集准备

可以选择 **（A）直接下载预处理好的数据** 或 **（B）自行生成数据**。

#### 方式 A：下载预处理数据

1. 从 [Google Drive](https://drive.google.com/file/d/1ry6Lz1Wa77UZmApEghfm9aSydXU2vWWr/view?usp=sharing) 下载数据压缩包
2. 解压到项目根目录下的 `data/` 文件夹，确保结构如下：

```
data/
 ├── single_pendulum/
 │     ├── train/
 │     │     └── data.pkl
 │     └── test/
 │           └── data.pkl
 └── double_pendulum/
       ├── train/
       │     └── data.pkl
       └── test/
             └── data.pkl
```

#### 方式 B：自行生成数据

运行以下命令生成训练和测试数据。

**生成训练数据（1000 条轨迹）**：

```bash
# 单摆
DATA_NAME=single_pendulum bash scripts/data_gen_train.sh

# 双摆（修改脚本中的变量或直接运行）
DATA_NAME=double_pendulum bash scripts/data_gen_train.sh
```

**生成测试数据（200 条轨迹）**：

```bash
# 单摆
DATA_NAME=single_pendulum bash scripts/data_gen_test.sh

# 双摆
DATA_NAME=double_pendulum bash scripts/data_gen_test.sh
```

生成的 `.pkl` 文件会保存在 `data/<system>/train/data.pkl` 和 `data/<system>/test/data.pkl`。可视化文件保存在 `data/<system>/train/vis/` 和 `data/<system>/test/vis/`。

> **注意**：数据生成使用不同的随机种子（训练 `seed=0`，测试 `seed=1`），以确保训练集和测试集的物理参数（摆长等）分布在统计上相似但不完全相同。

### 1.3 实验一：正向模拟

正向模拟的目标是：训练模型学会已知轨迹的动力学，然后从初始状态自回归地生成完整未来轨迹。

#### 步骤 1：训练模型

编辑 [`scripts/train.sh`](scripts/train.sh)，取消注释想要运行的实验名称，例如：

```bash
EXP_NAME=sinpend_kernel2_stride1  # 单摆，block_size=2, stride=1
```

然后运行：

```bash
bash scripts/train.sh
```

训练过程会：
- 在 `results/sinpend_kernel2_stride1/` 下保存 checkpoint 和 TensorBoard 日志
- 每 50 个 epoch 保存一个永久 checkpoint（`checkpoint_ep50.pth`、`checkpoint_ep100.pth`...）
- 每个 epoch 保存临时 checkpoint（`checkpoint.pth`）

查看训练曲线：

```bash
tensorboard --logdir results/sinpend_kernel2_stride1
```

#### 步骤 2：生成轨迹

编辑 [`scripts/generate.sh`](scripts/generate.sh)，设置相同的 `EXP_NAME`：

```bash
EXP_NAME=sinpend_kernel2_stride1
```

然后运行：

```bash
bash scripts/generate.sh
```

生成结果保存在 `results/sinpend_kernel2_stride1/gen_sequence/` 下，按生成配置分子目录（如 `denoise10_update1/`、`denoise5_update1/`、`denoise1_update1/`）。每个子目录包含：

- `result_dict.npy`：预测轨迹数据（`q_pred`、`p_pred`、`q_gt`、`p_gt`、`t` 等）
- `cond_dict.npy`：物理条件参数

### 1.4 实验二：部分轨迹补全

部分轨迹补全模拟了一个更实际的场景：面对一个**未见过的物理系统**（新参数），仅观测到前几步轨迹，需要推断系统参数并预测未来。

#### 步骤 1：训练模型（同实验一）

首先完成正向模拟中的步骤 1（模型训练）。

#### 步骤 2：提取潜在编码

编辑 [`scripts/extract_partial.sh`](scripts/extract_partial.sh)，设置相同的 `EXP_NAME`：

```bash
EXP_NAME=sinpend_kernel2_stride1
```

然后运行：

```bash
bash scripts/extract_partial.sh
```

这一步会：
- 加载已训练好的模型权重（冻结所有网络参数）
- 仅优化 codebook 嵌入向量，为测试集中的每条新轨迹学习一个最优的潜在编码
- 结果保存在 `results/sinpend_kernel2_stride1/extract/`

> **注意**：extract 阶段使用测试集数据，且在训练时只取轨迹的前 16 步（`crop_interval=0,16`），模拟"仅观测到前几步"的场景。codebook 大小缩减为 200（匹配测试集轨迹数量）。

#### 步骤 3：生成补全轨迹

编辑 [`scripts/generate_partial.sh`](scripts/generate_partial.sh)，设置相同的 `EXP_NAME`：

```bash
EXP_NAME=sinpend_kernel2_stride1
```

然后运行：

```bash
bash scripts/generate_partial.sh
```

生成结果保存在 `results/sinpend_kernel2_stride1/extract/gen_sequence/` 下。

### 1.5 不同配置的实验

以下列出所有可用的实验配置及其预期用途：

| EXP_NAME | block_size | block_step | 物理场景 |
|----------|------------|------------|----------|
| `sinpend_kernel2_stride1` | 2 | 1 | 单摆 |
| `sinpend_kernel4_stride2` | 4 | 2 | 单摆 |
| `sinpend_kernel8_stride4` | 8 | 4 | 单摆 |
| `doupend_kernel2_stride1` | 2 | 1 | 双摆 |
| `doupend_kernel4_stride2` | 4 | 2 | 双摆 |
| `doupend_kernel8_stride4` | 8 | 4 | 双摆 |

更大的块大小会增大训练难度，但可以让模型具有更大的时间感受野/上下文窗口，有利于降低长时程相位误差。

### 1.6 查看结果

#### TensorBoard 可视化

训练和提取阶段的 loss 曲线、轨迹可视化图像均记录在 TensorBoard 中：

```bash
tensorboard --logdir results/
```

#### 生成的 `.npy` 文件

生成的结果文件可用 NumPy 加载和分析：

```python
import numpy as np

# 加载生成结果
data = np.load('results/sinpend_kernel2_stride1/gen_sequence/denoise10_update1/result_dict.npy', allow_pickle=True).item()

print(data.keys())          # dict_keys(['t', 'q_gt', 'p_gt', 'q_pred', 'p_pred', 'p_scale'])
print(data['q_pred'].shape) # (num_trajectories, num_steps, q_dim)

# 计算预测误差
import numpy as np
mse = np.mean((data['q_pred'] - data['q_gt']) ** 2)
print(f'Q prediction MSE: {mse:.6f}')
```

## 二、复现结果
