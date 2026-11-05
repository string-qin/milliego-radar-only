# milliEgo 远程验证报告（2026-11-02）

## 当前结论
符合要求的 uv + PyTorch GPU 环境已经实际安装并通过训练算子测试。官方仓库是 TensorFlow/Keras 实现，尚未完成完整 PyTorch radar-only 训练与评测程序的迁移，因此不能将“环境验证成功”称作“论文复现完成”。

工作目录：/home/metaiot_guest/Workspace/qinzx/milliEgo
本轮代码、虚拟环境、下载、缓存、临时文件和报告均保存在此目录。原始数据保持只读。未运行正式训练，未修改 GPU 上的其他任务。

## 环境与 GPU 实测
- Python 3.10.12；uv 0.10.8。
- PyTorch 2.7.1+cu126；wheel 自带 CUDA 12.6 运行时；cuDNN 9.5.1。
- NumPy 1.26.4、SciPy 1.15.3、h5py 3.16.0，以及 PyYAML、Matplotlib、Pillow、tqdm、TensorBoard 已安装。
- uv pip check 通过；uv.lock 和 requirements.lock.txt 已生成。
- GPU 3（NVIDIA L40S）完成矩阵乘法反向传播，以及 9 层雷达 CNN + 通道注意力 + 两层 512 单元 LSTM + 位姿头的 3 次优化器更新。
- 使用真实 ColoRadar 点云转换的图像作为输入，使用合成零回归目标。该测试只验证计算链路，不评估模型精度。
- 输入 [1,3,6,64,256]，输出 [1,3,6]；单帧对 T=1 也通过。
- 梯度有限、参数确实更新；保存并重载后输出最大误差 0。
- 峰值已分配张量显存约 421.7 MiB；非正式训练显存估算。
- DataLoader 两个子进程与 HDF5 读写往返检查通过。
- 证据：reports/gpu_validation.json、reports/uv_sync_20261102.log。

## 官方源码与预训练权重
- 官方源码保存在 upstream，提交 9b163b92f871b7912dc3a27969b1761556cddad2。
- 官方 cnn.h5 下载成功，共 99,196,456 字节。
- SHA256：2e6e3152da902ea7c9d6b4d3b717f0240dfa87a01503039780d9eb0aa2dedab7。
- 已将实际使用的九层卷积从 HWIO 转换到 OIHW，保存为 weights/cnn_torch.pt；九层张量转置往返均完全相同。
- 原 cnn.h5 另有 conv6_1 和 fc_pose，官方 milliEgo 主干不使用这两层，本次也未导入。
- 首层 PyTorch 卷积与独立 NumPy 参考计算最大绝对误差约 1.52e-6。
- 加载转换权重后的完整诊断网络 GPU 前向输出有限。
- 这不是整个 TensorFlow/PyTorch 网络的数值等价认证；尚未验证 LSTM 权重、训练过程、原始数据预处理的完整一致性。
- 证据：reports/pretrained_download.json、reports/pretrained_conversion.json。

## 数据集适配检查
本轮完成 82 个独立序列/场景检查：ColoRadar 52、K-Radar 23、RaDelft 7。
“检查通过”表示规定范围内的读取和元数据检查成功，不表示每个原始文件都完整、不表示所有外参和同步已认证，也不表示无需数据适配。
最初有一次无超时审查因网络存储等待被终止；随后使用逐序列超时检查完成。RaDelft 检查中发现 Scene2–7 缺少可选 radar.topic.json，脚本已明确记录该缺口并继续检查其他可用数据，未补造时间戳。

### ColoRadar：最适合优先推进
- 52 序列共有 85,977 个单芯片雷达点云时间戳；点云索引检查未发现缺失。
- 全部雷达/GT 时间戳严格递增，位姿数值有限，四元数范数误差最大约 7.64e-13。
- 按“落在 GT 时间范围内、GT 插值间隔不超过 0.2 秒、相邻雷达间隔不超过 0.2 秒”的初筛规则，得到 72,979 个候选帧、72,771 对候选相邻帧。
- 这些是时间覆盖候选，尚未执行最终按序列划分、运动质量筛选和雷达坐标监督认证。
- 分别抽样 lab、aspen、outdoors 三条序列的实际 float32 N×5 点云，均能解码并转换为 64×256 距离图；另取 lab 四帧用于 GPU 诊断。
- 使用现成点云可绕过 ADC 信号处理；点云输入路线与原方法更接近。
- 已找到 base_to_single_chip 等外参文件；仍须核对变换方向、GT 参考坐标，随后把相对位姿转换至雷达系。当前生成的 BASE-frame 标签只作管线诊断，不可直接当作已验证的雷达监督。
- 当前诊断投影使用方位 ±60°、俯仰 ±30°、最大平面距离 30 m、最近点覆盖、[0,1] 归一化。这些参数尚不是原始论文数据预处理的等价声明。

### K-Radar：可以适配，预处理和监督约定仍需落实
- 当前存储包含 23 条序列、17,307 个雷达张量文件。
- info_label 与官方里程计 pose 各 14,744 条，逐序列数量匹配。
- 每序列检查首、中、末三个标签头的雷达索引，引用的张量均存在；旋转矩阵正交性最大误差约 5.13e-7。
- 实际读取 seq1/tesseract_00033.mat 的 arrDREA，完成 Doppler 聚合、诊断峰值提取和距离图转换。
- 原始输入是功率张量，不是 milliEgo 的现成距离图，需要制定并固定点云提取规则；本次峰值方案仅为可行性诊断，未声称是最终 CFAR。
- 必须依据标签头关联雷达帧；不能按相同编号或简单逐行配对。
- 标签时间是 LiDAR/label 参考时间，尚未独立认证为原生雷达采集时间。
- 官方 pose 为 LeGO-LOAM 参考轨迹；完整雷达到参考轨迹坐标的外参仍须验证。
- 纯雷达指模型输入只含雷达；其他传感器生成的参考轨迹可作为明确披露的监督/评测标签，不能混入模型输入。

### RaDelft：雷达可读，但当前位姿不足以支持完整 6DoF 监督声明
- 七个场景共有 16,975 个 radar_ososos 点云文件。
- 每场景实际解码首、中、末三个 NPY 点云，共 21 帧，均为可读 float32 N×4，并成功转换为诊断距离图。
- 按导出 NPY 文件名时间、gps_odom 时间范围和最大 0.1 秒插值间隔，16,855 帧有候选时间覆盖；这不是已验证的物理同步。
- 对全部已读 gps_odom 消息，z 恒为 0，四元数 x/y 恒为 0，实际只有平面位置和 yaw 信息；不能当作完整六自由度真值。
- Scene1 导出点云首帧文件名时间比 radar.topic.json 首消息时间晚 17.5 秒；其他场景缺少该 JSON，无法按同样方式交叉核对。不能盲目给所有数据统一增加偏移。
- 各场景抽查前 2,000 条 TF 消息仅见 odom/base_link/map 的非空矩阵；未核验到雷达到 an_device 的完整静态外参链。
- radar_gt NPY 是网格标签，不是里程计位姿。
- 可考虑在完成时间和外参认证后做明确标注的平面实验；若要求完整 6DoF，需取得可靠的完整位姿监督。
- 这些问题是当前服务器导出数据的发现，不代表原始数据集永远无法支持更完整的监督。

## 建议推进顺序
1. 先完成 ColoRadar radar-only 的坐标、投影、标签和完整训练/轨迹评估迁移。
2. 按序列冻结 train/val/test，明确帧间隔、旋转顺序/单位、ATE/RPE 与轨迹对齐规则。
3. 使用转换后的 CNN 权重，并记录冻结策略；不能把随机初始化结果与原论文预训练设置混为一谈。
4. 完成小样本过拟合与端到端轨迹检查，再启动正式 GPU 训练。
5. 再增加 K-Radar；RaDelft 的平面监督、时钟和外参问题单独解决。

## 复查命令
在服务器 bash 中运行：
```bash
cd /home/metaiot_guest/Workspace/qinzx/milliEgo
source env.sh
uv sync --locked --python /usr/bin/python3.10
uv pip check --python .venv/bin/python
CUDA_VISIBLE_DEVICES=3 uv run --locked python scripts/check_gpu.py
CUDA_VISIBLE_DEVICES=3 uv run --locked python scripts/check_pretrained.py
```
GPU 3 仅是本轮短时诊断使用的设备，不表示长期独占或已分配正式训练资源。
数据审查入口为 scripts/run_dataset_audit.py，已通过记录会复用；如需全量重新核验，应保留旧报告后明确建立新一轮检查。

## 主要证据
- pyproject.toml、uv.lock、requirements.lock.txt、env.sh
- reports/gpu_validation.json
- reports/pretrained_conversion.json
- reports/dataset_summary.json
- reports/dataset_audit_index.json
- reports/dataset_parts/*.json
- scripts/check_gpu.py、scripts/check_pretrained.py、scripts/audit_dataset_worker.py

## 参考资料
- https://github.com/ChristopherLu/milliEgo
- https://arxiv.org/abs/2006.02266
- https://docs.astral.sh/uv/guides/integration/pytorch/
- https://pytorch.org/get-started/previous-versions/
- https://github.com/kaist-avelab/K-Radar/blob/main/docs/odometry.md
- https://github.com/RaDelft/RaDelft-Dataset

