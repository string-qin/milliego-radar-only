# ColoRadar 单序列 PyTorch radar-only 复现试验

日期：2026-11-02。工作目录：/home/metaiot_guest/Workspace/qinzx/milliEgo。

## 结论
单序列的数据准备、GPU训练、验证选模、测试轨迹评估和独立雷达推理已完成。小样本过拟合通过，但保留测试段性能较差，不能宣称复现了原论文精度。当前权重是流程验证基线。

## 数据与协议
- 序列：12_21_2020_arpg_lab_run0，输入仅 single_chip 点云，不输入 IMU。
- 每隔5个原生雷达帧采样，约0.5秒一对；234帧、230对有效数据。
- 按时间划分60/20/20；训练138对、验证46对、测试46对；跨边界样本丢弃，端点原始雷达帧互不重叠。
- 同序列时间留出不代表跨序列泛化；该序列在已有其他 baseline 开发中也曾使用。
- GT插值要求有效时间范围和不超过0.2秒的原始GT间隔。
- 使用官方工具中的 T_world_radar = T_world_base @ T_base_radar 约定，标签为 inverse(T_world_radar_a) @ T_world_radar_b。
- 标签为雷达系平移米、外禀xyz欧拉角度；标签矩阵与欧拉角往返误差小于1e-15。
- 点云投影64×256，方位±60°、俯仰±30°、平面距离<30米，同像素取最近点，反距离uint8 [0,255]；每帧复制3通道，两帧拼6通道。
- 数据集投影参数与源数据存在适配差异，不声称与原论文数据处理完全等价。

## 网络与训练
- 官方九层CNN预训练权重、雷达通道注意力、两层512单元LSTM、独立128/64/3平移与旋转头。
- 删除IMU及跨模态注意力，保持T=1、每帧对重置隐状态。
- 使用PyTorch实现Keras 2.1.6的hard-sigmoid门控、tanh状态、unit forget bias及分门输入dropout。
- CNN冻结并缓存输出，其他层仍通过GPU梯度更新；缓存只由雷达输入计算，无GT特征输入。
- 模型源码：milliego/model.py；训练：scripts/train_single_sequence.py。
- RMSprop rho=.9，eps=1e-8；MSE_translation_m + .001*MSE_rotation_degrees；batch32；输入dropout与头部dropout .25；梯度裁剪10。
- 两档初始学习率1e-5和1e-4，各200轮；每25轮乘.75；只用验证加权MSE选模。
- 验证选择了1e-5第1轮，之后训练拟合改善但验证未提升，表明当前训练设置没有取得可靠泛化。
- 16对训练样本的独立过拟合诊断运行400步，关闭dropout，诊断lr=1e-3；损失降低98.88%，诊断模型丢弃，不用于验证选模。
- 运行GPU 3（L40S），由uv管理环境；训练退出码0。
- 缓存后的两档训练及诊断很快，仅约12.2秒，不含前面的原始数据读取和准备。缓存冻结CNN避免每轮重复卷积，不表示全CNN端到端训练只需这一时间。

## 测试结果
46对，47个位姿，参考路径约23.37米。所有轨迹从单位阵起点积分。

|方法|逐对平移向量RMSE (m)|逐对旋转测地RMSE (deg)|仅初始位姿对齐ATE RMSE (m)|SE(3)刚体对齐ATE RMSE (m)|
|---|---:|---:|---:|---:|
|模型|0.4564|14.0691|8.1515|4.3671|
|零运动|0.5315|14.0700|11.0568|6.2965|
|训练标签均值|0.4251|14.7373|10.6981|2.5270|

模型优于零运动；逐对平移和刚体对齐ATE未超过训练均值基线。旋转预测接近零运动，尚未学到有效测试段旋转。不得只选有利指标汇报。

## 已验证
- 雷达GT标签积分可还原对应参考轨迹（独立复核见verification.json）。
- LSTM多时间步公式与独立NumPy参考最大误差5.55e-17。
- 纯旋转杠杆臂检查通过，避免遗漏雷达外参平移。
- 缓存与原始CNN输入前向的微小误差来自GPU批次计算路径，检查阈值明确记录。
- 权重重载输出完全相同。
- scripts/infer_radar_pair.py只读取两帧雷达点云和checkpoint，不读取GT或IMU；实际测试帧953/958已推理通过。
- 从导出TUM文件独立重算ATE，与results.json一致。

## 产物
- best checkpoint: runs/pilot_lab0_gap5_v1/lr_1e-05/best.pt
- results: runs/pilot_lab0_gap5_v1/results.json
- verification: runs/pilot_lab0_gap5_v1/verification.json
- independent inference: runs/pilot_lab0_gap5_v1/independent_inference.json
- plot: runs/pilot_lab0_gap5_v1/trajectory_and_loss.png
- predictions: runs/pilot_lab0_gap5_v1/{train,val,test}_predictions.npz
- TUM trajectories: runs/pilot_lab0_gap5_v1/test_*segment0.tum
- data/provenance: data/coloradar_lab0_gap5_v1/manifest.json
- per-learning-rate configs and history: runs/pilot_lab0_gap5_v1/lr_*/{config.json,history.csv}
- 原始数据只读；新增文件全部在milliEgo目录内。

## 命令
```bash
cd /home/metaiot_guest/Workspace/qinzx/milliEgo
source env.sh
# 重跑必须使用新的output目录，以保留当前试验与测试结果。
CUDA_VISIBLE_DEVICES=3 uv run --locked python scripts/train_single_sequence.py --epochs 200 --overfit-steps 400 --learning-rates 1e-5,1e-4 --output runs/pilot_lab0_gap5_v2
```

## 下一步
当前测试段已经用于最终评估与诊断，后续反复调参不能继续把该段当作未接触测试。优先扩展训练序列并冻结新的独立验证/测试序列；若只做本序列诊断，可研究投影、CNN微调和时序长度，但须明确标记开发实验，与原版T=1设置分开汇报。

参考：官方milliEgo upstream固定提交9b163b92f871b7912dc3a27969b1761556cddad2；ColoRadar随附官方工具；Keras 2.1.6 keras/layers/recurrent.py。

