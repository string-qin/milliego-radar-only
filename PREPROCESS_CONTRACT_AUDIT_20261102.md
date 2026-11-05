# Baseline数据输入核对与修正（2026-11-02）

## 用户约束
这是用于公平对比的baseline。除用户明确要求的PyTorch迁移和radar-only消融外，不以改网络、改损失、缩放旋转输出、微调CNN或额外调参的结果冒充原算法。之前的五组优化器对照仅作诊断。

## 已确认的两处预处理错误
1. 输入未经过原打包阶段的归一化/均值减除。
原milliEgo data_loader只读HDF5；没有在loader除255，并不代表原始像素可直接送网络。其README指向的数据工具，在create_h5_dataset.py的load_normalize_1channel_img执行：
img.astype(float32)
img = (img-min_range)/(max_range-min_range)
img -= dataset_mean
此前原始uint8图转float32直接输入CNN，遗漏此步骤。该错误与网络层/学习率无关。

2. 原欧拉角标签约定与上一轮不一致。
原rotated_to_local调用原mat2euler，得到z,y,x后按x,y,z、度存储。等价于SciPy大写XYZ，而非上一轮的小写xyz。
上一轮使用自身encode/decode的round-trip只能证明自洽，没有证明与原函数一致。
本序列相同物理旋转的两种标签表示最大分量差6.66005度。此数值不是新旧模型的预测改善幅度，也不代表它单独解释全部误差。

## 已完成修正和验证
- 新数据目录：data/coloradar_lab0_contractfix_v2，旧版本与结果未覆盖。
- 归一化采用image/255-training_mean，训练帧均值0.004868511125153188，均值不读取验证/测试帧。
- 四个抽样PNG直接执行原load_normalize_1channel_img函数对照（仅用Pillow替代已移除的scipy.misc.imread灰度解码），最大误差0。
- 230对标签直接执行原rotated_to_local公式；与原mat2euler独立核对最大误差7.91e-12。
- 原euler2mat只补Python3兼容的functools.reduce，未修改公式。
- HDF5字段mmwave_middle_data为(N,1,64,256,1)，label_data为(1,N-1,6)，按连续段分别导出，避免跨GT缺口连接。
- model.py哈希检查通过，本轮未更改网络或训练/损失配置。
- 新数据标记baseline_ready=False，未用它重新训练、评测或继续调试已接触的测试段。

## 投影仍未完整认证
找到原README指向OdomBeyondVision的公开保留副本：
https://github.com/mfkiwl/OdomBeyondVision
固定提交d30b06427457bf65a06c9c52245874d736b585fa。
原MAPS-Lab链接当前404，不能宣称该副本已认证为原2020实验的完整同一版本。

工具显示：
- 相邻3帧叠加（手持/UGV；UAV说明为2帧），并投影后OpenCV默认插值放大4倍；
- 依赖pcl2depth配置给出的h_fov,v_fov,h_res,v_res,max_v,mmwave_dist_thre；
- 该保留仓库toolkit/config.yaml及其历史中未找到pcl2depth配置段；
- 主仓库utility/pcl2depth.py用平面距离sqrt(x²+y²)，后续数据工具用sqrt(x²+y²+z²)。不能未经确认混用；
- 原函数像素截断、额外边界行列和重复像素覆盖规则与上一轮nearest-depth直接64x256实现也不同。

本次v2仅隔离修复已确认的归一化与标签错误，仍使用v1的自选投影图像，因此不能称为正式baseline数据。
没有擅自补造上述配置，也没有把更新后的工具当作原论文流程的绝对证据。

## 为什么之前检查没发现
检查停留在文件可读、形状可输入、梯度存在、坐标积分自洽。遗漏了两层关键验证：
- 沿数据来源追踪到HDF5生成之前；
- 用原函数比较标签数值，而非只对照自写函数的逆变换。
这是复现执行中的核对不足，不是用户要求不清楚，也不是已证明原数据集有问题。

## 下一步准入条件
获得原始投影配置或可信的原处理样本，对同一合成点云及真实帧逐像素对照原函数，再冻结预处理配置。然后才使用原算法设置开展正式训练。此前训练和诊断报告只保留为探索记录，不作为baseline表格成绩。

## 证据
scripts/correct_data_contract.py
data/coloradar_lab0_contractfix_v2/verification.json
data/coloradar_lab0_contractfix_v2/normalization.json
reports/correct_data_contract.log
reports/model_before_preprocess_fix.sha256
reference_odombeyondvision/toolkit/create_dataset_milliego/create_h5_dataset.py
reference_odombeyondvision/toolkit/radar_preprocess/process_radar_handheld.py
upstream/eulerangles.py

