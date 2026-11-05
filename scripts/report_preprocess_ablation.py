import json,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/preprocess_ablation_v1"
r=json.loads((OUT/"results.json").read_text());a=r["aggregate_test_metrics"];old=a["A_old"];new=a["D_both_corrected"]
keys={"pair_translation_rmse_m":"逐对平移RMSE（米）","pair_rotation_rmse_deg":"逐对旋转RMSE（度）","ate_initial_rmse_m":"起点对齐ATE（米）","ate_se3_rmse_m":"刚体对齐ATE（米）"}
lines=["# 预处理修正后的成对重训练验证","","## 本轮结论","","本轮实际重新训练了4种数据处理×3个固定随机种子，每次200轮；不是仅修改文件或运行语法检查。","",
"网络、预训练CNN冻结、损失、原配置学习率1e-5、batch64、T=1、RMSprop、随机种子和划分在组间固定。所有组均使用恒定学习率，无旋转输出缩放、CNN微调或梯度裁剪。","","## 修正前后测试结果（三个种子的均值±标准差）","","|指标|旧处理|两项修正|变化|","|---|---:|---:|---:|"]
change={}
for key,label in keys.items():
    before=old[key]["mean"];after=new[key]["mean"];pct=(after-before)/before*100;change[key]=pct
    lines.append("|%s|%.5f ± %.5f|%.5f ± %.5f|%+.2f%%|"%(label,before,old[key]["std"],after,new[key]["std"],pct))
lines+=["","数值越小越好；负变化表示误差降低。不能只选择改善的指标汇报。","","## 四组全部结果","","|组|平移RMSE（米）|旋转RMSE（度）|起点ATE（米）|","|---|---:|---:|---:|"]
for name,met in a.items():lines.append("|%s|%.5f|%.5f|%.5f|"%(name,met["pair_translation_rmse_m"]["mean"],met["pair_rotation_rmse_deg"]["mean"],met["ate_initial_rmse_m"]["mean"]))
lines+=["","A_old：旧像素尺度+旧标签；B_normalization_only：只修正归一化；C_euler_only：只修正旋转标签；D_both_corrected：两项均修正。","",
"## 检查与证据","","- 每个相同种子在四组中的初始网络权重哈希完全一致。","- 同一组的模型仅按验证损失选轮次，测试结果没有用于调学习率或选择种子。","- 12次训练完整保存200轮train/val日志、预测和导出轨迹；每组第一个固定种子另存checkpoint。","- 保存重载输出对照、原始图像输入与缓存特征路径对照、从导出TUM独立重算ATE均通过。","- 本轮模型源码哈希保持不变。","","## 结果适用范围","","这是旧单序列上已接触测试段的预处理纠错检查，不是新的独立测试成绩。","投影仍保持上一轮自选参数，以隔离两项修正的影响；因此即使结果改善，也不能宣称已完成原论文baseline复现。","本轮与此前pilot表格的batch和调度不同；因而因果比较应使用本轮A_old与D_both_corrected，不跨轮直接归因。","",
"## 文件","","- protocol.json：运行前固定协议","- results.json：12次结果、三种子汇总、数值核验","- */history.csv：逐轮日志","- */test_prediction_segment0.tum：导出预测轨迹","- comparison.png：首个固定种子的轨迹与结果概览","",
"本轮运行时间：%.2f秒（不含之前数据准备）。"%r["wall_seconds"]]
(OUT/"REPORT.md").write_text("\n".join(lines)+"\n")
with (OUT/"comparison_summary.csv").open("w",newline="") as f:
    w=csv.writer(f);w.writerow(["metric","old_mean","corrected_mean","relative_error_change_percent"])
    for k in keys:w.writerow([k,old[k]["mean"],new[k]["mean"],change[k]])
print(json.dumps({"old_vs_corrected_relative_change_percent":change,"results":{k:{j:met[j] for j in keys} for k,met in a.items()},"verification":r["verification"]},indent=2))

