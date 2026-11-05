import json,csv,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/expanded_training_v1"
def main():
    r=json.loads((OUT/"results.json").read_text());cfg=r["protocol"];agg=r["test_summary"];data=json.loads((ROOT/"data/coloradar_lab_expanded_v1/manifest.json").read_text())
    keys={"pair_translation_rmse_m":"逐对平移RMSE（米）","pair_rotation_rmse_deg":"逐对旋转RMSE（度）","ate_initial_rmse_m":"起点对齐ATE（米）","ate_se3_rmse_m":"SE(3)刚体对齐ATE（米）","endpoint_error_m":"终点位置误差（米）"}
    controls=agg["repeated_base"]["full_200_epochs"];expanded=agg["expanded_data"]["full_200_epochs"]
    changes={k:(expanded[k]["mean"]/controls[k]["mean"]-1)*100 for k in keys}
    lines=["# 增加训练序列的固定算法对照","","## 实际执行","","增加lab run1–4作为训练集，保留lab run0原训练、验证、评估物理区间。原始数据只读。","",
    "- 原训练样本：%d 对；增加后：%d 对（约 %.2f 倍）。"%(cfg["base_train_pairs"],cfg["expanded_train_pairs"],cfg["expanded_train_pairs"]/cfg["base_train_pairs"]),
    "- 新增四序列有效训练对：%d。"%data["added_train_pairs"],
    "- 原run0验证230对、评估230对原生10Hz数据保持不变；主要指标仍将5个原生预测合成为原来的46个0.5秒区间，在相同47个时刻评价。",
    "- 网络、CNN冻结、损失、学习率1e-5、RMSprop、batch64、T=1均不变，没有旋转输出缩放或CNN微调。",
    "- 两组均训练200轮，每组每个种子%d次更新；重复原数据组匹配新增数据组每轮样本数。"%cfg["steps_per_arm"],
    "- 三个预先固定随机种子：%s。"%", ".join(map(str,cfg["seeds"])),
    "- 相同种子两组初始权重哈希相同；两个实验组使用相同的、仅从训练图像得到的归一化均值。",
    "- 只用验证损失选权重，全部训练完成后统一评估；保留全部种子的结果。","","## 同更新次数的结果（三个种子均值±标准差）","","|指标|重复原训练数据|新增四条训练序列|误差变化|","|---|---:|---:|---:|"]
    for k,label in keys.items():
        a=controls[k];b=expanded[k];lines.append("|%s|%.5f ± %.5f|%.5f ± %.5f|%+.2f%%|"%(label,a["mean"],a["std"],b["mean"],b["std"],changes[k]))
    lines+=["","负变化表示误差降低。该比较控制了更新次数，不能把额外迭代误说成新增数据效果。","","## 更新预算对照","","|组|更新预算|平移RMSE（米）|旋转RMSE（度）|起点ATE（米）|","|---|---|---:|---:|---:|"]
    for arm in cfg["arms"]:
        for mode in ["first_2200_updates","full_200_epochs"]:
            m=agg[arm][mode]
            lines.append("|%s|%s|%.5f|%.5f|%.5f|"%(arm,mode,m["pair_translation_rmse_m"]["mean"],m["pair_rotation_rmse_deg"]["mean"],m["ate_initial_rmse_m"]["mean"]))
    lines+=["","## 核验","","- 训练帧按序列ID和原生帧ID检查，与run0验证/评估帧无交集；新增录制时间与run0无重叠。",
    "- 所有六次训练保存200轮日志、两个固定更新预算的权重与预测。",
    "- 原始雷达图像输入与CNN缓存路径数值核验通过；从导出TUM独立重算ATE通过。",
    "- 模型源码SHA256保持不变；数据与训练配置保存在manifest/protocol中。","","## 适用范围","","这是针对已经在开发中多次查看的run0区间的诊断实验，不能再称为未接触的独立测试。",
    "所有训练来自同一lab环境，不能据此证明跨环境泛化。正式baseline仍需要冻结新的序列级划分。",
    "当前投影仍为明确标注的适配版，缺失的原投影配置问题没有因增加训练数据而自动解决。",
    "归一化对照使用统一训练均值；重复原样本组仅共享该标量，不接触新增序列的标签或输入样本。",
    "不得只报告改善指标而省略变差指标，也不得将新增数据造成的提升包装成新算法。",
    "","## 产物","","- protocol.json：固定实验协议","- results.json：所有种子、预算与误差","- comparison.png：首个预先固定种子的评估轨迹及验证曲线","- 各组目录/history.csv：200轮训练日志","- 各组目录/*.pt：验证选定权重","- 各组目录/*_test0.tum：评估轨迹","- data/coloradar_lab_expanded_v1/manifest.json：新增训练数据与隔离证据"]
    if changes["ate_initial_rmse_m"]<-10 and changes["pair_rotation_rmse_deg"]<-5:
        conclusion="同更新次数下，新增训练数据使平均轨迹与旋转误差均降低，支持训练数据覆盖不足是此前差结果的一个实质因素；仍不代表全部问题已解决。"
    elif changes["ate_initial_rmse_m"]<-5:
        conclusion="新增数据带来一定平均轨迹误差下降，但其他指标和跨种子波动必须同时考察，尚不能据此宣称复现成功。"
    else:
        conclusion="本轮未观察到足以解决问题的平均轨迹改善；不能把增加样本直接宣称为已找到主要原因。"
    lines[2:2]=["## 结论","",conclusion,""]
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n")
    counts=[];initial={}
    for row in r["runs"]:
        folder=OUT/(row["arm"]+"_seed%d"%row["seed"])
        h=list(csv.DictReader((folder/"history.csv").open()));assert len(h)==200
        assert np.isfinite(np.load(folder/(row["mode"]+"_predictions.npz"))["prediction"]).all()
        counts.append((row["arm"],row["seed"],row["mode"]))
        initial.setdefault(row["seed"],set()).add(row["initial_state_sha256"])
    assert len(counts)==12 and all(len(x)==1 for x in initial.values()) and r["model_unchanged"]
    checks={"six_complete_training_runs":True,"twelve_fixed_budget_evaluations":True,"all_histories_200_epochs":True,"initial_weights_matched_by_seed":True,"model_unchanged":True,"split_disjoint":data["split_frame_ids_disjoint"],"recording_times_disjoint":data["recording_times_disjoint"]}
    (OUT/"completion_checks.json").write_text(json.dumps(checks,indent=2))
    seed=cfg["seeds"][0];fig,ax=plt.subplots(1,2,figsize=(12,4.5))
    for arm,label in [("repeated_base","Repeated original data"),("expanded_data","Added four training sequences")]:
        folder=OUT/(arm+"_seed%d"%seed);P=np.loadtxt(folder/"full_200_epochs_test0.tum");G=np.loadtxt(folder/"full_200_epochs_reference0.tum")
        ax[0].plot(P[:,1],P[:,2],label=label)
        if arm=="repeated_base":ax[0].plot(G[:,1],G[:,2],"k--",label="Reference")
        h=np.genfromtxt(folder/"history.csv",delimiter=",",names=True);ax[1].semilogy(h["updates"],h["val_loss"],label=label)
    ax[0].set(xlabel="Radar-origin x [m]",ylabel="Radar-origin y [m]",title="Same evaluation trajectory; first fixed seed");ax[0].axis("equal");ax[0].legend(fontsize=8)
    ax[1].set(xlabel="Optimizer updates",ylabel="Validation weighted MSE",title="Same update budget and algorithm");ax[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/"comparison.png",dpi=170);plt.close(fig)
    print(json.dumps({"conclusion":conclusion,"train_pairs":[cfg["base_train_pairs"],cfg["expanded_train_pairs"]],"relative_change_percent":changes,"test_summary":agg,"checks":checks},ensure_ascii=False,indent=2))
if __name__=="__main__":main()

