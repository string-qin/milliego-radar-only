# 数据盘布局与迁移

代码目录：`/home/metaiot_guest/Workspace/qinzx/milliEgo`。

数据目录：`/storage/ssd/metaiot_guest/RadarProbMap/baselines/milliego`。

把代码目录的 `data/`、`runs/`、`weights/`、`reports/`、`.venv/`、`.cache/`、`.tmp/` 迁入数据目录，并在原位置保留同名绝对软链接。原脚本访问路径不变，`env.sh` 中的缓存与临时目录也会落在数据盘。公共原始数据仍从 `/storage/data/metaiot_data/public_dataset/coloradar` 只读使用。

迁移前应停止本项目训练与预处理任务，确保目标目录可写。迁移脚本默认仅检查路径：

```bash
python3 scripts/migrate_storage.py
python3 scripts/migrate_storage.py --apply
```

脚本逐目录复制，使用 rsync 校验文件内容与元数据。校验不一致就停止；一致后把原目录改名为临时备份，创建软链接，再移除已经核对过的原副本。各目录完成情况保存在数据盘 `migration_20261105.json`。中断后可重跑，已迁移的正确软链接会跳过。切换窗口内不要启动训练。

此前因系统盘配额限制而写在 `/dev/shm` 的 milliEgo 实验，也复制到数据盘 `ram_experiments/` 并校验；保留内存盘原件，避免打断已有引用。

新的代码 checkout 可以在数据迁移完成后执行：

```bash
python3 scripts/setup_storage.py --storage /storage/ssd/metaiot_guest/RadarProbMap/baselines/milliego
source env.sh
.venv/bin/python -c 'import torch; print(torch.__version__)'
```

虚拟环境包含服务器原代码路径，不应直接拿到另一台机器使用；其他机器应通过 `uv.lock` 重新建环境。本次目标位于独立 SSD 挂载点 `/storage/ssd`，与系统盘 `/` 分离。

Git 只存代码、环境锁文件、配置和小型实验记录。历史上放在 `runs/` 的代码另存到 `experiments/legacy_runs/`；`setup_storage.py` 会在缺失时恢复它们到运行目录，已有不同文件不会被覆盖。维护这些代码时同步更新运行副本。

2026-11-05 已完成上述七个目录迁移，所有复制内容经 rsync 校验通过；原位置均为正确软链接。五组内存盘实验也已复制到 SSD 并校验。详细记录见数据盘 `migration_20261105.json` 和仓库 `MIGRATION_REPORT_20261105.md`。
