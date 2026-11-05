# 2026-11-05 存储迁移记录

按用户最后指定路径，迁至 `/storage/ssd/metaiot_guest/RadarProbMap/baselines/milliego`。最初指定的 NAS 路径未写入，因当前账号没有写权限，用户随后改用 SSD。

代码工作目录仍为 `/home/metaiot_guest/Workspace/qinzx/milliEgo`。迁移目录为 `data`、`runs`、`weights`、`reports`、`.venv`、`.cache`、`.tmp`，原位置全部为指向 SSD 对应目录的绝对软链接。

复制采用 rsync 保留元数据、符号链接与目录内硬链接，并对源和目标进行内容校验。每个目录校验通过后才切换链接并移除系统盘原副本。七个目录合计校验 46,262 个普通文件；详见 `storage_migration_manifest.json`。不同顶层目录的硬链接不保证在迁移后继续共享磁盘块，因此目标占用与系统盘释放量不完全相同。

完成后，系统盘项目目录约 11 MiB；系统盘可用空间由检查初期约 288 GiB 增至约 309 GiB。共享服务器上其他任务也会影响磁盘空间，这两个数是检查时的近似读数。SSD 目标总占用约 33 GiB，包括五组此前位于内存盘的实验副本；内存盘原件保留。

验证结果：

- 七个目录均通过 rsync 文件内容及元数据一致性核对，原位置链接正确；重复运行迁移路径检查通过。
- 通过原目录 `source env.sh`，原 `.venv/bin/python` 可使用，PyTorch 为 2.7.1+cu126。
- 成功读取 `weights/cnn_torch.pt` 并完成雷达模型前向，输出形状 `(1,1,6)` 且数值有限。
- 通过原 `runs/` 路径读取 lab4 谱图缓存，形状为 `(1386,32,128)`。
- 五组 `/dev/shm` 实验已复制并校验至 SSD 的 `ram_experiments/`。
- 公共原始数据没有移动；网络结构、损失和训练权重没有修改，没有重新训练。

Git 仅维护代码、锁文件、配置和小型记录，数据与模型二进制通过 `.gitignore` 排除。服务器不具备本仓库 GitHub SSH 推送身份，因此通过本机已授权身份完成推送，并把相同提交以 Git bundle 同步回服务器。私钥没有复制到服务器。
