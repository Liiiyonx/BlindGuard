# Git 历史大文件清理手册

## 背景

当前工作区的 `.git` 已达到约 4.99 GiB。即使 `downloads/`、模型和训练产物已经写入
`.gitignore`，只要它们曾经被提交过，对象仍会留在 Git 历史中，普通删除工作区文件无法缩小
`.git`。

这不是常规开发步骤。历史重写会改变所有受影响提交的 SHA，并要求所有协作者重新克隆或按
专门流程恢复。执行前必须通知团队、暂停推送，并完成可验证的镜像备份。

## 先审计，不改历史

```powershell
python scripts/audit_git_history.py --top 30 --min-mb 1
python scripts/audit_git_history.py --path downloads --top 50 --min-mb 1
git count-objects -vH
git status --short
```

如果安装了 `git-filter-repo`，还可生成体积分析：

```powershell
git filter-repo --analyze
```

分析结果位于 `.git/filter-repo/analysis/`。先确认待删除路径、影响范围和最新提交，再继续。

## 必须完成的备份

在仓库父目录执行，不要覆盖已有目录：

```powershell
git clone --mirror . ..\BlindGuard-backup.git
git bundle create ..\BlindGuard-before-filter.bundle --all
git bundle verify ..\BlindGuard-before-filter.bundle
```

至少把镜像目录和 bundle 复制到另一块磁盘或受控备份位置，并在隔离目录验证可以克隆。
没有可恢复备份时，不要继续。

## 清理步骤

建议只删除明确的大目录。不要在一次操作中同时改写代码、删除历史和清理无关文件。

```powershell
python -m pip install git-filter-repo
git filter-repo --dry-run --path downloads/ --invert-paths
git filter-repo --path downloads/ --invert-paths
```

`--dry-run` 仅用于确认匹配范围；确认后执行第二条命令。`git filter-repo` 通常会移除
`origin` 远程，以防误推。重写后检查：

```powershell
git log --all --oneline -- downloads
git status --short
git count-objects -vH
```

若仍有旧对象占用空间，再清理 reflog 并执行本地垃圾回收：

```powershell
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git count-objects -vH
```

## 推送重写后的历史

只有在备份验证、团队确认和本地检查全部完成后执行。先把原始远程地址加回，并记录它是
共享仓库，不是个人仓库。

```powershell
git remote add origin <repository-url>
git fetch origin
git push --force-with-lease origin master
git push --force-with-lease --tags origin
```

不要使用无保护的 `--force`。若 `--force-with-lease` 被拒绝，说明远程在重写后发生了新提交；
停止推送并先协调，不能直接覆盖。

## 团队恢复

历史重写后，旧克隆不能继续普通 `pull`。每位协作者应：

1. 先把本地未提交改动导出为补丁或提交到临时分支。
2. 重新克隆共享仓库。
3. 将未完成改动应用到新克隆。
4. 删除旧克隆，避免误推旧历史。

如果重写结果错误，使用已验证的 mirror 或 bundle 恢复，不要尝试在已改写仓库中反向 merge。

## 防止再次发生

- 保持 `.gitignore` 对大目录、模型、视频和训练产物的覆盖。
- 提交前运行 `git status --short` 和 `git diff --cached --stat`。
- 对超过约 10 MiB 的待提交文件做来源确认。
- 二进制模型优先用 release、对象存储或 Git LFS 管理；本手册不会自动启用 Git LFS。
