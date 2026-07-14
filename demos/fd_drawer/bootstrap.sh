#!/usr/bin/env bash
# 把 demo 流水线铺到脚本所预期的目录布局并改写绝对路径。
# 用法:  ./bootstrap.sh [根目录]     默认根目录 = $HOME/Downloads
set -euo pipefail
ROOT="${1:-$HOME/Downloads}"
HERE="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$ROOT/FD-light/fd_walk/stage2_out" "$ROOT/FD-light/fd_walk/output_v3" "$ROOT/geo"
cp -v "$HERE"/fd_walk/*.py "$HERE"/fd_walk/*.json "$HERE"/fd_walk/*.npz "$ROOT/FD-light/fd_walk/"
cp -v "$HERE"/fd_walk/stage2_out/layout.json "$ROOT/FD-light/fd_walk/stage2_out/"
cp -v "$HERE"/assets/geo/*                    "$ROOT/geo/"
cp -v "$HERE"/assets/cabinet_v2_with_drawer_v0*.obj "$ROOT/"
# GOLD 物理帧数据放到执行器输出位:不跑 GPU 仿真也能直接合成/导 USD
cp -v "$HERE"/outputs/stage2_states_urun87_GOLD_v35.npz "$ROOT/FD-light/fd_walk/stage2_out/stage2_states.npz"

# 脚本内的绝对路径按新根目录改写(只改铺出去的副本,仓库文件不动)
if [ "$ROOT" != "/home/ps/Downloads" ]; then
  grep -rl '/home/ps/Downloads' "$ROOT/FD-light/fd_walk" --include='*.py' \
    | xargs -r sed -i "s|/home/ps/Downloads|$ROOT|g"
  sed -i "s|/home/ps/Downloads|$ROOT|g" "$ROOT/geo/tetify_toys.py" 2>/dev/null || true
fi

cat <<EOF

[bootstrap] 完成。还差两步(见 README):
  1) 从本仓库 GitHub Release (demo-fd-drawer-v1) 下载 fd-urdf-full.tar.gz,
     解压到 $ROOT/FD-light/   (=> $ROOT/FD-light/fd-urdf-full/FD-URDF/fd.urdf)
  2) clone 引擎 https://github.com/haoxiangNtu/Stiff-GIPC 到 $ROOT/Stiff-GIPC-v08
     并 git checkout v0.8.3,按其 README 编译安装(需 CUDA GPU)
EOF
