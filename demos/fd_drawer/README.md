# FD 人形机器人 · 纯物理抓放抽屉 Demo 制作流水线

FD 人形走到双抽屉柜前 → 右手舀抓玩偶 → 自然松指落入抽屉 → 张开手掌把抽屉推关到底 →
右腿向右后撤步立正 → 右转走离。操作段为 **Stiff-GIPC GPU IPC 纯物理仿真**(非关键帧动画):
玩偶是弹性 FEM 体,抽屉是被动棱柱导轨上的自由刚体,所有接触真实求解。

成片:[`outputs/fd_4stage_pure_physics_v3.mp4`](outputs/fd_4stage_pure_physics_v3.mp4)(67.8s)
| USD 回放验证:[`outputs/usd_playback_proof.mp4`](outputs/usd_playback_proof.mp4)

## 1. 目录

| 路径 | 内容 |
|---|---|
| `fd_walk/` | 全部流水线脚本与数据(运行时目录的 1:1 镜像,见 §4) |
| `assets/` | 柜子网格(v03=右抽屉半开工作态,v02=全关 rest 态)、`geo/` 玩偶网格+四面体化工具 |
| `outputs/` | 成片 mp4、USD 动画包(zip)、GOLD 物理帧数据、USD 回放验证视频 |
| `docs/` | v3 定稿技术文档(含完整运行史、失败分析与判定阈值依据) |
| `bootstrap.sh` | 一键铺设目录布局 + 改写绝对路径 |

大文件在 **GitHub Release `demo-fd-drawer-v1`**:`fd-urdf-full.tar.gz`(机器人 URDF+STL,128MB)、
`fd_4stage_animation_v3_full.usdc`(103MB 全精度 USD,可直接 Houdini File>Import;等同 outputs/ 里的 zip)。

## 2. 环境准备

```bash
# ① python 环境(建议 conda,名字任意,下文记作 $P311)
conda create -n test_stiff08 python=3.11 -y
pip install numpy==2.4.6 scipy==1.17.1 trimesh==4.12.2 polyscope==2.5.0 \
    fast_simplification==0.1.13 usd-core==26.5 pillow==12.3.0
# 版本已在全新环境实测验证;polyscope 必须 2.5.0——2.6.x 在无 show() 的批量截图
# 循环(composite_render)中会在首帧后阻塞。
# 系统需有 ffmpeg;需要 NVIDIA GPU(仅"重跑物理仿真"这一步需要,回放/合成/设计工具不需要)

# ② 铺设目录布局(默认铺到 $HOME/Downloads,脚本会自动改写路径)
./bootstrap.sh

# ③ 机器人 URDF:从 Release 下载 fd-urdf-full.tar.gz 解压到 <根目录>/FD-light/
# ④ 物理引擎(只在重跑仿真时需要):
#    git clone https://github.com/haoxiangNtu/Stiff-GIPC <根目录>/Stiff-GIPC-v08
#    cd <根目录>/Stiff-GIPC-v08 && git checkout v0.8.3   # 按其 README 编译;GOLD 定稿跑于 v0.8.2
```

## 3. 三个工具(demo 制作的日常界面)

以下 `$P311` = 你环境里的 `python3.11`;都需要显示器(polyscope 弹窗)。

**① 关键位姿设计器(纯 FK,秒开,零物理/零碰撞)**
```bash
$P311 fd_walk/fd_pose_designer.py
```
- 全关节滑条(URDF 真实限位)+ 手指分组通道 + 底座 x/y/z/yaw;柜子/玩偶参考网格可开关;
- KEYFRAMES:增/改/删/排序,存取 `fd_keyposes.json`,`preview play` 按成片同款 0.45rad/s 限速插值试播;
- STAGE scrub(movie-accurate):S1 走近 / S3 右后撤步 / S4 转身走离,逐帧复刻合成器的基座偏移
  (与 mp4 逐位一致,转弯不会视觉穿柜),任意帧可捕获为关键帧;
- S2 TIMELINE edit:载入 `stage2_user_timeline.json` 逐 row 查看/修改/写回,另存 `*_edited.json`;
  支持删单行/删连续区间(里程碑、舀抓伺服、keeper 释放等承重行会拒删并提示;删行后 meta
  行号引用自动重映射;缝隙默认保留原时刻,勾 `compact gap to 2s` 则整体提前)。

**② 粗网格实时交互台(真引擎,开场建模 ~1 分钟,需 GPU)**
```bash
$P311 fd_walk/fd_interact_bench.py
```
- v02 柜+玩偶实时物理:双臂/双手滑条直接推抽屉(看 `pull` 读数)、碰玩偶;
- `apply designer keypose` 把设计器存的关键帧设为马达目标 → 设计与物理验证闭环;
- Palm IK 定点解臂;`REC` 录制姿态到 `fw_user_push.json`(成片的推关抽屉手臂弧线就是这么录的)。

**③ 成片 USD 回放器(秒开)**
```bash
$P311 fd_walk/fd_usd_viewer.py [可选 .usdc 路径]
```
- 默认读 `output_v3/fd_4stage_animation_v3_full.usdc`(先解压 outputs/ 的 zip 或从 Release 下载);
- 帧滑条 0-1713 / play / 变速;`movie cam` 勾选=成片运镜,取消=自由视角逐帧检查。

## 4. 全流程复刻(设计 → 仿真 → 合成 → USD)

`bootstrap.sh` 已把 GOLD 物理帧放到 `stage2_out/stage2_states.npz`——**不跑 GPU 也能直接从第 4 步
开始**合成出成片。完整链路(根目录记作 `$ROOT`,在 `$ROOT/FD-light/fd_walk` 下执行):

```bash
V3ENV="FD_CAB_MESH=$ROOT/cabinet_v2_with_drawer_v03.obj FD_CAB_FRONT=0.48 \
  FD_DOLL_SPOT=2.836,-0.196 FD_TSCALE=1 FD_DENSITY=150 FD_JSR=300 FD_FSTR=12 FD_YOUNG=1e5"
rm -f $ROOT/Stiff-GIPC-v08/Assets/sorted_mesh/tmp_mesh_*        # ① 网格改动后必清引擎缓存
env $V3ENV $P311 stage2_user_timeline.py                        # ② 烘焙时间线 json
env $V3ENV S2_OUT=$PWD/stage2_out_v3 $P311 stage2_user_run.py   # ③ GPU 仿真 ~55min(4/4 里程碑)
cp stage2_out_v3/stage2_states.npz stage2_out/
env $V3ENV $P311 gen_layout.py                                  # ④ 顶点→链节布局映射
env FD_CAB_MESH=$ROOT/cabinet_v2_with_drawer_v03.obj $P311 composite_render.py --out composite_frames_v3
ffmpeg -y -framerate 25 -i composite_frames_v3/f%05d.png -c:v libx264 -pix_fmt yuv420p -crf 18 \
  output_v3/fd_4stage_pure_physics_v3.mp4                       # ⑤ 四段式成片
env FD_CAB_MESH=$ROOT/cabinet_v2_with_drawer_v03.obj $P311 export_usd2.py both   # ⑥ USD(全精度)
```

改动作的标准回路:**设计器摆姿/改 timeline → 交互台物理验证 → `*_edited.json` 改名替换
`stage2_user_timeline.json` → 从 ③ 重跑**。

## 5. 关键方法论(踩坑换来的,详见 docs/)

1. **GPU IPC 非逐位确定**:同配置两次结果可不同;毫米级刀锋接触=掷骰子,要用几何净空/keeper
   消除,而不是重试碰运气。
2. **抽屉状态机**:落料前 keeper(导轨摩擦)→ 用户示范回放用自由导轨 → `pull≤阈值` 事件闩锁 →
   **1s 线性目标斜坡软关**(该引擎任何弹簧强度都会瞬移,必须斜坡)→ 到底冻结。
3. **人手示范优于程序规划**:程序算的"张手推"因手指恒前伸 17cm 几何不可行;在交互台里人手录
   5 个 engage 姿态(仅 r1 扫 +5°→−43°),单关节弧线插值零重构风险。
4. **手臂轨迹与手指通道分离**:定稿后手臂 row 冻结,只调手指(松指时序、两段式退压笼)。
5. **弹性体释放要两段式**:0.40 深笼直接张手=侧向弹飞;先退压到 0.15(1s)再张,拇指先撤。
6. **关节插值会甩大弧**:远距离位形间必须走验证过的中转链(sweep→UPRIGHT→SIDE2→SIDE1→hang),
   每过渡行 ≥2s(罚函数臂收敛慢)。
7. **看门狗**:EMA 步耗时escalation + 碰撞上下文打印(双掌/玩偶/pull 坐标)当场定位剐蹭;
   abort 只在 keeper 释放后武装,避免拖拽期 FEM 高耗时误杀。
8. **USD 交付**:刚性链节=全精度网格一次存储+逐帧 xform(103MB);fast_simplification 96%+
   会把 URDF 网格剪成碎点(Houdini 里"破碎虚影");default-time 必须写 frame-0 世界位姿,
   否则 File>Import 看到原点堆;交付前用第三方采样渲染(pxr+polyscope)自证。
9. **引擎网格缓存**:FEM 网格/缩放改动后必须清 `Assets/sorted_mesh/tmp_mesh_*`,否则静默命中
   过期 metis 排序缓存,改动无效。
10. **自制柜网格**:抽屉必须真开顶(solidify 易顺手封顶,肉眼难察觉);台面前挑会吃拖放行程。

## 6. 版本说明

发布内容为 v3 定稿(用户自制双抽屉柜网格)。历史版本(v1 程序化柜、v2 对称柜)留在原
工作区未发布;`stage2_scene.py` 中对应 env 分支仍完整(不设 `FD_CAB_MESH` 即回退 v1)。
