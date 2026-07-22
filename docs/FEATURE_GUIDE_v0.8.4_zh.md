# stiff-physics v0.8.4 功能使用指南

针对仿真需求清单的五个问题，逐条给出 v0.8.4 的对应能力、API 和可运行示例。
所有示例都在 `examples/` 下且可直接运行（安装 v0.8.4 wheel 后 `python examples/xxx.py`）。

---

## 1. 软夹爪 / 灵巧手建模（GRIP 论文的 UMI / LEAP 等效做法）

### UMI 式软指（FEM 软体 + 论文同款物性参数）

论文做法：TetWild 四面体剖分（包络容差 εₑ=5×10⁻⁴）、杨氏模量 E=9.4×10⁶ Pa、
干摩擦系数 μ=3.5。stiff-physics 等效：

```python
from stiff_physics import Engine, Config
import numpy as np

cfg = Config(dt=0.01, density=1e3, poisson_rate=0.49,
             friction_rate=0.4,          # 全局摩擦（桌面/其它物体）
             assets_dir="Assets/")
eng = Engine(cfg)

# TetWild 剖分好的 .msh 四面体网格，按论文物性加载
eng.load_mesh("sim_data/umi_finger.msh", dimensions=3, body_type="FEM",
              transform=T_finger, young_modulus=9.4e6)
finger = eng.get_load_records()[-1]

# [v0.8.4 新] 摩擦不再只有全局一个值：软指单独设 μ=3.5
eng.set_body_friction(finger.body_offset, 3.5)
eng.finalize()
```

- 剖分建议：TetWild `--epsilon 5e-4`（论文同款包络容差）；输出直接喂 `load_mesh`。
- 混合软硬爪（刚性骨架 + FEM 软垫 + 缝合弹簧 + 固定关节）的完整搭建流程见
  `examples/replay_foldshirt_multienv.py` 的 `build_env_abd` / `build_env_fem`
  （STRATEGY_F 资产就是这种结构）。
- 每个软体还可以单独设密度：`eng.load_mesh(..., density=…)` 或
  `eng.set_soft_body_density(body_offset, rho)`（示例 `examples/test_perbody_density.py`）。

### LEAP 式高分辨率碰撞网格（刚体直接用视觉网格）

IPC 的鲁棒性允许 ABD 刚体直接用视觉网格当碰撞网格，但 v0.8.4 有两条工程红线：

1. **网格质量**：v0.8.4 会对表面网格刚体做惯量物理性检查——绕序整体反向会被自动
   翻转纠正；非水密/坏拓扑会在加载时**点名警告**（严重时拒绝加载）。历史教训：
   一个 44 个洞的网格积出**负定惯量**，导致刚体接触时以 16 m/s 弹飞。
2. **顶点规模**：碰撞网格顶点数直接进接触检测。实测一个 36 万顶点的盘子让抓取帧
   从 0.1 秒涨到 40 秒；官方后来换成 244 顶点的简化网格。**建议碰撞网格 ≤5k 顶点**，
   视觉高模只用于渲染。

---

## 2. 按物体单独定义摩擦系数（不再只有引擎级全局参数）

v0.8.3 及以前 `friction_rate` 是全局唯一值；v0.8.4 起：

```python
eng.set_body_friction(body_offset, mu)                    # 该物体自身 μ
eng.set_body_friction(body_offset, mu, ground_mu=0.01)    # 与地面的 μ 单独给
cfg = Config(gd_friction_rate=0.2, ...)                   # 全局“地面摩擦”与物体摩擦解耦
```

- 两个物体接触时组合规则：**几何平均** √(μᵢ·μⱼ)。
- 不调用 `set_body_friction` 的物体沿用全局 `friction_rate` —— 完全向后兼容
  （未用该特性时代码路径与旧版位级一致）。
- 可运行验证：`examples/test_perbody_friction.py`（三组滑块对照：μ_gd=0.4 / 0.01 /
  override，滑行距离 0.43 m vs 1.45 m）。

---

## 3. 接触力 / 应力分布反馈接口（复现论文的 Stress 对比图）

```python
# 每顶点接触力（含地面），单位：牛顿（已验证：静置物体净竖直反力 = 重量，4 位有效）
F = eng.get_vertex_contact_forces(include_ground=True)    # (N_verts, 3)

# FEM 软体的 von Mises 应力（Neo-Hookean Cauchy 应力 → von Mises → 每顶点取 max）
vm = eng.get_fem_von_mises_stress()                       # (N_fem_verts,)
```

画论文那种 0–100 kPa 色标应力图：

```python
import polyscope as ps
ps.init()
m = ps.register_surface_mesh("obj", V, Faces)
m.add_scalar_quantity("von Mises (Pa)", vm_per_vertex, vminmax=(0, 1e5), cmap="jet")
ps.show()
```

- 可运行验证：`examples/test_contact_force_stress.py`（627.2 N == 重量校验）。
- ⚠️ 旧接口 `get_body_contact_force_batched` 返回的是增量势能梯度（力×dt²），
  **不是牛顿**；新代码请一律用 `get_vertex_contact_forces`。

---

## 4. 并行环境隔离、独立终止、资源释放、故障隔离（论文三件事的逐条映射）

### ① 环境隔离 + 各自步长

```python
cfg = Config(multienv_mode="isolated", ...)   # 或 "strict"（见下）
eng = Engine(cfg)
# 逐环境加载 body 后，按 body 归属分组（finalize 之前调用）：
eng.set_body_groups([0,0,...,1,1,...])        # 每个碰撞体属于哪个 env
```

- `isolated`：每环境独立 BVH、独立接触半径（absolute dHat）、**独立 line-search 步长**、
  分段 PCG——物理上完全独立演化。
- `strict`：isolated + 确定性机器。v0.8.4 发布门槛（foldshirt 1551 帧完整轨迹）：
  **run-to-run N=2、run-to-run N=8、cross-env（env0==env1）、batch-invariance
  （env0@N2==env0@N8）四门全部位级一致**。RL 训练要公平性/可复现就用 strict。
- 多环境平铺时建议 `Config(absolute_dhat=...)` 固定接触半径（防 merged-bbox 膨胀）。

### ② 每环境独立终止准则 + 冻结释放

```python
cfg = Config(per_env_exit=True,            # 每 env 达到自己的收敛判据即冻结
             env_newton_iter_cap=100, ...) # 单 env 迭代预算，超限只冻结该 env
...
eng.step()
iters  = eng.get_per_env_newton_iters()    # 每 env 本帧实际牛顿迭代数
status = eng.get_per_env_status()          # 0=active 1=converged 2=timeout 3=diverged
```

收敛的环境被冻结并从后续求解排除（资源让给活跃环境）；半隐式早退也支持逐环境
（`semi_implicit_enabled=True` 时每 env 维护自己的 β）。

### ③ 故障检测与隔离

- 某 env 出现 NaN/Inf：自动隔离（status=3 DIVERGED 冻结），**不污染批内其它环境**。
- 地面接触塌缩（纳米级钉死）：引擎主动抛错点名坏状态，而不是静默产出错误数据。
- 可运行验证：`examples/test_perenv_telemetry.py`（A 正常收敛 / B 超限冻结存活 /
  C 逐环境半隐式三案例）。

---

## 5. 布料初始状态变化（毛巾揉皱/打乱作为初始状态）

核心 API：`teleport_fem_vertices(positions, velocities)`（v0.8.4 修复了 METIS 重排下
写入乱序的问题，布料 round-trip 误差 0.319 m → 0.000）。

**正确姿势——两段式配方**（`examples/recipe_towel_scramble.py`）：

```python
# 第一段：生成揉皱状态（随机落姿自然沉降）
tilt, yaw, h = rng.uniform(...)            # 随机倾角/朝向/落高
eng.load_mesh("towel.obj", dimensions=2, body_type="FEM", transform=T(tilt,yaw,h))
for _ in range(120): eng.step()            # 自然坠落堆叠成皱
np.save("scramble_00.npy", eng.get_vertices())

# 第二段：任何新会话直接复用为初始状态
eng2.load_mesh("towel.obj", ...); eng2.finalize()
eng2.teleport_fem_vertices(np.load("scramble_00.npy"), velocities=None)  # 零速度落位
```

实测：揉皱态复用误差 0.000，40 帧静置漂移 25 µm。

- ⚠️ **不要**直接给平铺布料加位置噪声来"打乱"——噪声幅度超过网格间距的一半就会
  自穿插，IPC 无法从穿插状态恢复（实测 8 mm 噪声 vs 13 mm 网格间距直接炸）。
- ⚠️ `set_vertex_velocities_gpu` 单独调用不会重建 xTilta（物体不动）；改初始状态
  一律走 `teleport_fem_vertices`。

---

## 附：v0.8.4 新增 Config 参数速查

| 参数 | 默认 | 说明 |
|---|---|---|
| `gd_friction_rate` | 跟随 friction_rate | 地面摩擦与物体摩擦解耦 |
| `per_env_exit` | False | 逐环境独立终止/冻结 |
| `env_newton_iter_cap` | 0（关） | 单环境牛顿迭代预算，超限冻结该 env |
| `line_search_max_iter` | 64 | 线搜索回溯预算（旧版硬编码 8 会静默接受非下降步） |
| `semi_implicit_enabled` + `semi_implicit_beta_tol` | False | 半隐式早退（多环境下逐 env β） |
| `absolute_dhat` | 0（bbox 派生） | 固定接触激活半径（多环境平铺必设） |
| `collision_detection_buff_scale` | 1.0 | 碰撞缓冲预留倍率（引擎也会自动增长） |

关节侧新增：`add_revolute_joint(..., passive=True)` / `add_prismatic_joint(..., passive=True)`
——真被动铰链/滑轨（无位置伺服、限位有效），示例 `examples/test_passive_revolute.py`。
