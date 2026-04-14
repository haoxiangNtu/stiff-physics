# StiffGIPC — GPU IPC Physics Engine for Python

StiffGIPC is a GPU-accelerated Incremental Potential Contact (IPC) physics engine with Python bindings. It supports rigid-body articulations (URDF), deformable cloth/shell simulation, and coupled rigid-deformable interactions — all running on CUDA.

The `stiff-physics` package provides a self-contained pre-compiled engine. No C++ toolchain or CUDA SDK is required to use it.

## System Requirements

| Requirement | Details |
|---|---|
| OS | Linux x86_64 (Ubuntu 20.04+) |
| GPU | NVIDIA RTX 4090 (sm_89) or RTX 5090 (sm_120) |
| Driver | NVIDIA driver with CUDA 12.x support |
| Python | 3.10 |

## Installation

### 1. Install the engine (from GitHub Release)

```bash
pip install https://github.com/haoxiangNtu/stiff-physics/releases/download/v0.1.0/stiff_physics-0.1.0-cp310-cp310-linux_x86_64.whl
```

### 2. Install visualization dependencies

```bash
pip install polyscope scipy
```

## Quick Start

```bash
git clone https://github.com/haoxiangNtu/stiff-physics.git
cd stiff-physics
python examples/case_26_arm_cloth_semi_implicit.py
```

This launches an interactive scene with an XArm7 robot arm and a free-falling shirt. Click **Run** to start the simulation, and drag the joint sliders to interact with the cloth in real-time.

## API Overview

```python
from stiff_physics.engine import Engine, Config
from stiff_physics.robot import Robot

# Configure the simulation
config = Config(
    dt=0.01,                              # time step (seconds)
    cloth_young_modulus=1e4,              # cloth stiffness
    semi_implicit_enabled=True,           # fast semi-implicit solver
    assets_dir="/path/to/your/assets/",   # directory containing URDF/mesh files
)

# Create engine and load scene
engine = Engine(config)
engine.native.load_urdf(assets_dir + "robot.urdf", transform, True, False, 1e7)
engine.load_mesh("cloth.obj", dimensions=2, body_type="FEM", transform=tf)
engine.finalize()

# Step the simulation
engine.step()

# Read back vertex positions
vertices = engine.get_vertices()
faces = engine.get_surface_faces()
```

### Key Classes

| Class | Description |
|---|---|
| `Config` | Simulation parameters (time step, material properties, solver settings) |
| `Engine` | Core simulation engine — load meshes/URDFs, step physics, read results |
| `Robot` | Joint-level control for articulated bodies (revolute/prismatic joints) |

### Config Parameters

| Parameter | Default | Description |
|---|---|---|
| `dt` | 0.01 | Time step in seconds |
| `cloth_young_modulus` | 1e5 | Cloth stiffness |
| `cloth_density` | 200 | Cloth mass density |
| `friction_rate` | 0.4 | Coulomb friction coefficient |
| `semi_implicit_enabled` | False | Enable semi-implicit solver for faster convergence |
| `assets_dir` | `""` | Path to assets directory (URDF, meshes) |
| `gravity` | (0, -9.8, 0) | Gravity vector |

## Project Structure

```
stiff-physics/
  README.md
  examples/
    case_26_arm_cloth_semi_implicit.py    # Interactive arm + cloth demo
  assets/
    sim_data/urdf/xarm/                   # XArm7 robot URDF + collision meshes
    triMesh/shirt_6436v.obj               # Shirt cloth mesh
```

## Troubleshooting

**`ImportError: libstiffgipc_core.so: cannot open shared object file`**
The wheel bundles the core library. If you see this, make sure you installed the wheel (not just cloned the repo).

**`CUDA error: no kernel image is available for execution on the device`**
Your GPU architecture is not included in this build. The pre-built wheel supports sm_89 (RTX 4090) and sm_120 (RTX 5090) only.

**`ImportError: liburdfdom_model.so: cannot open shared object file`**
Install the urdfdom system library: `sudo apt install liburdfdom-dev`

## License

This project is provided as pre-compiled binaries. Contact the authors for licensing information.
