# Embodied RSI Benchmark 数据下载、规范化、去重、配对与验收执行手册

> 版本：Core v1.0 数据准备规范  
> 冻结日期：2026-09-15  
> 面向执行者：Coding Agent / Data Agent  
> 执行原则：**严格照本手册执行，不允许自行改变数据源、计数口径、去重键、抽样数量或配对规则。**

---

## 0. 这份任务最终要得到什么

本任务不是“把几个 benchmark 下载到一起”这么简单。最终要得到的是一个可用于 **Embodied Agent RSI / Experience Learning** 实验的、经过统一规范化和严格防泄漏的数据集合。

原始数据池共 **8,046 条 raw task entries**：

| 数据集 | 原始条数 | 本项目使用范围 |
|---|---:|---|
| EB-ALFRED | 300 | 6 个官方子集，每个 50；Core 中禁止把语言改写当成新物理任务 |
| EB-Habitat | 300 | 6 个官方子集，每个 50；按 episode 物理状态去重 |
| SpatialWorld | 438 | **只用单智能体 AI2-THOR 311 + ProcTHOR 127** |
| AsgardBench | 108 | 只用 `magt_benchmark`，禁止使用 sanity set |
| TVRBench | 6,900 | SFT 1,600 + RL 4,800 + Eval 500 |
| **总计** | **8,046** | 作为完整 raw registry |

最终 **Core v1.0** 固定为 **2,200 个互斥的物理任务实例**：

| Role | 最终条数 | 含义 |
|---|---:|---|
| Experience | **1,500** | Agent 可以实际经历、学习、总结经验的任务 |
| ID Probe | **300** | 与 Experience 同分布/同技能，但不是同一个物理任务 |
| Transfer Probe | **300** | 新场景、新 backend、新配置或新对象上的技能迁移测试 |
| Retention Probe | **100** | 在后续学习之后反复重测的旧技能锚点 |
| **Core 总计** | **2,200** | 四个 role 之间物理任务必须互斥 |

另外必须保留：

- `task_registry_raw.parquet`：**8,046 行**，任何 raw entry 都不能删除；
- `task_registry_normalized.parquet`：**8,046 行**，一一对应 raw entry，但增加统一字段；
- EB-ALFRED / EB-Habitat 的语言改写条目放入 `extensions/language_robustness/`，**不计入 Core 2,200**；
- 所有原始下载内容必须保持只读，禁止在 source 目录里直接改文件。

---

# 1. 不允许违反的硬规则

执行 Agent 在开始前必须把下面规则读完。

## 1.1 禁止自行扩大或缩小数据集范围

本项目只使用以下 5 个来源：

1. EB-ALFRED
2. EB-Habitat
3. SpatialWorld：仅 AI2-THOR + ProcTHOR 单智能体部分
4. AsgardBench
5. TVRBench

禁止加入：

- EmbodiedBench 的 EB-Navigation；
- EmbodiedBench 的 EB-Manipulation；
- SpatialWorld 的 VirtualHome、CARLA、EmbodiedCity、Multi-AI2THOR、Multi-ProcTHOR、3D Games；
- AsgardBench sanity set；
- TVRBench Hugging Face 上的 SFT trajectory 数据；
- 任何模型生成的 rollout / trajectory 数据集；
- 任何第三方重新标注后但无法追溯到原任务的集合。

## 1.2 禁止把“语言改写”当作“新任务”

尤其是 EB-ALFRED：

```text
base(task A)
common_sense(task A)
complex_instruction(task A)
```

如果底层 `task + repeat_idx` 相同，它们是同一个 physical task 的不同 instruction variant。

这些条目：

- 可以保留在 raw registry；
- 可以用于额外的 language robustness 测试；
- **不能被分到不同的 Experience / ID / Transfer / Retention role 中。**

## 1.3 禁止按 8,046 条等比例抽样

TVRBench 占 raw pool 的约 85.8%。如果按 raw 数量比例抽样，Core 会退化成 viewpoint/navigation benchmark。

所以最终必须严格执行第 11 节中的 **固定 source quota + role quota**。

## 1.4 Object 只统计任务语义相关对象

禁止统计 simulator 场景中所有 mesh / visible object instance。

只统计：

- `target/manipulated object class`
- `receptacle/appliance/support class`

两者必须分开存。

TVRBench 的任务目标是 target viewpoint，不是某个物体，因此：

```text
TVR semantic_target_objects = []
```

## 1.5 Skill family 和 action primitive 不是一回事

例如：

```text
Heating / Cooking
```

是一个**技能 family**；它通常由：

```text
Pick -> Place -> Close -> ToggleOn -> ...
```

多个 primitive 组合完成。

禁止把二者混为一个字段。

## 1.6 四个 Core role 必须 physical-level 互斥

最终要求：

```text
Experience physical_task_id
∩ ID physical_task_id
∩ Transfer physical_task_id
∩ Retention physical_task_id
= empty
```

更具体地说，任意两个 role 的 `physical_task_id` 集合交集都必须为 0。

---

# 2. 建议的项目目录

从一个空目录开始：

```bash
mkdir -p embodied_rsi_data
cd embodied_rsi_data
```

最终目录必须整理成：

```text
embodied_rsi_data/
├── sources/                         # 原始上游数据，只读
│   ├── embodiedbench/
│   ├── eb_alfred_hf/
│   ├── spatialworld/
│   ├── asgardbench/
│   └── tvrbench/
│
├── runtime_assets/                  # 运行 simulator 所需的重资产；不计入 task 数
│   ├── eb_habitat/
│   └── README.md
│
├── configs/
│   ├── canonical_primitives.yaml
│   ├── skill_family_rules.yaml
│   └── core_v1_quotas.yaml
│
├── scripts/
│   ├── 00_download_sources.sh
│   ├── 01_validate_raw_sources.py
│   ├── 02_parse_eb_alfred.py
│   ├── 03_parse_eb_habitat.py
│   ├── 04_parse_spatialworld.py
│   ├── 05_parse_asgard.py
│   ├── 06_parse_tvrbench.py
│   ├── 07_merge_registry.py
│   ├── 08_build_physical_groups.py
│   ├── 09_assign_taxonomy.py
│   ├── 10_build_core_split.py
│   ├── 11_build_pairs.py
│   └── 12_validate_release.py
│
├── work/
│   ├── parsed/
│   ├── normalized/
│   └── reports/
│
└── release/
    ├── registry/
    │   ├── task_registry_raw.parquet
    │   ├── task_registry_normalized.parquet
    │   └── physical_groups.parquet
    ├── core_v1/
    │   ├── experience.parquet
    │   ├── experience.jsonl
    │   ├── id_probe.parquet
    │   ├── id_probe.jsonl
    │   ├── transfer_probe.parquet
    │   ├── transfer_probe.jsonl
    │   ├── retention_probe.parquet
    │   ├── retention_probe.jsonl
    │   ├── selected_core.parquet
    │   ├── id_pairs.parquet
    │   ├── transfer_pairs.parquet
    │   └── retention_schedule.json
    ├── extensions/
    │   └── language_robustness/
    │       ├── eb_alfred.parquet
    │       └── eb_habitat.parquet
    ├── source_manifest.json
    ├── stats.json
    ├── README_DATASET.md
    └── VALIDATION_REPORT.md
```

---

# 3. 基础环境

推荐 Python 3.10。

```bash
python --version
```

至少安装：

```bash
pip install pandas pyarrow numpy pyyaml tqdm datasets huggingface_hub
```

检查 Git-LFS：

```bash
git lfs version
```

如果没有：

```bash
# 有 conda 时优先
conda install -y -c conda-forge git-lfs

git lfs install
```

**重要：EB-Habitat 的 `.pickle` 是 Git-LFS 文件。普通 clone 后如果看到每个 pickle 只有约 131 B，那只是 LFS pointer，不能算下载成功。**

全局固定随机种子：

```text
SEED = 20260915
```

不要依赖 Python `random.shuffle()` 的隐式顺序。需要稳定抽样时统一使用：

```python
rank = sha256(f"20260915|{global_task_id}".encode()).hexdigest()
```

按 `rank` 升序排序后取前 N 个。

---

# 4. Phase A：只下载，不处理

这一阶段的规则是：

> **只把 upstream source 完整拿到本地，并确认文件存在。不要开始抽样，不要开始合并，不要先删“重复任务”。**

每个 repo clone 完后都必须记录：

```bash
git rev-parse HEAD
```

写入：

```text
sources/<dataset>/SOURCE_COMMIT.txt
```

这样以后上游仓库更新，也能知道当前 Core v1.0 是基于哪个 commit 构建的。

---

## 4.1 下载 EmbodiedBench 主仓库

来源：

```text
https://github.com/EmbodiedBench/EmbodiedBench.git
```

执行：

```bash
cd sources
git clone https://github.com/EmbodiedBench/EmbodiedBench.git embodiedbench
cd embodiedbench
git lfs install
git lfs pull
git rev-parse HEAD > SOURCE_COMMIT.txt
cd ../..
```

这个 repo 同时提供：

- EB-ALFRED 的 split 定义；
- EB-Habitat 的 6 个 pickle benchmark split；
- 两套 simulator wrapper 与数据结构定义。

### 此阶段必须存在的 EB-ALFRED split 文件

```text
sources/embodiedbench/embodiedbench/envs/eb_alfred/data/splits/splits.json
```

它必须包含 6 个 key：

```text
base
common_sense
complex_instruction
spatial
visual_appearance
long_horizon
```

每个 key 必须是 50 条。

因此 raw split entries：

```text
50 × 6 = 300
```

### 此阶段必须存在的 EB-Habitat 文件

目录：

```text
sources/embodiedbench/embodiedbench/envs/eb_habitat/datasets/
```

Core 只读取下面 6 个：

```text
base.pickle
common_sense.pickle
complex_instruction.pickle
spatial_relationship.pickle
visual_appearance.pickle
long_horizon.pickle
```

**禁止使用：**

```text
long_horizon_old.pickle
train_validation.pickle
```

执行：

```bash
ls -lh sources/embodiedbench/embodiedbench/envs/eb_habitat/datasets/*.pickle
```

如果 6 个目标 pickle 仍然显示约 131 B，说明 Git-LFS 没拉下来：

```bash
cd sources/embodiedbench
git lfs pull
cd ../..
```

然后重新检查。

---

## 4.2 下载 EB-ALFRED 官方 task 数据

官方数据来源：

```text
https://huggingface.co/datasets/EmbodiedBench/EB-ALFRED
```

官方仓库大小约 540 MB。这里需要的是 ALFRED task trajectory metadata，不要下载 `EB-Alfred_trajectory_dataset` 那个模型 rollout 数据集。

执行：

```bash
cd sources
git clone https://huggingface.co/datasets/EmbodiedBench/EB-ALFRED eb_alfred_hf
cd ..
```

如果 Hugging Face 使用 Xet/LFS 导致 clone 不完整，可以改用：

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="EmbodiedBench/EB-ALFRED",
    repo_type="dataset",
    local_dir="sources/eb_alfred_hf",
)
PY
```

### 不要直接数 HF repo 的所有 task 数

HF repo 是 ALFRED task 数据库。我们此次使用的 **300 raw entries 是由 EmbodiedBench 的 `splits.json` 指定的**。

正确验收逻辑：

1. 读取 `splits.json`；
2. 一共应该得到 300 个 split entry；
3. 对每个 entry：

```python
entry["task"]
entry["repeat_idx"]
entry["instruction"]
```

4. 必须在 HF task root 下找到该 task 对应的 `traj_data.json`；
5. 对应 repeat annotation 也必须存在。

典型 ALFRED task 目录：

```text
<task_type>-<object>-<movable_receptacle>-<receptacle>-<scene_num>/trial_.../
```

其中核心 metadata 是：

```text
traj_data.json
```

必须从 `traj_data.json` 读取：

```text
task_id
task_type
pddl_params
scene.floor_plan
scene.scene_num
scene.random_seed
scene.init_action
scene.object_poses
scene.object_toggles
scene.dirty_and_empty
plan.high_pddl
plan.low_actions
```

不要依赖 task folder 名猜 scene；以 `traj_data.json` 为准。

---

## 4.3 EB-Habitat：这次不要下载官方 trajectory rollout 数据集

**不要下载：**

```text
EmbodiedBench/EB-Habitat_trajectory_dataset
```

它约 6.38 GB，内容是模型 rollout，不是我们构建 Core 所需的 benchmark task definition。

Core 的 source-of-truth 是 EmbodiedBench repo 里的 6 个 `.pickle`。

### 为了方便解析 metadata，可额外下载一个“只用于转换/校验”的 JSONL mirror

推荐：

```text
https://huggingface.co/datasets/oscarqjh/EB-Habitat_easi
```

它是从 EB-Habitat 官方 benchmark 转出的 6×50 JSONL，字段包括：

```text
episode_id
instruction
instruct_id
scene_id
sampled_entities
goal_preds
start_preds
subgoals
start_position
start_rotation
rigid_objs
ao_states
name_to_receptacle
sampler_info
```

用途：

- 优先用于**解析 metadata**，避免为了读 pickle 先启动整个 Habitat simulator；
- 但 `source_dataset` 仍然写 `EB-Habitat`，不能写成 EASI；
- 必须验证 6 个 split 都是 50；
- 如果 mirror 与官方 pickle 的 episode_id / split 数不一致，以官方 pickle 为准并立即停止报告。

下载：

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="oscarqjh/EB-Habitat_easi",
    repo_type="dataset",
    local_dir="sources/eb_habitat_metadata_mirror",
)
PY
```

### 真正运行 EB-Habitat 时才需要 simulator assets

如果后续要执行 episode，而不只是生成 registry，再执行官方流程：

```bash
conda activate embench
python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets
```

官方 README 的做法是把得到的 `data/` 放到：

```text
embodiedbench/envs/eb_habitat/data/
```

这些重资产**不计入 300 task 数**。

---

## 4.4 下载 SpatialWorld

来源：

```text
https://github.com/Hongcheng-Gao/SpatialWorld.git
```

执行：

```bash
cd sources
git clone https://github.com/Hongcheng-Gao/SpatialWorld.git spatialworld
cd spatialworld
git rev-parse HEAD > SOURCE_COMMIT.txt
cd ../..
```

本项目只使用：

```text
data/ai2thor/tasks/
data/procthor/tasks/
task_classification_detail.csv
```

### 必须排除

```text
data/ai2thor/dual/
data/procthor/dual/
data/virtualhome/
data/carla/
data/embodiedcity/
data/game/
```

以及 classification CSV 中的：

```text
mutil-ai2thor
mutil-procthor
virtualhome
carla
embodiedcity
...
```

### 预期数量

```text
AI2-THOR single-agent: 311
ProcTHOR single-agent:   127
Total:                   438
```

每个 task 目录至少应该有：

```text
init.json
task.json
```

典型 `task.json` 中会有：

```text
task_id
task_name
instruction
golden_actions.steps
golden_actions.actions
scene
target_object_types
success_conditions
success_logic
target_description
```

`task_classification_detail.csv` 是以下字段的唯一官方来源：

```text
environment
task_id
instruction
category
task_type
```

其中 `task_type` 只允许：

```text
Navigation
Interaction
Hybrid
```

---

## 4.5 下载 AsgardBench

来源：

```text
https://github.com/microsoft/AsgardBench.git
```

执行：

```bash
cd sources
git clone https://github.com/microsoft/AsgardBench.git asgardbench
cd asgardbench
git rev-parse HEAD > SOURCE_COMMIT.txt
cd ../..
```

**唯一允许读取的 benchmark 目录：**

```text
sources/asgardbench/Generated/magt_benchmark/
```

必须正好有：

```text
108 个 task directory
108 个 plan.json
```

**禁止读取：**

```text
Generated/magt_benchmark_sanity/
```

每个 `plan.json` 需要提取：

```text
name
task_description
scene
step_count
initial_pose
goal
setup_actions
object_setup
randomization
```

AsgardBench 官方 12 个 task type 的数量必须满足：

```text
9, 6, 12, 6, 12, 3, 15, 3, 27, 9, 3, 3
```

排序后应为：

```text
3, 3, 3, 3, 6, 6, 9, 9, 12, 12, 15, 27
```

总和：

```text
108
```

12 个 task type 是：

1. Consume coffee from a mug, then wash and store the mug — 9
2. Microwave a potato and serve it in a bowl — 6
3. Cook an egg in a pan and serve it on a plate — 12
4. Make a slice of toast and serve it on a plate — 6
5. Fry a potato slice and serve it on a plate — 12
6. Put away the Food — 3
7. Put away the Dishes — 15
8. Put away the Silverware — 3
9. Slice Lettuce/Apple/Tomato and put a piece in Plate/Bowl/Pan/Pot — 27
10. Clear the dining table and set it with coffee + toast — 9
11. Clean the mirror — 3
12. Turn on the television — 3

---

## 4.6 下载 TVRBench

来源：

```text
https://github.com/aim-uofa/TVRBench.git
```

执行：

```bash
cd sources
git clone https://github.com/aim-uofa/TVRBench.git tvrbench
cd tvrbench
git rev-parse HEAD > SOURCE_COMMIT.txt
cd ../..
```

必须存在：

```text
data/scene_splits.json
data/tasks/sft.json
data/tasks/rl.json
data/tasks/eval.json
data/procthor-10k/train.jsonl.gz
data/procthor-10k/val.jsonl.gz
data/procthor-10k/test.jsonl.gz
```

### 精确数量

```text
sft.json   = 1,600 tasks = 40 scenes
rl.json    = 4,800 tasks = 120 scenes
eval.json  =   500 tasks = 80 scenes
Total      = 6,900 tasks = 240 unique scenes
```

240 scene 构成：

```text
120 iTHOR
120 ProcTHOR
```

`scene_splits.json` 中应为：

```text
iTHOR:    sft=20, rl=60, eval=40
ProcTHOR: sft=20, rl=60, eval=40
```

因此：

```text
Experience-side scenes = 160
Official eval scenes   = 80
```

每个 task 的核心字段：

```text
task_id
dataset          # ithor / procthor
scene
difficulty
seg_count
bfs_distance
target.position
target.rotation_y
target.horizon
start.position
start.rotation_y
start.horizon
```

四个 difficulty stratum：

```text
iTHOR easy
ithor hard
ProcTHOR easy
ProcTHOR hard
```

语义上统一命名为：

```text
SR-easy
SR-hard
LR-easy
LR-hard
```

---

# 5. Phase A 总验收：必须得到 8,046 条 raw entry

写 `scripts/01_validate_raw_sources.py`。

脚本必须输出：

```text
EB-ALFRED     300
EB-Habitat    300
SpatialWorld  438
AsgardBench   108
TVRBench     6900
-----------------
TOTAL        8046
```

任何一项不等于上面的数字：

```text
立即退出，exit code != 0
```

禁止“差几条也继续”。

同时生成：

```text
work/reports/RAW_SOURCE_VALIDATION.json
```

字段至少包括：

```json
{
  "eb_alfred": 300,
  "eb_habitat": 300,
  "spatialworld_ai2thor": 311,
  "spatialworld_procthor": 127,
  "asgardbench": 108,
  "tvr_sft": 1600,
  "tvr_rl": 4800,
  "tvr_eval": 500,
  "grand_total": 8046
}
```

---

# 6. 建立统一 raw registry：8,046 行，一个都不能少

每个 source parser 最后都输出一个 parquet：

```text
work/parsed/eb_alfred.parquet       300
work/parsed/eb_habitat.parquet      300
work/parsed/spatialworld.parquet     438
work/parsed/asgardbench.parquet      108
work/parsed/tvrbench.parquet        6900
```

合并后：

```text
release/registry/task_registry_raw.parquet = 8046 rows
```

## 6.1 统一字段 schema

每一行至少必须有这些字段：

```text
global_raw_id                  string
source_dataset                 string
source_split                   string|null
source_backend                 string
source_task_id                 string
source_entry_index             int|null
source_path                    string
instruction                    string|null
native_task_type               string|null
native_category                string|null
scene_id_raw                   string|null
scene_id_canonical             string|null
physical_task_id               string
language_variant_group_id      string|null
is_language_rewrite            bool
skill_family_primary           string|null
skill_family_set               list[string]
primitive_set                  list[string]
target_object_classes          list[string]
receptacle_classes             list[string]
appliance_classes              list[string]
difficulty_native              string|null
difficulty_canonical           string|null
golden_step_count              int|null
raw_metadata_json              string
eligible_core                  bool
eligible_experience            bool
eligible_id                    bool
eligible_transfer              bool
eligible_retention             bool
exclusion_reason               string|null
```

## 6.2 `global_raw_id`

必须稳定，可重复生成。

建议：

```python
global_raw_id = sha256(
    f"{source_dataset}|{source_split}|{source_task_id}|{source_entry_index}".encode()
).hexdigest()[:24]
```

不得使用 DataFrame row number 作为唯一 ID，因为重新排序后会变。

---

# 7. 每个数据集到底抽哪些字段

## 7.1 EB-ALFRED parser

输入：

```text
splits.json
+
HF task directories / traj_data.json
```

### raw row 粒度

`split entry` 是 raw row。

所以 6×50 = 300 行，即使不同 split 指向同一个 physical task，也仍然保留 300 行。

### `source_task_id`

用：

```text
<task_path>|repeat_idx=<N>
```

### `physical_task_id`

EB-ALFRED 最重要的防泄漏键：

```python
physical_key = {
    "task": split_entry["task"],
    "repeat_idx": split_entry["repeat_idx"],
}
physical_task_id = "eba:" + sha256(canonical_json(physical_key)).hexdigest()[:24]
```

不要把 `instruction` 放进 physical key。

因为同一物理任务换一种说法仍然是同一任务。

### scene

从：

```python
traj_data["scene"]["floor_plan"]
```

读取，统一：

```text
source_backend = ai2thor
scene_id_canonical = ai2thor::<floor_plan>
```

例如：

```text
ai2thor::FloorPlan19
```

### native task type

从：

```python
traj_data["task_type"]
```

必须属于 ALFRED 7 类之一，例如：

```text
pick_and_place_simple
pick_two_obj_and_place
pick_and_place_with_movable_recep
pick_clean_then_place_in_recep
pick_heat_then_place_in_recep
pick_cool_then_place_in_recep
look_at_obj_in_light
```

### 对象字段

从：

```python
pddl = traj_data["pddl_params"]
```

提取：

```text
object_target      -> target_object_classes
mrecep_target      -> receptacle_classes / movable receptacle
parent_target      -> receptacle_classes
 toggle_target     -> appliance_classes
```

如果 `object_sliced=true`，仍保留基础对象类，同时可在 raw metadata 中标 `sliced=true`。

### primitive

优先从：

```python
traj_data["plan"]["high_pddl"]
```

读取 high-level action 名称，再按第 9 节映射到 canonical primitive。

### language rewrite 标记

构建：

```text
language_variant_group_id = physical_task_id
```

如果同一个 `physical_task_id` 出现在多个 split，所有相关行：

```text
is_language_rewrite = true
```

特别注意：

```text
base / common_sense / complex_instruction
```

经常存在这种重用。

### Core eligibility

Core v1.0 默认只允许这 4 个 split：

```text
base
spatial
visual_appearance
long_horizon
```

下面两个 split 默认：

```text
common_sense
complex_instruction
```

设置：

```text
eligible_core = false
exclusion_reason = language_rewrite_extension
```

并输出到：

```text
release/extensions/language_robustness/eb_alfred.parquet
```

该 extension raw 行数应为：

```text
100
```

但其中 physical group 数可能小于 100，这是正常的。

---

## 7.2 EB-Habitat parser

输入优先级：

1. 官方 6 个 pickle 是 source-of-truth；
2. `EB-Habitat_easi` JSONL 只用于易读 metadata 转换；
3. 若两者 episode_id / split 数冲突，停止执行。

### 必须得到

每个 split 50：

```text
base                  50
common_sense          50
complex_instruction   50
spatial_relationship  50
visual_appearance     50
long_horizon          50
Total                 300
```

### row 字段

至少读取：

```text
episode_id
instruction
instruct_id
scene_id
sampled_entities
goal_preds
start_preds
subgoals
start_position
start_rotation
sampler_info
```

### scene

```text
source_backend = habitat_replicacad
scene_id_canonical = habitat_replicacad::<basename(scene_id)>
```

### physical task key

不能只用 instruction，也不要只用 episode_id。

构造一个 canonical state signature：

```python
physical_key = {
    "scene_id": canonical_scene_id,
    "sampled_entities": sort_recursively(sampled_entities),
    "start_preds": sorted(start_preds),
    "goal_preds": sort_recursively(goal_preds),
    "start_position": round_list(start_position, 4),
    "start_rotation": round_list(start_rotation, 4),
}
physical_task_id = "ebh:" + sha256(canonical_json(physical_key)).hexdigest()[:24]
```

### object / receptacle

不要扫描 ReplicaCAD 全场景。

只从：

```text
sampled_entities
goal_preds
start_preds
```

提取任务涉及的：

- manipulated object class；
- source receptacle；
- target receptacle；
- fridge / cabinet 等 interactive receptacle。

分别写入：

```text
target_object_classes
receptacle_classes
appliance_classes
```

### native primitive

EB-Habitat 统一成以下 5 大语义类：

```text
Navigation
Pick
Place
Open
Close
```

虽然原环境有约 70 个 parameterized discrete skills，但 canonical primitive 不按参数实例数计算。

### Core eligibility

和 EB-ALFRED 一样，默认只让：

```text
base
spatial_relationship
visual_appearance
long_horizon
```

进入 Core 候选。

把：

```text
common_sense
complex_instruction
```

放入：

```text
release/extensions/language_robustness/eb_habitat.parquet
```

extension raw 数应为：

```text
100
```

---

## 7.3 SpatialWorld parser

只遍历：

```text
data/ai2thor/tasks/*
data/procthor/tasks/*
```

必须得到：

```text
311 + 127 = 438
```

对每一个 task：

1. 读 `task.json`；
2. 读 `init.json`；
3. 去 `task_classification_detail.csv` 找同 `task_id` 的唯一一行；
4. environment 必须对应 `ai2thor` 或 `procthor`；
5. 如果 classification 找不到或找到多行，立即报错。

### physical task id

```text
SpatialWorld 的 task_id 本身就是任务实例 ID。
```

构造：

```text
physical_task_id = spatial::<environment>::<task_id>
```

### scene canonicalization

AI2-THOR：

```text
ai2thor::FloorPlanXX
```

ProcTHOR：

```text
procthor::<house_id_or_scene_id>
```

不要把 `procthor107` 这种 task_id 错当 scene id。

scene 必须从 `task.json/init.json` 的真实环境配置读。

### object

优先：

```python
task_json["target_object_types"]
```

再结合 `success_conditions` 判断：

- 哪些是 manipulated objects；
- 哪些是 receptacles；
- 哪些是 appliances/stateful objects。

不要仅靠 instruction NER。

### primitive

从：

```python
task_json["golden_actions"]["actions"]
```

解析。

SpatialWorld upstream 的 unified action interface 有：

```text
Move
Rotate
Tilt
ChangePosture
Pick
Place
ChangeState
Manipulate
EndTask
Communicate
```

本项目只用 single-agent，因此 `Communicate` 通常不会出现；如果出现，也保留 raw，但不要把它误映射为 embodied manipulation primitive。

### native task type

从 classification CSV 读取：

```text
Navigation / Interaction / Hybrid
```

---

## 7.4 AsgardBench parser

只遍历：

```text
Generated/magt_benchmark/*/plan.json
```

必须 108。

### physical task id

每个 `plan.json` 是一个受控 task instance。

构造：

```python
physical_key = {
    "name": plan["name"],
    "scene": plan["scene"],
    "initial_pose": plan["initial_pose"],
    "goal": plan["goal"],
    "setup_actions": plan["setup_actions"],
    "object_setup": plan["object_setup"],
}
```

hash 后加前缀 `asgard:`。

### scene

统一成：

```text
ai2thor::<FloorPlan...>
```

### task type

不要简单用目录字符串第一个 `_` 前缀；必须根据 12 个官方 task description template 归类。

最终 12 类数量必须严格等于：

```text
9, 6, 12, 6, 12, 3, 15, 3, 27, 9, 3, 3
```

### object

核心 manipulated/goal class 应主要来自：

```text
Apple
Bread
Egg
Lettuce
Potato
Tomato
Bowl
Cup
Mug
Plate
Pot
Pan
ButterKnife
Fork
Knife
Ladle
Spatula
Spoon
Mirror
Television
```

这是 20 个核心 goal/manipulated semantic classes。

但 `Fridge / Microwave / CoffeeMachine / Faucet / SinkBasin / Cabinet / Drawer / CounterTop ...` 必须放在 receptacle/appliance 字段，不要混成 20 个目标类。

### primitive

Asgard 的 12 个原生动作：

```text
CLEAN
CLOSE
EMPTY
DRINK
FIND
OPEN
PICKUP
PUT
SLICE
SPRAY
TOGGLE_ON
TOGGLE_OFF
```

映射到第 9 节的 canonical primitive。

---

## 7.5 TVRBench parser

直接读三个 JSON 数组：

```text
sft.json  1600
rl.json   4800
eval.json  500
```

### physical task id

TVR task instance 由：

```text
scene + target pose + start pose
```

唯一确定。

建议：

```python
physical_key = {
    "dataset": row["dataset"],
    "scene": row["scene"],
    "target": canonical_pose(row["target"]),
    "start": canonical_pose(row["start"]),
}
```

不要只用 target，因为同一 target 可有多个不同 start。

### scene

```text
ithor     -> ai2thor::<scene>
procthor  -> procthor::<scene_or_house_index>
```

### instruction

TVR 没有普通 household instruction。

不要伪造物体任务。

允许：

```text
instruction = null
native_task_type = target_viewpoint_reproduction
```

### object

必须：

```text
target_object_classes = []
```

### native primitive

9 个 action labels：

```text
MoveAhead
MoveBack
MoveLeft
MoveRight
RotateRight
RotateLeft
LookUp
LookDown
Stop
```

canonical 化后主要变为：

```text
Locomotion / Search
Viewpoint / Orientation
Terminate
```

---

# 8. Physical dedup：保留 8,046 行，但建立物理任务分组

执行：

```text
scripts/08_build_physical_groups.py
```

输出：

```text
release/registry/physical_groups.parquet
```

至少包含：

```text
physical_task_id
source_dataset
member_raw_ids
member_count
num_distinct_instructions
num_distinct_splits
is_cross_split_duplicate
```

重要：

- raw registry 仍然是 8,046 行；
- dedup 的意义不是删除 raw row；
- dedup 的意义是**防止同一物理任务跨 role 泄漏**。

验收：

```text
sum(member_count) = 8046
```

并且 EB-ALFRED 必须实际出现：

```text
member_count > 1
```

的 physical group；如果一个都没有，说明 parser 或 physical key 写错了。

---

# 9. 统一 taxonomy

## 9.1 12 个 reusable skill families

固定为：

```text
S01 Search / Navigation
S02 Active Viewpoint / Orientation
S03 Pick / Grasp
S04 Place / Relocation
S05 Receptacle Access
S06 Toggle / Device State
S07 Cleaning / Washing
S08 Heating / Cooking
S09 Cooling
S10 Cutting / Slicing
S11 Liquid Handling
S12 Multi-object / Long-horizon Composition
```

一条 task 可以有多个 family。

例如：

```text
“slice tomato and put one piece in pan”
```

可以是：

```text
[S03, S04, S10]
```

如果还是多物体/长轨迹，也可附加 S12。

`skill_family_primary` 只作为配对主索引；完整信息必须保留在 `skill_family_set`。

## 9.2 13 个 canonical primitive classes

固定为：

```text
P01 Locomotion / Search
P02 Viewpoint / Orientation
P03 Posture Change
P04 Pick / Grasp
P05 Place / Drop
P06 Open
P07 Close
P08 Toggle / State Change
P09 Slice / Cut
P10 Clean
P11 Liquid Interaction
P12 Generic Manipulation
P13 Terminate
```

## 9.3 明确映射示例

```text
EB-ALFRED Find/GotoLocation     -> P01
EB-ALFRED PickupObject          -> P04
EB-ALFRED PutObject             -> P05
EB-ALFRED OpenObject            -> P06
EB-ALFRED CloseObject           -> P07
EB-ALFRED ToggleObjectOn/Off    -> P08
EB-ALFRED SliceObject           -> P09

EB-Habitat Navigation           -> P01
EB-Habitat Pick                 -> P04
EB-Habitat Place                -> P05
EB-Habitat Open                 -> P06
EB-Habitat Close                -> P07

Spatial Move                    -> P01
Spatial Rotate/Tilt             -> P02
Spatial ChangePosture           -> P03
Spatial Pick                    -> P04
Spatial Place                   -> P05
Spatial ChangeState(open)       -> P06
Spatial ChangeState(close)      -> P07
Spatial ChangeState(on/off)     -> P08
Spatial ChangeState(sliced)     -> P09
Spatial ChangeState(clean)      -> P10
Spatial Manipulate(...)         -> P12
Spatial EndTask                 -> P13

Asgard FIND                     -> P01
Asgard PICKUP                   -> P04
Asgard PUT                      -> P05
Asgard OPEN                     -> P06
Asgard CLOSE                    -> P07
Asgard TOGGLE_ON/OFF            -> P08
Asgard SLICE                    -> P09
Asgard CLEAN                    -> P10
Asgard EMPTY/DRINK/SPRAY        -> P11

TVR Move*                       -> P01
TVR Rotate*/Look*               -> P02
TVR Stop                        -> P13
```

### 注意

`Heating / Cooking`、`Cooling` 等不要硬塞成 primitive。

它们应由：

- native task type；
- goal predicate；
- success condition；
- high-level task template；

来赋值为 skill family。

---

# 10. Core v1.0 固定 source quota

最终按下面表构造，不允许自行改数字。

| Dataset | Experience | ID | Transfer | Retention | Core 合计 |
|---|---:|---:|---:|---:|---:|
| TVRBench | **1080** | **160** | **140** | **28** | **1408** |
| SpatialWorld | **220** | **70** | **100** | **20** | **410** |
| AsgardBench | **46** | **46** | **0** | **12** | **104** |
| EB-ALFRED | **75** | **12** | **30** | **20** | **137** |
| EB-Habitat | **79** | **12** | **30** | **20** | **141** |
| **总计** | **1500** | **300** | **300** | **100** | **2200** |

为什么不让 TVR 按 raw 占比进入：

```text
raw: 6900 / 8046 = 85.8%
core: 1408 / 2200 = 64.0%
```

它仍然是最大来源，但不会完全统治 benchmark。

### 如果某个 source 在严格 dedup 后达不到 quota

**不要自行从其他数据集补。**

必须：

1. 输出 `BLOCKED_QUOTA_SHORTFALL.json`；
2. 写清：source、role、需要多少、可用多少；
3. 停止整个 Core 构建；
4. 等人工调整规范。

不要让 Agent 自己“聪明地补齐”。

---

# 11. 每个数据集具体怎么切 Experience / ID / Transfer / Retention

这是整个任务最重要的一节。

---

## 11.1 TVRBench：1080 / 160 / 140 / 28

### Step 1：先冻结 official scene split

从 `scene_splits.json` 读，不允许重新随机划 scene。

```text
sft + rl scenes = 160 个 experience-side scene
eval scenes     = 80 个 official OOD scene
```

### Step 2：选 28 Retention

四个 difficulty stratum 各 7 条：

```text
SR-easy  7
SR-hard  7
LR-easy  7
LR-hard  7
Total   28
```

来源只能是 sft/rl side，不能使用 eval。

要求：

- 28 个 physical_task_id 全部唯一；
- 尽量覆盖不同 scene；
- 用固定 hash rank 决定，不人工挑。

### Step 3：选 160 ID

对 **160 个 experience-side scene，每个 scene 恰好拿 1 条**未被 Retention 占用的 task 作为 ID。

因此：

```text
ID count = 160
```

同 scene 的其它 task 仍可作为 Experience。

这使 TVR 的 ID 含义非常清楚：

> 同一个环境分布/scene universe 中，从未经历过的 start-target task instance。

### Step 4：选 140 Transfer

只从官方 `eval.json` 选。

四档各 35 条：

```text
SR-easy 35
SR-hard 35
LR-easy 35
LR-hard 35
Total  140
```

由于 eval scene 与 sft/rl scene 官方就是 disjoint，这 140 条天然是 clean scene-OOD transfer。

### Step 5：选 1080 Experience

只能从：

```text
sft + rl
```

中扣除已选 ID / Retention 后抽。

要求四个 difficulty stratum 尽量严格平衡：

```text
270 × 4 = 1080
```

同时尽量保持原 SFT:RL 的 1:3 比例：

```text
约 270 SFT
约 810 RL
```

如果为了四档整数分配有 ±1 的误差可以，但总数必须：

```text
1080
```

---

## 11.2 SpatialWorld：220 / 70 / 100 / 20

这是最适合做 cross-backend transfer 的一组。

原始：

```text
AI2-THOR 311
ProcTHOR  127
```

### AI2-THOR 的 311 条这样分

```text
Experience  220
ID           70
Retention    20
Reserve       1
----------------
Total        311
```

这非常方便，所以禁止把 AI2-THOR 的 task 用作 SpatialWorld Transfer target。

### ProcTHOR 的 127 条这样分

```text
Transfer  100
Reserve    27
---------------
Total     127
```

### Experience 220

从 AI2-THOR 中按：

```text
task_type: Navigation / Interaction / Hybrid
category: Daily / Work / Entertain / Travel
```

做分层抽样。

不要简单按 task_id 前 220。

### Retention 20

先从 AI2-THOR 抽 20，要求：

- 三种 task_type 都有；
- 尽量覆盖不同 scene；
- 尽量覆盖不同 primary skill family。

推荐目标：

```text
Navigation   7
Interaction  7
Hybrid       6
```

若某一类实际不足，则用 proportional largest-remainder 分配，但总数必须 20。

### ID 70

从剩余 AI2-THOR task 中抽 70。

每个 ID probe 必须匹配到至少一个 Experience anchor，规则：

```text
same source_dataset = SpatialWorld
same backend = ai2thor
same native task_type
physical_task_id different
prefer scene_id different
primary skill family same
```

### Transfer 100

只从 ProcTHOR 127 中选 100。

每个 transfer probe 要匹配到 AI2-THOR Experience：

最低要求：

```text
Experience backend = ai2thor
Transfer backend   = procthor
native task_type same
skill_family_primary same
physical task different
```

优先条件：

```text
primitive Jaccard >= 0.5
object class 至少有一个交集
```

Navigation task 无语义 object 时，可以跳过 object-intersection 条件。

这 100 条定义为：

> **cross-backend transfer**

而不是普通 ID。

---

## 11.3 AsgardBench：46 / 46 / 0 / 12

Asgard 的 108 条全部是受控 variation，最适合做 clean ID 与 retention。

### 对每个 12 task type 单独操作

对 family 内 task 按 deterministic hash rank 排序。

### Step 1：每个 family 取 1 个 Retention

```text
12 families × 1 = 12
```

### Step 2：剩余每个 family 两两配成 Experience-ID

对 family 大小 `n_f`：

```python
k = floor((n_f - 1) / 2)
```

选：

```text
k Experience
k ID
```

所有 family 合计应得到：

```text
46 Experience
46 ID
```

还会剩下：

```text
4 个 reserve
```

因此：

```text
46 + 46 + 12 + 4 = 108
```

### Pair 的含义

Asgard 同一个 task family 中变化的是：

- object cleanliness；
- fill state；
- object placement；
- scene configuration；
- sink/cabinet 等初始状态。

所以 ID pair 用于测：

> 是否学习到 task rule / conditional plan，而不是背某条固定 trajectory。

Asgard 本版本不承担 Transfer quota：

```text
Transfer = 0
```

---

## 11.4 EB-ALFRED：75 / 12 / 30 / 20

先做 physical dedup。

Core 候选只来自：

```text
base
spatial
visual_appearance
long_horizon
```

也就是最多 200 raw rows，再按 `physical_task_id` 去重后得到候选 physical pool。

**如果去重后 `<137` 个 physical task，立即停止，不允许使用 common_sense / complex_instruction 偷补 quota。**

### Retention 20

先选。

要求 7 个 ALFRED native task type 都被覆盖。

建议：

- 每个 native task type 至少 2 个；
- 剩余 6 个按 family 数量 proportional 分配；
- scene 尽量不同。

### Transfer 30

优先从：

```text
spatial
visual_appearance
long_horizon
```

中选。

一个 Transfer probe 必须满足：

```text
和 Experience 同 primary skill family
physical_task_id 不同
scene_id_canonical 不同
不能属于同一个 language_variant_group_id
```

优先再满足：

```text
target object class 发生变化
或 receptacle class 发生变化
```

### ID 12

要求：

```text
same native task type
same primary skill family
physical task different
同 backend=ai2thor
```

允许不同 scene，但不要求一定不同。

### Experience 75

从剩余 physical pool 中做 skill-family 分层抽样。

禁止只拿 `base` 前 75 条。

---

## 11.5 EB-Habitat：79 / 12 / 30 / 20

Core 候选只来自：

```text
base
spatial_relationship
visual_appearance
long_horizon
```

最大 200 raw rows，按 physical state key 去重。

如果去重后 `<141`：

```text
停止，不允许用 language rewrite split 补。
```

### Retention 20

优先覆盖：

```text
Navigation
Pick
Place
Open
Close
```

以及不同 `instruct_id`、不同 ReplicaCAD scene。

### Transfer 30

至少满足：

```text
primary skill family same
physical task different
scene different
sampled_entities 不完全相同
```

优先从：

```text
spatial_relationship
visual_appearance
long_horizon
```

中选择作为 probe。

### ID 12

要求：

```text
same backend = habitat_replicacad
same primary skill family
same/similar instruction template or instruct_id family
physical state different
```

### Experience 79

从剩余 pool 中分层抽样，尽量平衡：

- scene；
- target object；
- target receptacle；
- skill family。

---

# 12. ID pair 构造：必须正好 300 对

输出：

```text
release/core_v1/id_pairs.parquet
```

必须正好 300 行。

字段：

```text
pair_id
experience_task_id
probe_task_id
source_dataset
pair_type = ID
shared_skill_family
shared_native_type
experience_scene
probe_scene
experience_objects
probe_objects
matching_rule
```

### source 分布

```text
TVRBench       160
SpatialWorld    70
AsgardBench     46
EB-ALFRED       12
EB-Habitat      12
------------------
Total          300
```

### 必须验证

每个 ID pair：

```text
experience_task_id != probe_task_id
physical_task_id different
skill_family_primary compatible
probe role == ID
anchor role == Experience
```

EB-ALFRED / EB-Habitat 额外要求：

```text
language_variant_group_id 不能相同
```

---

# 13. Transfer pair 构造：必须正好 300 对

输出：

```text
release/core_v1/transfer_pairs.parquet
```

必须 300 行。

source 分布：

```text
TVRBench       140
SpatialWorld   100
EB-ALFRED       30
EB-Habitat      30
AsgardBench      0
------------------
Total          300
```

字段至少：

```text
pair_id
experience_task_id
probe_task_id
source_dataset
transfer_axis
experience_backend
probe_backend
experience_scene
probe_scene
shared_skill_family
primitive_jaccard
object_jaccard
matching_rule
```

### `transfer_axis` 只允许

```text
scene_ood
backend_ood
object_transfer
state_or_configuration_transfer
```

本版本典型：

```text
TVR           -> scene_ood
SpatialWorld  -> backend_ood
EB-ALFRED     -> scene/object/config transfer
EB-Habitat    -> scene/object/config transfer
```

---

# 14. Retention Probe：100 个固定锚点

输出：

```text
release/core_v1/retention_probe.parquet
```

source 分布：

```text
TVRBench        28
SpatialWorld    20
AsgardBench     12
EB-ALFRED       20
EB-Habitat      20
------------------
Total          100
```

这些 task 在整个实验期间保持不变。

不要每轮重新抽。

另外写：

```text
retention_schedule.json
```

初始可以定义：

```json
{
  "anchors": 100,
  "checkpoints": [
    "pre_experience",
    "after_stage_1",
    "after_stage_2",
    "after_stage_3",
    "final"
  ]
}
```

注意：

```text
100 个 retention anchors × 5 次测量
```

是 500 次 evaluation measurements，**不是 500 个 unique tasks**。

---

# 15. Experience Pool：1,500 条

输出：

```text
experience.parquet
experience.jsonl
```

精确 source 数量：

```text
TVRBench       1080
SpatialWorld    220
AsgardBench      46
EB-ALFRED        75
EB-Habitat       79
-------------------
Total          1500
```

所有 Experience 都应该保存：

```text
source locator
scene
objects
skills
primitives
native task metadata
```

但**不要把 golden answer/trajectory 直接暴露为 agent evaluation prompt 的一部分**。

`golden_actions`、Asgard goal 等字段可以作为 evaluator metadata 保存，但必须单独标：

```text
private_eval_metadata
```

后续跑 agent 时，不得拼入 observation/prompt。

---

# 16. 统一 scene 统计规则

scene namespace 必须带 backend 前缀：

```text
ai2thor::FloorPlan11
procthor::<house-id>
habitat_replicacad::<scene-instance>
```

禁止只存：

```text
FloorPlan11
```

然后把不同 simulator 的同名字符串误认为同一 scene。

### TVR 已知精确 scene 数

```text
240
```

其中：

```text
120 iTHOR
120 ProcTHOR
```

### 全数据集总 unique scene 数

**不要预先硬编码。**

必须在 8,046 行 registry 建完后计算：

```python
n_unique_scenes = df.scene_id_canonical.dropna().nunique()
```

原因：

- EB-ALFRED / Asgard / Spatial 的 iTHOR 会互相重叠；
- SpatialWorld ProcTHOR 和 TVR ProcTHOR 也可能重叠；
- EB-Habitat 是单独的 ReplicaCAD namespace。

把最终精确数写入：

```text
release/stats.json
```

---

# 17. Object 统计规则

最后至少输出两个集合：

```text
unique_target_object_classes
unique_receptacle_appliance_classes
```

不要混成一个总数。

### TVR

```text
target_object_classes = []
```

### Asgard sanity check

Asgard 核心 manipulated/goal semantic class 应至少能够恢复这 20 类：

```text
Apple, Bread, Egg, Lettuce, Potato, Tomato,
Bowl, Cup, Mug, Plate, Pot, Pan,
ButterKnife, Fork, Knife, Ladle, Spatula, Spoon,
Mirror, Television
```

如果 parser 连这些都抽不出来，说明 object extraction 有问题。

---

# 18. 输出 statistics

生成：

```text
release/stats.json
```

至少包含：

```json
{
  "raw_entries": 8046,
  "core_tasks": 2200,
  "experience": 1500,
  "id_probe": 300,
  "transfer_probe": 300,
  "retention_probe": 100,
  "source_counts_raw": {},
  "source_counts_core": {},
  "physical_group_count": 0,
  "scene_count_total": 0,
  "scene_count_by_backend": {},
  "target_object_class_count": 0,
  "receptacle_appliance_class_count": 0,
  "skill_family_count": 12,
  "primitive_class_count": 13,
  "id_pair_count": 300,
  "transfer_pair_count": 300
}
```

其中 `0` 是占位，程序运行时必须填成实际值。

---

# 19. 最终泄漏检查

`scripts/12_validate_release.py` 必须做下面全部检查。

## 19.1 行数

```python
assert len(raw_registry) == 8046
assert len(normalized_registry) == 8046
assert len(experience) == 1500
assert len(id_probe) == 300
assert len(transfer_probe) == 300
assert len(retention_probe) == 100
assert len(selected_core) == 2200
assert len(id_pairs) == 300
assert len(transfer_pairs) == 300
```

## 19.2 source quota

必须精确：

```text
TVRBench      1408
SpatialWorld   410
AsgardBench    104
EB-ALFRED      137
EB-Habitat     141
```

## 19.3 role/source 交叉表

必须精确等于：

```text
                EXP    ID   TRANSFER   RET
TVRBench       1080   160      140      28
SpatialWorld    220    70      100      20
AsgardBench      46    46        0      12
EB-ALFRED        75    12       30      20
EB-Habitat       79    12       30      20
TOTAL          1500   300      300     100
```

## 19.4 physical leakage

```python
roles = {
    "exp": set(experience.physical_task_id),
    "id": set(id_probe.physical_task_id),
    "transfer": set(transfer_probe.physical_task_id),
    "ret": set(retention_probe.physical_task_id),
}
for a, b in all_role_pairs:
    assert len(roles[a] & roles[b]) == 0
```

同时：

```python
assert selected_core.physical_task_id.nunique() == 2200
```

## 19.5 EB language leakage

对 EB-ALFRED / EB-Habitat：

同一个 `language_variant_group_id` 不允许出现在两个 core role 中。

更严格：

```text
若某 language group 的 physical task 已进入 Core，
其其它 instruction rewrite 只能留在 extension，不得再次进入 Core。
```

## 19.6 TVR scene leakage

必须检查：

```text
TVR Transfer scene ∩ TVR Experience-side scene = empty
```

因为 Transfer 必须来自 official eval scenes。

## 19.7 Spatial backend leakage

SpatialWorld：

```text
Experience backend = ai2thor
ID backend         = ai2thor
Retention backend  = ai2thor
Transfer backend   = procthor
```

必须严格成立。

## 19.8 Asgard

必须：

```text
12 retention = 12 families 各 1 个
46 experience
46 ID
4 reserve
```

---

# 20. 最终生成 VALIDATION_REPORT.md

不要只打印一句 `PASS`。

必须写完整报告：

```text
1. Source commits
2. Raw source counts
3. Missing file check
4. Git-LFS check
5. Raw registry = 8046
6. Physical group statistics
7. Duplicate/language rewrite statistics
8. Unique scenes by backend
9. Target object class statistics
10. Receptacle/appliance class statistics
11. 12 skill family coverage
12. 13 primitive coverage
13. Core role/source cross-table
14. ID pair statistics
15. Transfer pair statistics
16. Leakage checks
17. Reserve pool counts
18. Final PASS/FAIL
```

最终报告最后一行只有全部 assert 都通过时才能写：

```text
FINAL_STATUS: PASS
```

任何 assert 失败：

```text
FINAL_STATUS: FAIL
```

并返回非 0 exit code。

---

# 21. Agent 每完成一个阶段必须产出的中间文件

不要只靠 terminal log。

## 下载阶段

```text
work/reports/SOURCE_DOWNLOAD_REPORT.md
```

## Raw 验收

```text
work/reports/RAW_SOURCE_VALIDATION.json
```

## Registry

```text
release/registry/task_registry_raw.parquet
release/registry/task_registry_normalized.parquet
release/registry/physical_groups.parquet
```

## Taxonomy

```text
work/reports/SKILL_FAMILY_COUNTS.csv
work/reports/PRIMITIVE_COUNTS.csv
work/reports/OBJECT_COUNTS.csv
work/reports/SCENE_COUNTS.csv
```

## Core split

```text
release/core_v1/selected_core.parquet
work/reports/CORE_SPLIT_COUNTS.csv
```

## Pairs

```text
release/core_v1/id_pairs.parquet
release/core_v1/transfer_pairs.parquet
```

## 最终

```text
release/stats.json
release/README_DATASET.md
release/VALIDATION_REPORT.md
```

---

# 22. 推荐的执行顺序：不要跳步

Agent 必须严格按下面顺序：

```text
[01] 建目录
 ↓
[02] clone EmbodiedBench + git lfs pull
 ↓
[03] 下载 EB-ALFRED HF task data
 ↓
[04] 可选下载 EB-Habitat JSONL metadata mirror
 ↓
[05] clone SpatialWorld
 ↓
[06] clone AsgardBench
 ↓
[07] clone TVRBench
 ↓
[08] 运行 raw source validator
      必须得到 8046
 ↓
[09] 解析 5 个 source
 ↓
[10] merge raw registry
      必须 8046 rows
 ↓
[11] 构建 physical_task_id
 ↓
[12] 检测 language rewrite / duplicate groups
 ↓
[13] 统一 scene / object / skill / primitive taxonomy
 ↓
[14] 生成 normalized registry
      仍然必须 8046 rows
 ↓
[15] 先选 Retention
 ↓
[16] 再选 ID
 ↓
[17] 再选 Transfer
 ↓
[18] 最后填 Experience
 ↓
[19] 生成 ID pairs
      300
 ↓
[20] 生成 Transfer pairs
      300
 ↓
[21] 全量 leakage audit
 ↓
[22] 生成 release
      Core=2200
 ↓
[23] 生成 VALIDATION_REPORT.md
```

**为什么一定要先 Retention / ID / Transfer，再选 Experience：**

因为 probe 更难满足约束。如果先把最好的 task 全塞进 Experience，后面很容易发现没有足够干净的 probe 可以配。

---

# 23. 出错时 Agent 应该怎么处理

以下情况禁止自动修：

### 情况 A：raw count 不匹配

例如：

```text
SpatialWorld ai2thor = 310，而不是 311
```

处理：

```text
STOP
```

检查上游下载和分支，不要偷偷把总数改成 437。

### 情况 B：EB-Habitat pickle 131 B

说明 LFS pointer 未解析。

处理：

```bash
git lfs pull
```

不是用 `pickle.load()` 硬读。

### 情况 C：EB-ALFRED 同一 physical task 出现在多个 split

这是预期现象。

处理：

```text
保留 raw rows
统一 physical_task_id
禁止跨 core role
```

不是删掉其中一个 raw row。

### 情况 D：Core candidate 不够 quota

处理：

```text
输出 BLOCKED_QUOTA_SHORTFALL.json
停止
```

禁止从其它 source 自行补。

### 情况 E：Transfer 找不到足够 matched experience anchor

先检查：

1. primary family 是否映射过窄；
2. primitive parser 是否漏动作；
3. object extraction 是否错误；

但**不要为了凑数把不同技能硬配在一起。**

### 情况 F：最终不是 2,200 个 unique physical tasks

直接 FAIL。

---

# 24. 最终验收清单

执行 Agent 完成后，逐条打勾。

## Source

- [ ] EmbodiedBench repo 已 clone
- [ ] Git-LFS 已 pull
- [ ] EB-ALFRED HF task data 已下载
- [ ] SpatialWorld 已 clone
- [ ] AsgardBench 已 clone
- [ ] TVRBench 已 clone
- [ ] 所有 source commit 已记录

## Raw count

- [ ] EB-ALFRED = 300
- [ ] EB-Habitat = 300
- [ ] Spatial AI2-THOR = 311
- [ ] Spatial ProcTHOR = 127
- [ ] Spatial total = 438
- [ ] Asgard = 108
- [ ] TVR SFT = 1600
- [ ] TVR RL = 4800
- [ ] TVR Eval = 500
- [ ] TVR total = 6900
- [ ] Grand total = 8046

## Registry

- [ ] raw registry = 8046
- [ ] normalized registry = 8046
- [ ] every row has stable global_raw_id
- [ ] every row has physical_task_id
- [ ] scene namespace includes backend
- [ ] objects/receptacles separate
- [ ] skill family assigned
- [ ] primitive set assigned

## Core

- [ ] Experience = 1500
- [ ] ID = 300
- [ ] Transfer = 300
- [ ] Retention = 100
- [ ] Core total = 2200
- [ ] `physical_task_id.nunique() = 2200`

## Pair

- [ ] ID pairs = 300
- [ ] Transfer pairs = 300
- [ ] every ID pair has compatible skill
- [ ] every Transfer pair has explicit transfer axis
- [ ] Spatial transfer is AI2THOR → ProcTHOR
- [ ] TVR transfer comes only from official eval scenes

## Leakage

- [ ] 4 role physical sets pairwise disjoint
- [ ] EB language rewrite groups do not cross core roles
- [ ] TVR eval scene does not leak into TVR experience
- [ ] Asgard sanity set absent
- [ ] Spatial multi-agent tasks absent
- [ ] no model trajectory dataset mixed into task registry

## Release

- [ ] `stats.json` complete
- [ ] `README_DATASET.md` complete
- [ ] `VALIDATION_REPORT.md` ends with `FINAL_STATUS: PASS`

---

# 25. 最终应该看到的最关键数字

如果 Agent 最终报告中的数字不是下面这样，数据准备就没有完成：

```text
RAW TASK ENTRIES
================
EB-ALFRED        300
EB-Habitat       300
SpatialWorld     438
AsgardBench      108
TVRBench        6900
--------------------
TOTAL           8046

CORE V1.0
================
Experience      1500
ID Probe         300
Transfer Probe   300
Retention Probe  100
--------------------
TOTAL           2200

CORE BY SOURCE
================
TVRBench        1408
SpatialWorld     410
AsgardBench      104
EB-ALFRED        137
EB-Habitat       141
--------------------
TOTAL           2200

PAIRS
================
ID pairs         300
Transfer pairs   300
Retention anchors 100

TAXONOMY
================
Reusable skill families  12
Canonical primitives     13
```

最后再强调一次：

> **8,046 是 raw registry 的规模，不是最终 benchmark 要等权使用的规模。Core v1.0 必须是经过物理去重、防语言改写泄漏、scene/backend/object/skill 规范化后得到的 2,200 个互斥 physical task instances。**

这才是后续进行 Experience Learning、ID、Transfer、Retention 以及 RSI 持续迭代实验时应该使用的数据底座。
