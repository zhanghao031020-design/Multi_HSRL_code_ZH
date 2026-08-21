# 三波长 HSRL 初步仿真

本项目实现 355、532、1064 nm HSRL 的第一阶段光学量级仿真：在固定
`3beta+2alpha` 基线的前提下，量化加入 `alpha1064` 对低维微物理产品的条件边际价值。

当前范围包括 Arm A（无噪声局部可识别性）、Arm B（相关协方差下的光学量 Monte Carlo）和 Arm C（单层/单距离门六路原始信号 Monte Carlo）。Arm C 使用参数化理想鉴频器，并将泊松计数、背景、增益、激光能量、透过率和串扰 nuisance 写入原始观测记录；它仍不等价于实测硬件或垂直廓线结论。

运行测试：

```powershell
& .\.venv\Scripts\python.exe -m pytest
```

运行小规模试验：

```powershell
& .\.venv\Scripts\python.exe scripts\run_optical_mc.py --config configs\smoke.yaml
```

运行 Arm C 阶段试验：

```powershell
& .\.venv\Scripts\python.exe scripts\run_signal_mc.py --config configs\signal_mc_stage1.yaml --output-dir outputs\signal_stage1
```

Arm C 输出同时保存 `raw_signal[6]`、`raw_counts[6]`、计数协方差、五/六光学量、逐臂 retrieval、paired retrieval 和报告。原始通道顺序为 `mixed_355、molecular_355、mixed_532、molecular_532、mixed_1064、molecular_1064`；其中 molecular 通道是鉴频器输出标签，不是“纯分子信号”。
