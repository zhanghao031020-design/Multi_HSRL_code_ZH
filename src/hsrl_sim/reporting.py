from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PRODUCT_LABELS = {
    "Vf": "Fine-mode volume",
    "Vc": "Coarse-mode volume",
    "Vf_over_Vc": "Fine/coarse volume ratio",
    "reff_total": "Total effective radius",
    "reff_coarse": "Coarse-mode effective radius",
}


def write_optical_report(
    output_dir: Path,
    summary_records: Sequence[Mapping[str, object]],
    arm_a_records: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
) -> None:
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    aggregate = [row for row in summary_records if str(row.get("scenario_id")) == "all"]
    figure, axes = plt.subplots(len(PRODUCT_LABELS), 1, figsize=(8, 12), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, product_id in zip(axes, PRODUCT_LABELS):
        rows = sorted(
            (row for row in aggregate if row["product_id"] == product_id),
            key=lambda row: float(row["alpha1064_relative_error"]),
        )
        x = [float(row["alpha1064_relative_error"]) for row in rows]
        y = [float(row["gain"]) for row in rows]
        low = [float(row["bootstrap_ci_low"]) for row in rows]
        high = [float(row["bootstrap_ci_high"]) for row in rows]
        axis.plot(x, y, marker="o")
        axis.fill_between(x, low, high, alpha=0.2)
        axis.axhline(0.10, color="tab:green", linestyle="--", linewidth=0.8)
        axis.axhline(0.0, color="black", linewidth=0.6)
        axis.set_ylabel("RMSE gain")
        axis.set_title(PRODUCT_LABELS[product_id])
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("alpha1064 relative error")
    figure.tight_layout()
    figure.savefig(figures_dir / "gain_response.png", dpi=160)
    plt.close(figure)

    scenario_rows = [
        row for row in summary_records if str(row.get("scenario_id")) != "all"
    ]
    robust_records = []
    for product_id in PRODUCT_LABELS:
        errors = sorted({
            float(row["alpha1064_relative_error"])
            for row in scenario_rows
            if row["product_id"] == product_id
        })
        for error in errors:
            rows = [
                row for row in scenario_rows
                if row["product_id"] == product_id
                and float(row["alpha1064_relative_error"]) == error
            ]
            robust_records.append({
                "product_id": product_id,
                "alpha1064_relative_error": error,
                "worst_gain": min(float(row["gain"]) for row in rows),
                "worst_ci_low": min(float(row["bootstrap_ci_low"]) for row in rows),
                "worst_coverage": min(float(row["coverage_plus"]) for row in rows),
                "worst_informative_rate": min(float(row["informative_interval_rate_plus"]) for row in rows),
                "max_multiple_solution_rate": max(float(row["multiple_solution_rate_plus"]) for row in rows),
                "max_failure_rate": max(float(row["failure_rate_plus"]) for row in rows),
            })
    certified = [
        row for row in robust_records
        if bool(config.get("model_mismatch", False))
        and float(row["worst_gain"]) >= 0.10
        and float(row["worst_ci_low"]) > 0.0
        and float(row["max_failure_rate"]) == 0.0
        and float(row["max_multiple_solution_rate"]) == 0.0
        and float(row["worst_coverage"]) >= float(config["coverage_target"])
        and float(row["worst_informative_rate"]) >= float(config["coverage_target"])
    ]
    errors = sorted({float(row["alpha1064_relative_error"]) for row in aggregate})
    lines = [
        "# 三波长 HSRL 光学量级 Arm A/B 阶段报告",
        "",
        "## 结论边界",
        "",
        "本报告只评价在已获得 alpha1064 光学量、双峰对数正态真值、球形 Mie 正演和四参数反演假设下，加入 alpha1064 的条件边际价值。它不等价于六路原始信号硬件性能结论；后者需要 Arm C。",
        "",
        f"配置：seed={config['seed']}，真值数={config['n_truth']}，重复数={config['replicates']}，alpha1064 相对误差={errors}。",
        "",
        "## 判据",
        "",
        "有效提升同时要求所有场景的最差 RMSE gain >= 0.10、paired bootstrap 置信区间下界 > 0、失败率为 0、多解率为 0、覆盖率和信息性区间率达到目标，并且模型失配实验已开启。后验变窄不单独作为准确率提升证据。",
        "",
        "## 汇总结果",
        "",
        "以下表格按所有场景取最差值（scenario-robust）；不使用跨场景总平均进行认证。",
        "",
        "| 产品 | alpha1064 相对误差 | 最差 RMSE gain | 最差 CI 下界 | 最差覆盖率 | 最差信息区间率 | 最大多解率 | 结论 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(
        robust_records,
        key=lambda item: (str(item["product_id"]), float(item["alpha1064_relative_error"])),
    ):
        accepted = row in certified
        lines.append(
            f"| {row['product_id']} | {float(row['alpha1064_relative_error']):.2f} | "
            f"{float(row['worst_gain']):.3f} | {float(row['worst_ci_low']):.3f} | "
            f"{float(row['worst_coverage']):.3f} | "
            f"{float(row['worst_informative_rate']):.3f} | "
            f"{float(row['max_multiple_solution_rate']):.3f} | "
            f"{'满足初步判据' if accepted else '未满足初步判据'} |"
        )
    lines.extend([
        "",
        "## Arm A 局部可识别性",
        "",
        "Arm A 比较 base/plus 的 Fisher 最小特征值、奇异值和条件数；它只能说明局部敏感度，不替代含噪声的误差统计。详细数据见 arm_a.csv。",
        "",
        f"满足全部初步判据的产品-误差组合数：{len(certified)}。当前结果不认证 alpha1064 的硬件价值；下一阶段需完成六路信号级 Arm C，并用 VSD 交叉检查低维先验依赖。",
        "",
        "输出文件：truth.csv、observations.csv、retrievals.csv、summary.csv、arm_a.csv、figures/gain_response.png。",
    ])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_signal_report(
    output_dir: Path,
    summary_records: Sequence[Mapping[str, object]],
    observation_rows: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
) -> None:
    """Write the Arm C signal-level report and gain/count diagnostics."""

    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    aggregate = [row for row in summary_records if str(row.get("scenario_id")) == "all"]
    levels = sorted({float(row["molecular_1064_transmission_scale"]) for row in aggregate})
    figure, axes = plt.subplots(len(PRODUCT_LABELS), 1, figsize=(8, 12), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, product_id in zip(axes, PRODUCT_LABELS):
        rows = sorted(
            (row for row in aggregate if row["product_id"] == product_id),
            key=lambda row: float(row["molecular_1064_transmission_scale"]),
        )
        x = [float(row["molecular_1064_transmission_scale"]) for row in rows]
        y = [float(row["gain"]) for row in rows]
        low = [float(row["bootstrap_ci_low"]) for row in rows]
        high = [float(row["bootstrap_ci_high"]) for row in rows]
        axis.plot(x, y, marker="o")
        axis.fill_between(x, low, high, alpha=0.2)
        axis.axhline(0.10, color="tab:green", linestyle="--", linewidth=0.8)
        axis.axhline(0.0, color="black", linewidth=0.6)
        axis.set_ylabel("RMSE gain")
        axis.set_title(PRODUCT_LABELS[product_id])
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("1064 molecular transmission scale")
    figure.tight_layout()
    figure.savefig(figures_dir / "signal_gain_response.png", dpi=160)
    plt.close(figure)

    molecular_counts = [
        float(row["alpha1064_molecular_expected_counts"])
        for row in observation_rows
        if np.isfinite(float(row["alpha1064_molecular_expected_counts"]))
    ]
    robust_records = []
    scenario_rows = [row for row in summary_records if str(row.get("scenario_id")) != "all"]
    for product_id in PRODUCT_LABELS:
        for level in levels:
            rows = [
                row for row in scenario_rows
                if row["product_id"] == product_id
                and float(row["molecular_1064_transmission_scale"]) == level
            ]
            if not rows:
                continue
            robust_records.append({
                "product_id": product_id,
                "level": level,
                "worst_gain": min(float(row["gain"]) for row in rows),
                "worst_ci_low": min(float(row["bootstrap_ci_low"]) for row in rows),
                "worst_coverage": min(float(row["coverage_plus"]) for row in rows),
                "max_failure_rate": max(float(row["failure_rate_plus"]) for row in rows),
                "max_multiple_solution_rate": max(float(row["multiple_solution_rate_plus"]) for row in rows),
            })
    certified = [
        row for row in robust_records
        if row["worst_gain"] >= 0.10
        and row["worst_ci_low"] > 0.0
        and row["worst_coverage"] >= float(config.get("coverage_target", 0.90))
        and row["max_failure_rate"] == 0.0
        and row["max_multiple_solution_rate"] == 0.0
    ]
    lines = [
        "# 三波长 HSRL 六通道原始信号级 Arm C 阶段报告",
        "",
        "## 结论边界",
        "",
        "本报告评价单层/单距离门、理想参数化鉴频器下的六路原始计数。仪器直接输出的是 mixed/molecular 信号，不是 β/α；β/α 是对原始计数进行标定和单距离门衰减提取后的派生观测。结果不等价于实测硬件或垂直廓线性能。",
        "",
        f"配置：seed={config['seed']}，真值数={config['n_truth']}，重复数={config['replicates']}，1064 分子通道透过率等级={levels}。",
        f"1064 分子通道期望计数范围：{min(molecular_counts):.2f}–{max(molecular_counts):.2f}。",
        "",
        "## 正演与噪声",
        "",
        "每个波长输出一个混合通道和一个分子通道；消光通过两程指数衰减进入期望计数。噪声在原始计数层加入泊松抽样，并单独抽样增益、激光能量、鉴频器透过率和串扰比例扰动；系统误差没有被伪装成独立白噪声。",
        "",
        "## 配对原则",
        "",
        "同一 truth_id/replicate_id/signal level 只生成一份六路原始观测。base 使用提取后的前五个光学量，plus 使用同一观测的六个光学量；详细原始数据见 observations.csv，逐臂记录见 retrievals.csv，成对记录见 paired_retrievals.csv。",
        "",
        "## 场景稳健汇总",
        "",
        "有效提升同时要求最差场景 RMSE gain >= 0.10、paired bootstrap 下界 > 0、plus 覆盖率达到目标、失败率为 0 且多解率为 0。后验区间变窄不单独作为准确率提升证据。",
        "",
        "| 产品 | 1064 分子透过率等级 | 最差 gain | 最差 CI 下界 | 最差覆盖率 | 最大失败率 | 最大多解率 | 结论 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(robust_records, key=lambda item: (item["product_id"], item["level"])):
        accepted = row in certified
        lines.append(
            f"| {row['product_id']} | {row['level']:.2f} | {row['worst_gain']:.3f} | "
            f"{row['worst_ci_low']:.3f} | {row['worst_coverage']:.3f} | "
            f"{row['max_failure_rate']:.3f} | {row['max_multiple_solution_rate']:.3f} | "
            f"{'满足初步判据' if accepted else '未满足初步判据'} |"
        )
    lines.extend([
        "",
        f"满足全部初步判据的产品-信号等级组合数：{len(certified)}。",
        "",
        "## 阶段结论",
        "",
        "Arm C 只回答原始信号计数层是否会削弱或保留光学量级的边际价值；当前版本不认证 1064 nm 分子通道的硬件价值，也不宣称可以唯一恢复完整粒径谱。下一步应使用更接近实测的鉴频器透过率、死时间、背景和垂直廓线数据进行压力测试。",
        "",
        "输出文件：truth.csv、observations.csv、retrievals.csv、paired_retrievals.csv、summary.csv、figures/signal_gain_response.png。",
    ])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
