# Bash 脚本快速入口

这些脚本是给新手准备的快捷入口，通常只需要修改脚本开头的变量，或者在运行时用环境变量覆盖默认值。

推荐顺序：

1. `bash scripts/01_setup_env.sh`
2. `bash scripts/02_optimize.sh`
3. `bash scripts/03_benchmark.sh`
4. `bash scripts/04_analyze.sh`
5. `bash scripts/05_verify.sh`

一次跑完整流程：

```bash
bash scripts/07_full_pipeline.sh
```

运行更细的 MF-RLCD 参数实验：

```bash
bash scripts/03_benchmark_advanced.sh
```

运行论文用 15 个 MF-RLCD 分级目标批量实验：

```bash
bash scripts/08_mfrlcd_paper_batch.sh
```

运行 MF-RLCD 消融实验：

```bash
bash scripts/09_mfrlcd_ablation.sh
```

示例：覆盖目标参数后运行单目标优化。

```bash
E_TARGET=112000 SIGMA_Y_TARGET=930 KT_TARGET=1.18 bash scripts/02_optimize.sh
```
