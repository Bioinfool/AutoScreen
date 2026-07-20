# 使用 AutoDock Vina 运行 AutoScreen

## 依赖

1. 安装 [AutoDock Vina](https://github.com/ccsb-scripps/AutoDock-Vina)，确保 `vina` 在 PATH 中。
2. 安装 OpenBabel（`obabel`）用于配体 PDB → PDBQT 转换。
3. 准备受体 `.pdbqt`，并在 `configs/vina_demo.yaml` 中设置对接盒参数。

## 配置示例

```yaml
executor: vina
vina:
  receptor: /path/to/receptor.pdbqt
  box_center: [x, y, z]
  box_size: [20, 20, 20]
  vina_bin: vina
```

## 运行

```bash
python -m autoscreen.cli run --config configs/vina_demo.yaml
```

若未配置 `receptor` 或找不到 `vina`，`VinaExecutor` 会抛出明确的 `RuntimeError`。  
离线 Replay campaign 不需要安装 Vina。
