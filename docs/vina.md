# 使用 AutoDock Vina 运行 AutoScreen

统一文档：[vina_setup.md](vina_setup.md)。

**最小闭环（推荐）：**

```bash
python scripts/run_vina_closed_loop.py
# 或
python -m autoscreen.cli run --config configs/vina_mini.yaml
```

分数约定：`activity = -vina_affinity`（maximize）。换受体时改 YAML 中的 `vina.receptor` / `box_*`。
