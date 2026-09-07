## Installation & usage

```bash
pip install psutil pynvml   # pynvml = NVIDIA, ou pip install gputil en alternative
```

```bash
python hw_logger.py                    # intervalle 5s → hw_stats.log
python hw_logger.py 2                  # toutes les 2s
python hw_logger.py 10 monitor.log    # 10s dans un fichier nommé
```

---

### Ce qui est loggé

**CPU**
- Utilisation globale + par cœur
- Fréquence actuelle / min / max
- Charge système (1 / 5 / 15 min)
- Températures capteurs (Linux/macOS)
- RAM utilisée / totale + Swap

**GPU (NVIDIA via pynvml)**
- Utilisation core & contrôleur mémoire
- VRAM utilisée / libre / totale
- Température + ventilateur
- Puissance (W) vs limite TDP
- Fréquences GPU / SM / mémoire
- Performance state (P0–P8)
- PID des processus utilisant le GPU

### Format du fichier log

```
========================================================================
TIMESTAMP : 2026-06-05 14:32:10
========================================================================
[CPU]
  Cœurs physiques / logiques : 8 / 16
  Utilisation globale        : 23.4 %
  Par cœur                   : C0:18%  C1:45%  C2:12% ...
  Fréquence actuelle         : 3600 MHz  (min 400 / max 5000)
  RAM  utilisée / totale     : 11.20 GB / 32.00 GB  (35.0 %)
------------------------------------------------------------------------
[GPU #0] NVIDIA GeForce RTX 4080 — Driver : 545.29  — Perf : P2
  Utilisation GPU            : 67.0 %
  VRAM utilisée / totale     : 8192 MB / 16384 MB  (50.0 %)
  Température                : 74 °C
  Puissance                  : 210.5 W / 320 W
```

