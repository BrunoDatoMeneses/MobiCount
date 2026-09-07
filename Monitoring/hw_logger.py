#!/usr/bin/env python3
"""
HW Logger — Enregistre l'état GPU + CPU dans un fichier log
Usage : python hw_logger.py [intervalle_secondes] [fichier_log]
        python hw_logger.py 5 system.log
"""

import time
import sys
import os
import datetime

# ── Imports ────────────────────────────────────────────────────────────────
try:
    import psutil
except ImportError:
    print("❌ psutil manquant → pip install psutil")
    sys.exit(1)

try:
    import pynvml
    pynvml.nvmlInit()
    NVIDIA = True
except Exception:
    NVIDIA = False

try:
    import GPUtil
    GPUTIL = True
except ImportError:
    GPUTIL = False

# ── Paramètres ─────────────────────────────────────────────────────────────
INTERVAL  = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
LOG_FILE  = sys.argv[2]        if len(sys.argv) > 2 else "hw_stats.log"
SEP_BLOCK = "=" * 72
SEP_LINE  = "-" * 72


# ── Collecte CPU ───────────────────────────────────────────────────────────
def get_cpu_info():
    info = {}

    # Utilisation globale et par cœur
    info["usage_global"]  = psutil.cpu_percent(interval=None)
    info["usage_per_core"] = psutil.cpu_percent(interval=None, percpu=True)

    # Fréquences
    freq = psutil.cpu_freq(percpu=False)
    if freq:
        info["freq_current_mhz"] = freq.current
        info["freq_min_mhz"]     = freq.min
        info["freq_max_mhz"]     = freq.max

    # Températures (Linux / macOS)
    temps = {}
    try:
        sensors = psutil.sensors_temperatures()
        for name, entries in sensors.items():
            for e in entries:
                label = e.label or name
                temps[label] = e.current
    except AttributeError:
        pass
    info["temperatures"] = temps

    # RAM
    vm = psutil.virtual_memory()
    info["ram_total_gb"]   = vm.total  / 1024**3
    info["ram_used_gb"]    = vm.used   / 1024**3
    info["ram_free_gb"]    = vm.available / 1024**3
    info["ram_percent"]    = vm.percent

    # Swap
    sw = psutil.swap_memory()
    info["swap_total_gb"]  = sw.total / 1024**3
    info["swap_used_gb"]   = sw.used  / 1024**3
    info["swap_percent"]   = sw.percent

    # Charge système (1 / 5 / 15 min)
    try:
        la = os.getloadavg()
        info["load_1m"], info["load_5m"], info["load_15m"] = la
    except AttributeError:
        pass

    # Nombre de cœurs
    info["cores_physical"] = psutil.cpu_count(logical=False)
    info["cores_logical"]  = psutil.cpu_count(logical=True)

    return info


# ── Collecte GPU (pynvml) ──────────────────────────────────────────────────
def get_gpu_info_nvml():
    gpus = []
    count = pynvml.nvmlDeviceGetCount()
    for i in range(count):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        g = {"index": i}

        name = pynvml.nvmlDeviceGetName(h)
        g["name"] = name.decode() if isinstance(name, bytes) else name

        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        g["mem_total_mb"] = mem.total / 1024**2
        g["mem_used_mb"]  = mem.used  / 1024**2
        g["mem_free_mb"]  = mem.free  / 1024**2

        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        g["gpu_util_pct"] = util.gpu
        g["mem_util_pct"] = util.memory

        try:
            g["temp_c"] = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            g["temp_c"] = None

        try:
            g["fan_pct"] = pynvml.nvmlDeviceGetFanSpeed(h)
        except Exception:
            g["fan_pct"] = None

        try:
            g["power_w"]       = pynvml.nvmlDeviceGetPowerUsage(h) / 1000
            g["power_limit_w"] = pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000
        except Exception:
            g["power_w"] = g["power_limit_w"] = None

        try:
            g["clk_gpu_mhz"] = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_GRAPHICS)
            g["clk_mem_mhz"] = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)
            g["clk_sm_mhz"]  = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
        except Exception:
            g["clk_gpu_mhz"] = g["clk_mem_mhz"] = g["clk_sm_mhz"] = None

        try:
            ps = pynvml.nvmlDeviceGetPerformanceState(h)
            g["perf_state"] = f"P{ps}"
        except Exception:
            g["perf_state"] = "N/A"

        try:
            driver = pynvml.nvmlSystemGetDriverVersion()
            g["driver"] = driver.decode() if isinstance(driver, bytes) else driver
        except Exception:
            g["driver"] = "N/A"

        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
            g["processes"] = [{"pid": p.pid, "mem_mb": (p.usedGpuMemory or 0) / 1024**2}
                               for p in procs]
        except Exception:
            g["processes"] = []

        gpus.append(g)
    return gpus


def get_gpu_info_gputil():
    gpus = []
    for g in GPUtil.getGPUs():
        gpus.append({
            "index":        g.id,
            "name":         g.name,
            "gpu_util_pct": g.load * 100,
            "mem_util_pct": None,
            "mem_used_mb":  g.memoryUsed,
            "mem_free_mb":  g.memoryFree,
            "mem_total_mb": g.memoryTotal,
            "temp_c":       g.temperature,
            "fan_pct":      None,
            "power_w":      None,
            "power_limit_w":None,
            "clk_gpu_mhz":  None,
            "clk_mem_mhz":  None,
            "clk_sm_mhz":   None,
            "perf_state":   "N/A",
            "driver":       "N/A",
            "processes":    [],
        })
    return gpus


def get_gpu_info():
    if NVIDIA:
        try:
            return get_gpu_info_nvml()
        except Exception:
            pass
    if GPUTIL:
        return get_gpu_info_gputil()
    return []


# ── Formatage d'une entrée de log ──────────────────────────────────────────
def format_entry(ts, cpu, gpus):
    lines = []
    lines.append(SEP_BLOCK)
    lines.append(f"TIMESTAMP : {ts}")
    lines.append(SEP_BLOCK)

    # ── CPU ──
    lines.append("[CPU]")
    lines.append(f"  Cœurs physiques / logiques : {cpu.get('cores_physical')} / {cpu.get('cores_logical')}")
    lines.append(f"  Utilisation globale        : {cpu['usage_global']:.1f} %")

    per_core = cpu.get("usage_per_core", [])
    if per_core:
        core_str = "  ".join(f"C{i}:{v:.0f}%" for i, v in enumerate(per_core))
        lines.append(f"  Par cœur                   : {core_str}")

    if "freq_current_mhz" in cpu:
        lines.append(f"  Fréquence actuelle         : {cpu['freq_current_mhz']:.0f} MHz"
                     f"  (min {cpu['freq_min_mhz']:.0f} / max {cpu['freq_max_mhz']:.0f})")

    if "load_1m" in cpu:
        lines.append(f"  Charge système (1/5/15m)   : {cpu['load_1m']:.2f} / {cpu['load_5m']:.2f} / {cpu['load_15m']:.2f}")

    temps = cpu.get("temperatures", {})
    if temps:
        t_str = "  ".join(f"{k}: {v:.1f}°C" for k, v in list(temps.items())[:8])
        lines.append(f"  Températures               : {t_str}")

    lines.append(f"  RAM  utilisée / totale     : {cpu['ram_used_gb']:.2f} GB / {cpu['ram_total_gb']:.2f} GB  ({cpu['ram_percent']:.1f} %)")
    if cpu["swap_total_gb"] > 0:
        lines.append(f"  Swap utilisée / totale     : {cpu['swap_used_gb']:.2f} GB / {cpu['swap_total_gb']:.2f} GB  ({cpu['swap_percent']:.1f} %)")

    # ── GPU ──
    lines.append(SEP_LINE)
    if not gpus:
        lines.append("[GPU] Aucune carte graphique détectée ou librairie absente.")
    else:
        for g in gpus:
            lines.append(f"[GPU #{g['index']}] {g['name']}  — Driver : {g.get('driver','N/A')}  — Perf : {g.get('perf_state','N/A')}")
            lines.append(f"  Utilisation GPU            : {g['gpu_util_pct']:.1f} %")
            if g["mem_util_pct"] is not None:
                lines.append(f"  Utilisation ctrl mémoire   : {g['mem_util_pct']:.1f} %")
            lines.append(f"  VRAM utilisée / totale     : {g['mem_used_mb']:.0f} MB / {g['mem_total_mb']:.0f} MB  "
                         f"({g['mem_used_mb']/g['mem_total_mb']*100:.1f} %)")
            if g["temp_c"] is not None:
                lines.append(f"  Température                : {g['temp_c']} °C")
            if g["fan_pct"] is not None:
                lines.append(f"  Ventilateur                : {g['fan_pct']} %")
            if g["power_w"] is not None:
                limit_str = f" / {g['power_limit_w']:.0f} W" if g["power_limit_w"] else ""
                lines.append(f"  Puissance                  : {g['power_w']:.1f} W{limit_str}")
            if g["clk_gpu_mhz"] is not None:
                lines.append(f"  Fréq. GPU / SM / Mém       : {g['clk_gpu_mhz']} / {g['clk_sm_mhz']} / {g['clk_mem_mhz']} MHz")
            if g["processes"]:
                for p in g["processes"]:
                    lines.append(f"  Process GPU                : PID {p['pid']}  VRAM {p['mem_mb']:.0f} MB")

    lines.append("")  # ligne vide entre entrées
    return "\n".join(lines)


# ── Boucle principale ──────────────────────────────────────────────────────
def main():
    # Initialisation du fichier
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{'#'*72}\n")
        f.write(f"# Session démarrée le {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Intervalle : {INTERVAL}s\n")
        f.write(f"{'#'*72}\n\n")

    print(f"📄 Logging → {os.path.abspath(LOG_FILE)}")
    print(f"⏱  Intervalle : {INTERVAL}s")
    print(f"🖥  GPU backend : {'pynvml' if NVIDIA else 'gputil' if GPUTIL else 'aucun'}")
    print(f"Ctrl+C pour arrêter\n")

    # Première mesure CPU à vide (psutil a besoin d'un intervalle)
    psutil.cpu_percent(interval=0.2)

    try:
        while True:
            ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cpu  = get_cpu_info()
            gpus = get_gpu_info()
            entry = format_entry(ts, cpu, gpus)

            # Écriture dans le fichier
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(entry + "\n")

            # Affichage console (résumé)
            gpu_summary = ""
            if gpus:
                g = gpus[0]
                t = f"  {g['temp_c']}°C" if g["temp_c"] is not None else ""
                gpu_summary = f"  │  GPU {g['gpu_util_pct']:.0f}%  VRAM {g['mem_used_mb']:.0f}/{g['mem_total_mb']:.0f}MB{t}"
            print(f"[{ts}]  CPU {cpu['usage_global']:.1f}%  RAM {cpu['ram_used_gb']:.1f}/{cpu['ram_total_gb']:.1f}GB{gpu_summary}")

            time.sleep(INTERVAL)

    except KeyboardInterrupt:
        print(f"\n✅ Logging arrêté. Fichier : {os.path.abspath(LOG_FILE)}")
        if NVIDIA:
            pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
