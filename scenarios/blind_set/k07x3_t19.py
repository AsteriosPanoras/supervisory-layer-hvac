"""K07-X3: day-1 class of a weather strip WITHOUT simulation: T19 = mean of the outside-air temperature the K07 policy observes at steps 0..227 (the first 19 h),
read from the kit's own ReplayWeatherController (linear interpolation of the hourly csv) at start_timestamp + i x 5 min, the start read as UTC (as the simulator does:
the naive start is localised to UTC by the weather controller; reading it in the config time_zone US/Pacific is off by up to 3 K).
Classes as in the K07 runner v9x: E < -15 C, C < 11 C, M < 24.5 C, H >= 24.5 C.  Validated against the 18 strips whose T19 the K07-X 229-step probes observed (K07X_STRIP_DAY1_CLASS.json).
usage as a module: t19(kit, csv_path, start, tz) ; as a script: <kit venv python> k07x3_t19.py <kit_root> <k07x_dir>  (prints the validation)"""
import sys, os, json
import numpy as np, pandas as pd
def classify(T): return "E" if T < -15 else "C" if T < 11 else "M" if T < 24.5 else "H"
_CTL = {}
def t19(kit, csv_path, start, tz="UTC", offset=0, n=228):
    sys.path.insert(0, str(kit))
    from smart_buildings.smart_control.simulator.weather_controller import ReplayWeatherController
    if csv_path not in _CTL: _CTL[csv_path] = ReplayWeatherController(local_weather_path=str(csv_path), convection_coefficient=1.0)
    c = _CTL[csv_path]; t0 = pd.Timestamp(start, tz=tz)
    return float(np.mean([c.get_current_temp(t0 + pd.Timedelta(minutes=5 * (i + offset))) - 273.15 for i in range(n)]))
if __name__ == "__main__":
    import yaml
    kit, k07x = sys.argv[1], sys.argv[2]; ref = json.load(open(os.path.join(k07x, "K07X_STRIP_DAY1_CLASS.json"))); worst = {}
    for off in (0, 1):
        errs = []
        for w, r in sorted(ref.items()):
            y = yaml.safe_load(open(os.path.join(kit, "smart_buildings/weather", f"{w}.yaml")))
            T = t19(kit, os.path.join(kit, "smart_buildings/weather", y["weather_csv"]), str(y["start_timestamp"]), offset=off)
            errs.append(abs(round(T, 2) - r["day1_T19_C"])); assert off != 0 or classify(T) == r["day1_class"], (w, T, r)
            if off == 0: print(f"{w:14s} probe {r['day1_T19_C']:7.2f} {r['day1_class']}  offline {T:7.3f} {classify(T)}")
        worst[off] = max(errs)
    assert worst[0] <= 0.006, worst
    print("max |offline - probe| (K):", {f"offset {k}": round(v, 3) for k, v in worst.items()})
