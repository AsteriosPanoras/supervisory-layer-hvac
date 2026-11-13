"""Crossing a building with a weather window: the configuration of an existing scenario with its weather lines replaced.

A scenario of the first or second set is a building and a weather window.  Its config.gin holds, besides the building, exactly these weather-dependent lines:
  # Scenario: <building>/<weather>                                         (heading, a comment)
  convection_coefficient = <x>                                              (a constant of the station: one value per station in all 156 configurations)
  weather_csv_path / floor_plan_filepath / zone_map_filepath / metrics_path   (four paths that name the scenario folder <building>/<weather>)
  start_timestamp = '<date> 06:00'                                          (the start of the window)
cross_config() writes the configuration of the same building in another folder with another station's convection coefficient and another start; the weather file itself is
linked by the kernel runner (weather/<station>.csv).  It fails loudly if the text does not have exactly these lines.
Used by runner/kernel_runner_v2.py (in the kernel) and by experiments/*/make_design.py checks (locally).
"""
import re


def cross_config(text: str, building: str, source_weather: str, new_weather: str, convection: str, start: str) -> str:
    old, new = f"evaluation_scenarios/{building}/{source_weather}/", f"evaluation_scenarios/{building}/{new_weather}/"
    if text.count(old) != 4:
        raise ValueError(f"expected 4 paths of {building}/{source_weather}, found {text.count(old)}")
    text = text.replace(old, new)
    heading = f"# Scenario: {building}/{source_weather}\n"
    if text.count(heading) != 1:
        raise ValueError("the heading line of the scenario was not found exactly once")
    text = text.replace(heading, f"# Scenario: {building}/{new_weather} (crossed: the building of {building}/{source_weather} on the weather window {new_weather})\n")
    text, n = re.subn(r"^convection_coefficient = [\d.]+$", f"convection_coefficient = {convection}", text, flags=re.M)
    if n != 1:
        raise ValueError("convection_coefficient was not found exactly once")
    text, n = re.subn(r"^start_timestamp = '[^']+'$", f"start_timestamp = '{start}'", text, flags=re.M)
    if n != 1:
        raise ValueError("start_timestamp was not found exactly once")
    return text
