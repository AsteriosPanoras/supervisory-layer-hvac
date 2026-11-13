# Parameters (extracted from the frozen research code; see the paper's appendices for their meaning)

## Controller configurations (constants of the public K07 controller that differ from its published values)

- **B** (baseline, tuned for a first-workday objective): RAMP=0.25, COLD_SAT=-0.8, HEAT_U_SCALE=0.875, KP=0.075, NIGHT_HEAT_DROP_K=3, MILD_SHIFT_K=0.75
- **G30** (tuned for the 30-day horizon on 12 development episodes): RAMP=0.3548, COLD_SAT=-1.0, KI=0.0068, KP=0.0699, HEAT_U_SCALE=0.8747, PRESTART_H=2, EVENING_STOP_H=0, NIGHT_HEAT_DROP_K=3.5886, MILD_SHIFT_K=0.7701, WATER_SCALE=0.9907, MIN_DAMPER=0.0

## Weather-conditioned profile library (one profile per initial regime; class from the mean outside air observed over the first 19 h:
extreme cold < -15 C, cold < 11 C, mild < 24.5 C, hot otherwise)

- **extreme cold (E)**: RAMP=0.3096, COLD_SAT=-0.2481, KI=0.0058, KP=0.0714, HEAT_U_SCALE=0.8628, PRESTART_H=2, EVENING_STOP_H=0, NIGHT_HEAT_DROP_K=2.9775, MILD_SHIFT_K=0.8475, WATER_SCALE=1.0, WARM_SAT=-0.7764, MIN_DAMPER=0.013
- **cold (C)**: RAMP=0.3096, COLD_SAT=-0.2481, KI=0.0058, KP=0.0714, HEAT_U_SCALE=0.8628, PRESTART_H=2, EVENING_STOP_H=0, NIGHT_HEAT_DROP_K=2.9775, MILD_SHIFT_K=0.8475, WATER_SCALE=0.8842, WARM_SAT=-0.7764, MIN_DAMPER=0.013
- **mild (M)**: RAMP=0.25, COLD_SAT=-0.8, HEAT_U_SCALE=0.875, KP=0.075, NIGHT_HEAT_DROP_K=3, MILD_SHIFT_K=0.75, PRESTART_H=1
- **hot (H)**: RAMP=0.1711, COLD_SAT=-0.8432, KI=0.0051, KP=0.0671, HEAT_U_SCALE=0.864, PRESTART_H=2, EVENING_STOP_H=0, NIGHT_HEAT_DROP_K=2.9772, MILD_SHIFT_K=0.5459, FORCED_SAT=-0.5923, FORCED_PRESSURE=-0.9813, MIN_DAMPER=0.0728

## Learning source (fixed in advance; never tuned on test data)

Learned optimal start (time constant learned online within bounds):

| parameter | value |
|---|---|
| DSTAR | 0.3 |
| LMIN | 2.0 |
| LMAX | 4.0 |
| LMAX_ID | 3.0 |
| TAU0 | 1.0 |
| ALPHA | 0.5 |
| TAU_LO | 0.3 |
| TAU_HI | 3.0 |
| INF_D0 | 0.3 |
| CHECK_MARGIN | 0.5 |
| GATE_OAT_C | 99.0 |
| DEEP_OAT_C | -15.0 |
| LMAX_DEEP | 0.0 |
| BEYOND_NOCHECK | 0.0 |

Pressure reset (trim & respond of the duct static-pressure command, Pa; a floor under the controller's own command):

| parameter | value |
|---|---|
| SPMIN | 200.0 |
| SPMAX | 400.0 |
| I | 0.0 |
| U_REQ | 0.95 |
| U_REL | 0.85 |
| HI_EXTRA_K | 1.0 |
| TRIM | 12.0 |
| RES | 15.0 |
| RESMAX | 32.0 |
| UNRESP_STEPS | 24.0 |
| UNRESP_DAYS | 2.0 |
| GUARD_FRAC | 0.9 |
| GUARD_DAYS | 3.0 |

Learned night precooling (for an energy-dominated controller):

| parameter | value |
|---|---|
| PC_GATE_C | 11.0 |
| PC_ECON_K | 2.0 |
| PC_BELOW_HI_K | 0.5 |
| PC_STEP_K | 0.25 |
| PC_ALPHA | 0.5 |
| PC_NEED_K | 0.0 |
| PC_WINDOW_H | 5.0 |

Source files (sha256): run_episode_k07_v9supe.py e1b7921f058fe200d1234f4b6480be88f6ceb9f7a16365583004658ff262df17; run_episode_penguin_v3pc.py 0a4fd26dcc9b9b44fe405472b0e162d257c95d2fa1922194e0b228f8785c866c.
