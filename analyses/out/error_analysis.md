## T. Time (gain of the layer vs G30, composite)
- by week: {'days 1-7': '+3.28', 'days 8-14': '+3.36', 'days 15-21': '+2.14', 'days 22-30': '+2.70'}
- by day type: {'Monday': '+3.05', 'Tue-Fri': '+2.83', 'weekend': '+1.05'}
- first 5 weekdays vs later: {'first 5 weekdays': '+3.29', 'later weekdays': '+2.80'}
- by hour (change in % of total G30 loss: energy / carbon / comfort; gain within hour):
  - 00h: +0.001 / +0.004 / +0.000; within-hour gain -1.73 %
  - 01h: -0.001 / +0.000 / +0.000; within-hour gain +0.16 %
  - 02h: +0.270 / +0.551 / +0.000; within-hour gain -187.04 %
  - 03h: +0.466 / +1.105 / +0.000; within-hour gain -369.43 %
  - 04h: -0.164 / -0.269 / +0.000; within-hour gain +10.18 %
  - 05h: +0.004 / +0.031 / +0.000; within-hour gain -0.82 %
  - 06h: -0.137 / -0.138 / -0.813; within-hour gain +11.31 %
  - 07h: -0.074 / -0.093 / -0.471; within-hour gain +7.80 %
  - 08h: +0.029 / -0.057 / -0.393; within-hour gain +5.74 %
  - 09h: +0.051 / -0.059 / -0.370; within-hour gain +5.41 %
  - 10h: +0.015 / -0.067 / -0.301; within-hour gain +5.20 %
  - 11h: -0.001 / -0.068 / -0.256; within-hour gain +4.90 %
  - 12h: -0.002 / -0.058 / -0.222; within-hour gain +4.31 %
  - 13h: -0.003 / -0.054 / -0.198; within-hour gain +3.96 %
  - 14h: -0.002 / -0.050 / -0.184; within-hour gain +3.74 %
  - 15h: -0.001 / -0.043 / -0.172; within-hour gain +3.50 %
  - 16h: +0.000 / -0.038 / -0.172; within-hour gain +3.51 %
  - 17h: -0.003 / -0.037 / -0.171; within-hour gain +3.66 %
  - 18h: -0.008 / -0.033 / -0.146; within-hour gain +3.40 %
  - 19h: -0.001 / +0.001 / +0.000; within-hour gain -0.10 %
  - 20h: -0.003 / -0.004 / +0.000; within-hour gain +2.47 %
  - 21h: -0.000 / +0.002 / +0.000; within-hour gain -0.80 %
  - 22h: -0.003 / -0.003 / +0.000; within-hour gain +1.96 %
  - 23h: -0.003 / -0.004 / +0.000; within-hour gain +2.30 %

## X. Weather extremes and variability (calendar days, composite)
- by_daily_mean_oat: < -5 C: +4.46 % (n=44); -5..5 C: +3.30 % (n=134); 5..15 C: +1.46 % (n=163); 15..25 C: +1.32 % (n=307); >= 25 C: +3.53 % (n=252)
- extreme_days: coldest 10% (< -0.2 C): +4.11 % (n=95); middle 80%: +2.10 % (n=716); hottest 10% (> 28.5 C): +4.18 % (n=89)
- by_day_to_day_change: cold snap <= -6 K: +3.39 % (n=33); -6..-2 K: +4.00 % (n=107); stable -2..2 K: +2.79 % (n=566); 2..6 K: +2.59 % (n=151); warm surge >= 6 K: +1.33 % (n=13)
- by_within_day_range: small (< 7.6 K): +2.54 % (n=311); medium: +3.41 % (n=294); large (>= 11.7 K): +2.72 % (n=295)
- daily gain distribution: {'days': 900, 'share_better': 0.5211111111111111, 'share_worse_than_1pct': 0.025555555555555557, 'p5': -0.23355170909704173, 'median': 1.0412430695438957e-05, 'p95': 8.859366783587102}
- worst 10 days: building_52205|kansascity22_w0206 2022-02-21 -16.1 % (Tmean 11.2, dT 3.3, dow 0); building_52205|orlando22_w1009 2022-10-10 -5.9 % (Tmean 25.7, dT -0.5, dow 0); building_52212|tampa22_w0626 2022-07-04 -4.9 % (Tmean 29.4, dT 0.9, dow 0); building_52203|albuquerque22_w1120 2022-12-11 -4.7 % (Tmean 5.4, dT 0.2, dow 6); building_52207|seattle22_w0102 2022-01-17 -4.6 % (Tmean 6.1, dT 1.9, dow 0); building_52205|kansascity22_w0206 2022-02-27 -4.6 % (Tmean 3.7, dT 5.6, dow 6); building_52208|raleigh22_w0612 2022-07-04 -3.5 % (Tmean 28.5, dT -0.8, dow 0); building_52203|albuquerque22_w1120 2022-11-24 -2.8 % (Tmean 3.4, dT -0.4, dow 3); building_52201|spokane22_w1106 2022-11-17 -2.6 % (Tmean -2.4, dT 2.0, dow 3); building_52201|spokane22_w1106 2022-11-27 -2.5 % (Tmean -0.3, dT 1.4, dow 6)

## D. Learned-start decisions under weather unknown at decision time
- mornings 122, capped 17, freezes ['building_52211|billings22_w0206']
- prediction error at opening (K): {'median': 0.2166735602776879, 'p10': -0.006706992592697011, 'p90': 0.5919273703341119, 'share_within_0p5K': 0.8688524590163934, 'share_miss_gt_0p5K': 0.13114754098360656}
- by surprise during recovery: {'colder by > 2 K': {'n': 4, 'median_err_K': 0.3185226189855097, 'share_miss': 0.0, 'mean_day_gain_pct': 14.948565227282675}, 'colder 0.5-2 K': {'n': 29, 'median_err_K': 0.2509880402494911, 'share_miss': 0.10344827586206896, 'mean_day_gain_pct': 9.192282265938868}, 'stable +-0.5 K': {'n': 81, 'median_err_K': 0.1749276141715001, 'share_miss': 0.14814814814814814, 'mean_day_gain_pct': 9.001647432228113}, 'warmer 0.5-2 K': {'n': 7, 'median_err_K': 0.11749756906485792, 'share_miss': 0.14285714285714285, 'mean_day_gain_pct': 7.781402066931759}, 'warmer by > 2 K': {'n': 1, 'median_err_K': 0.20392453780007613, 'share_miss': 0.0, 'mean_day_gain_pct': 20.913504909745317}}
- by surprise of the next occupied day: {'day colder by > 6 K': {'n': 9, 'median_err_K': 0.3030537467734222, 'share_miss': 0.1111111111111111, 'mean_day_gain_pct': 10.438701551148716}, 'colder 2-6 K': {'n': 19, 'median_err_K': 0.2509880402494911, 'share_miss': 0.21052631578947367, 'mean_day_gain_pct': 11.895318140419949}, 'stable +-2 K': {'n': 63, 'median_err_K': 0.21344145844602413, 'share_miss': 0.12698412698412698, 'mean_day_gain_pct': 8.00909484445198}, 'warmer 2-6 K': {'n': 25, 'median_err_K': 0.1749276141715001, 'share_miss': 0.12, 'mean_day_gain_pct': 9.220197820096546}, 'warmer by > 6 K': {'n': 6, 'median_err_K': 0.10179498990132502, 'share_miss': 0.0, 'mean_day_gain_pct': 12.641654779816152}}
- Spearman error vs recovery surprise -0.30; vs next-day surprise -0.17
- cost of misses: {'misses': 16, 'hits': 106, 'mean_day_gain_pct_miss': 12.801564617247525, 'mean_day_gain_pct_hit': 8.736435605039079, 'mean_comfort_06_09_delta_miss': -0.6075657388282707, 'mean_comfort_06_09_delta_hit': -0.2689849917142319}
- looked right but missed: {'n': 0, 'of_mornings_predicted_at_target': 0}

## B. Buildings
- Spearman gain vs: {'learned_tau': 0.81, 'envelope_C': -0.48, 'facade_U': -0.16, 'floors': 0.12, 'zones': -0.45, 'ahus': -0.48}
