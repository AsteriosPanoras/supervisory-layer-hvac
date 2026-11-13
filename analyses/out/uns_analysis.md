## Official training scenes (change in loss, %; negative = better)
| scene | U vs G30 | UR vs U | R vs G30 | night mean OAT | max zone T unocc. G30 / U | zone T at 06h G30 / U | learned-start extensions (UR) |
|---|---|---|---|---|---|---|---|
| building_train_1|denver_7 | +0.00 | +0.00 | +0.00 | 11.5 | 22.9 / 22.9 | 22.1 / 22.1 | 0 |
| building_train_1|fairbanks_1 | +0.00 | -0.08 | -0.08 | -21.0 | 22.8 / 22.8 | 21.7 / 21.7 | 12 |
| building_train_1|phoenix_5 | -11.44 | +0.00 | +0.00 | 35.9 | 25.4 / 31.8 | 23.3 / 23.4 | 1 |
| building_train_1|singapore_8 | -1.97 | +0.00 | +0.00 | 26.3 | 25.0 / 27.1 | 23.2 / 23.2 | 0 |
| building_train_2|fairbanks_2 | +0.00 | +0.00 | +0.00 | -9.6 | 22.7 / 22.7 | 21.9 / 21.9 | 0 |
| building_train_4|phoenix_8 | +0.00 | +0.00 | +0.00 | 13.4 | 23.1 / 23.1 | 22.5 / 22.5 | 0 |

Where the setup changes the loss (% of G30 loss):
- building_train_1|phoenix_5: energy+carbon, weekday nights -2.34; energy+carbon, weekends -20.12; energy+carbon, occupied +8.86; comfort, occupied +2.16; comfort, weekday mornings 06-09 +1.66; learned start on weekday-morning comfort (UR vs U) +0.000
- building_train_1|singapore_8: energy+carbon, weekday nights +1.15; energy+carbon, weekends -5.24; energy+carbon, occupied +1.95; comfort, occupied +0.18; comfort, weekday mornings 06-09 +0.18; learned start on weekday-morning comfort (UR vs U) +0.000

## Development (gain, %; positive = better)
- UR vs G30 by architecture: {'52000': '+1.37', '52004': '-0.30', '52005': '+1.30', '52007': '+1.95', '52008': '+3.51', '52010': '-0.29', '52101': '+2.13', '52103': '+7.55', '52105': '-0.49'}
- UR vs G30 by first-day class: {'C': ('+4.76', 4), 'M': ('-0.20', 8), 'H': ('+1.52', 6)}
- U vs G30 (finished cells): - (n=0); UR vs U: - (n=0)
