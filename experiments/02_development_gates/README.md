# 02 development_gates

**Question.** Development gates of the learning source: each mechanism had to gain on the composite development architectures with no architecture more than 1 % worse.

**Where in the paper.** Appendix H.

**Outcome.** Learned start and pressure reset passed on K07; the start failed on Penguin, learned night precooling passed.

**Files here.**
One folder per gate, each with its protocol, analyses (fidelity, gate, held-out check), results and manifests:
- `k07_learned_start/`;
- `k07_pressure_reset/` (with the combined protocol);
- `penguin_learned_start/`.

The Penguin precooling gate is in `06_penguin_precooling`.

**Layout.**
- `PROTOCOL.md`: what was fixed before the runs. Later amendments are separate files, each registered before any outcome.
- `analysis/`: the analysis scripts, hashed before the runs they read.
- `results/`: their outputs.
- `execution/`: job lists and launch scripts.
- `EXECUTION_NOTES.md`: events during the runs, recorded as they happened.
