# PREREG_GNNSUP_v1: the unchanged layer on a graph-network controller (12 November 2026)

**Status:** written and hashed BEFORE the imitation data are recorded, before training, before the gate and before any blind run.

- Choices: imitation of the neural controller; blind test on the laptop and host-B.

## Question
Does the unchanged supervisory layer (learned optimal start + trim-and-respond pressure reset; code byte-identical to the K07, ZT Best and Residual PPO tests) improve a **graph-network controller**, on building architectures that the network never saw?

## Controller (GN)
- **Base:** the public Residual PPO controller (alexchilton, Apache-2.0), which is a tuned rule with an off-hours rule plus a residual network.
- **Network:** the residual network is replaced by `gnn_net.GraphNets` (d = 64, 2 rounds), a graph network on the building's own structure:
  - **nodes:** zones (VAVs), air handlers and the plant;
  - **edges:** zone-AHU from the metadata, and AHU-plant;
  - **messages, each round:** a zone reads its AHU; an AHU reads the mean of its zones and the plant; the plant reads the mean of the AHUs;
  - **inputs:** each device's raw fields, the base rule's command and the observation's clock and occupancy fields; no derived features;
  - **outputs:** the teacher's residual heads.
- **Unchanged:** the base rule, the residual scale, the hold and the latch, all from the teacher's checkpoint.
- **Runner:** `run_episode_k07_v9supe_gnn.py`, built by `make_runner_gnn.py` from the Residual PPO runner (557d4aab). The supervisor lines are unchanged, as the generator checks.

## Training, fixed now
- **Data:** recordings of the teacher, the Residual PPO network alone (key k07@), 2016 steps from each scenario's start. The split is in `SPLIT_GNN_v1.md`:
  - training: 24 held-out-set scenes (k07x) and 10 official scenes;
  - validation: 3 k07x scenes and 1 official scene, of buildings not used in training;
  - the blind architectures (k07x3) are never read.
- **Command:** `train_gnn.py --iters 6000 --batch 64 --lr 1e-3 --d 64 --seed 0 --threads 4`. The loss is masked MSE to the teacher's residual means, summed over zones, AHUs and the plant.

## Launch gate (fail-fast, on validation scenes only, before any blind run)
- **G1.** GN alone (k07@) runs on the 4 validation scenes, 2016 steps each. Every run must end with status OK and 0 illegal actions.
- **G2.** GN's pooled loss on those 4 scenes is at most **1.25 times** the teacher's pooled loss on the same scenes and steps. The teacher's loss is read from the recording runs, which are the teacher's own episodes.
- **One retry:** if G1 or G2 fails, the network is retrained once with `--iters 20000 --d 128` (nothing else changes) and the gate is applied again.

## Fidelity (before the blind)
- **F1.** Recording mode equals the plain Residual PPO runner bit for bit: continuous and discrete actions and rewards (smoke scene 52000 minneapolis_2, 600 steps).
- **F2 and F3.** `axsup/p275_ax_fidelity.py`, unchanged, on GN with the trained weights (the scenes and steps of the Residual PPO fidelity test):
  - F2: the layer switched off equals GN bit for bit;
  - F3: the layer switched on is identical to GN up to the layer's first action.
- **Gate for the blind:** `RESULT_GNNSUP_FIDELITY.json` with ALL_PASS.

## Blind test
- **Scenes:** the 36 scenes of the blind set, the same as for K07, ZT Best and Residual PPO: 12 architectures × 3 weather windows, 8640 steps each.
- **Arms:** GN (k07@) and GNS (k07sup@;ROS=1,TR=1), 72 runs.
- **Setup:** the same kit, tracker overlay and runner on the laptop and on host-B.
- **Weights:** the file passed by the gate. Its sha256 goes into an addendum, hashed and timestamped before the first blind run.

## Analyses (scripts hashed now)
- **p285_gnnsup_blind.py** (the p277b logic):
  - gain of GNS over GN pooled over the 10 composite architectures, with a 95 % architecture-cluster bootstrap (3000 replicates, seed 1);
  - **Primary rule:** pooled gain > 0 AND a majority of the changed composite architectures better AND no architecture worse than 1 %. This is the rule of the Residual PPO test.
  - Descriptive: GN vs the teacher, GN vs K07-G30, and the components.
- **p286_headroom_gnn.py** (the p278 logic): registered predictions P1-P4 of the headroom account, with GN as the base:
  - **P1:** the per-episode gain rises with GN's comfort share (Spearman ρ > 0, one-sided permutation p < 0.05).
  - **P2:** the lowest-share third gains within ±0.5 %.
  - **P3:** the gain over all 36 episodes is higher or lower than K07's 2.69 % as GN's mean comfort share is higher or lower than K07's.
  - **P4:** the comfort share of GN's first week ranks the gain over days 8-30 (ρ > 0).
- **Reporting:** every result is reported whatever it is, with the failed parts named as failed.
