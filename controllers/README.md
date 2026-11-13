# Third-party controllers

- **`k07/`:** the published K07 classes and constants (Danil Malyshev, Apache 2.0).
- **`residual_ppo/`:** the public Residual PPO controller (Alex Chilton, Apache 2.0).
  - Contents: code, weights `skynet.pt`, the author's README, licence notice.
  - The weights are loaded only with `torch.load(..., weights_only=True)`.
- **ZT Best and Penguin:** used through the runners. Their sources, authors and licences are in `THIRD_PARTY.md`.
