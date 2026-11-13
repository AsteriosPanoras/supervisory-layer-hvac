# 10 extension_ablation_oracle_rules

**Question.** How the layer improves and extends: (A) which rule gives the gain on the neural controller, (B) the value of knowing the learned start's time constant from day 1 (an oracle), (C) a third rule (the unoccupied setup) added to the same template.

**Where in the paper.** Sections 5.2 and 5.4; Appendix J (Tables 4, 29).

**Outcome.** (A) On the neural controller the pressure reset alone gained 2.99 % [1.60, 4.53], all 10 composite architectures better, and the learned start alone 0.49 % [-0.10, 1.03]; the start brought both losses of the full layer (52209 and 52211). (B) Knowing the time constant from day 1 added 0.02 % [-0.003, 0.057] over the layer: the value of knowing it early, not an upper bound. (C) The unoccupied setup added to the layer gained 0.38 % [-0.45, 1.49], lowering energy and carbon and raising the comfort loss. Every registered prediction held.

**Files here.** PROTOCOL.md (registered before any run), EXECUTION_NOTES.md (including the machine change and the slot rule of A, both recorded before its outcome), code with the analyses (p289, p290, p291), manifests and job lists.

**Layout.**
- `PROTOCOL.md`: what was fixed before the runs. Later amendments are separate files, each registered before any outcome.
- `analysis/`: the analysis scripts, hashed before the runs they read.
- `results/`: their outputs.
- `execution/`: job lists and launch scripts.
- `EXECUTION_NOTES.md`: events during the runs, recorded as they happened.
