# The supervisory layer

The layer leaves the controller's code unchanged. It sets a few of the controller's values, and only within fixed bounds.

Each rule has the same five parts:
1. the signal it reads;
2. the condition that triggers it;
3. the command or value it may change;
4. the bounds of that change;
5. the condition that hands control back to the controller.

A new rule is added by naming the loss term it targets. The headroom account of Section 5.3 tells in advance where it can pay.

| Rule | Targets | Reads | Acts on | Bounds | Learns | Needs from the controller |
|---|---|---|---|---|---|---|
| Learned optimal start | comfort at opening | zone temperatures before opening | the controller's start time | lead within fixed bounds (h) | recovery time constant τ | a start-time setting |
| Pressure reset (trim and respond) | comfort, fan energy | zone requests | floor of the duct static-pressure command | fixed band (Pa) | — (Guideline 36 trim and respond) | access to the final pressure command |
| Learned night precooling | energy | outdoor and zone temperatures before opening | unoccupied cooling ceiling | between the comfort bounds | the building's morning excess | an unoccupied cooling setting |
| Unoccupied setup (Guideline 36) | unoccupied energy | occupancy schedule | unoccupied cooling limit | fixed setpoint | — | an unoccupied cooling setting |

The exact parameters are in `PARAMETERS.md`. `official_interface/` holds the layer in the form used for the leaderboard submissions.
