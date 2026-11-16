# Ball possession from automatic tracking: technical note

More than a Hack 2027, Phase 1. Team: **bueno**.
Code: REPO_URL. One command rebuilds the submitted file byte for byte (see Reproduction).

## 1. Approach in one paragraph

The labels are mechanical: the corrected tracking's "ball alive" flag plus the team that owns the ball. The open
DFL/Sportec Bundesliga data (IDSSE, 7 matches, CC-BY 4.0) has per-frame `BallStatus` and `BallPossession`, which
is the same definition. So we **train only on those 7 matches**, after **degrading their clean tracking until it
looks like BAR-ATH's automatic tracking**. The noise model is calibrated on label-free aggregate statistics of the
BAR-ATH tracking and checked by adversarial validation. A **set encoder over all tracked players plus hand-made
team features feeds a bidirectional GRU** (40 s windows), and it outputs P(H), P(A), P(D) per frame. Six such
networks and a LightGBM model are averaged, with home/away-swap test-time augmentation. A **3-state Viterbi decoder**
then turns the probabilities into possession spells. A **kickoff rule** sets the ball dead before each half's
kick-off tag. No label from the challenge match is used anywhere, including the public 5 minutes.

## 2. Why this and not something else

- **spellF1 rewards whole possessions, not frames.** A model that is right on 95% of frames can still score
  badly if it flickers. So the core is a sequence model plus a decoder that pays a cost to open a new spell.
- **The ball is missing on about 41% of BAR-ATH frames, mostly during slow play and stoppages.** The baseline's
  "no ball = dead" rule is wrong about 41% of the time. Our model reads team shape and player motion, so it
  works when the ball is missing.
- **Labelled data from the target domain is forbidden, but the label definition exists elsewhere.** IDSSE has it
  exactly. The difficulty is the domain gap (clean optical tracking vs. BAR-ATH's automatic tracking), so most of
  the effort went into closing that gap. Model architecture came second.

## 3. Closing the domain gap: the degradation bench

Each IDSSE match is degraded with 8 random seeds × {original, home/away swapped} for training. A separate seed
(99) is used for held-out evaluation, leave-one-match-out. Every parameter of the noise model comes from
**label-free aggregates** of the BAR-ATH tracking (no frame labels):

| Component | What BAR-ATH shows | What we do to IDSSE |
|---|---|---|
| Smoothing / drift | Player jerk about 200× lower than raw optical tracking | Smoothing plus slow positional drift |
| Ball presence | Depends on tempo: 6% at walking pace, about 90% at match tempo | Persistent latent process thresholded at the measured presence-vs-tempo curve, iterated until the bench curve matches within 0.01 |
| Dead-ball ball | Around restarts the ball is visible, about 4 m inside the pitch, moving about 1 m/s | The taker picks up, carries and places the ball (DFL freezes it at the out spot, a giveaway real tracking never shows) |
| Player loss | 7.8 players missing on average at < 0.5 m/s collective speed, about 0.5 at match tempo, losses lasting up to about 20 s | Per-tempo quantiles through a persistent copula, random players |
| Keeper depth | Keepers about 9 m off their line (IDSSE about 15 m) | Depth beyond 6 m scaled by U(0.3, 1) per team and half |

An earlier version tied ball dropout to the true alive/dead state, which **leaked the label**: "ball missing"
predicted D with AUC 0.89. We replaced it with the tempo-driven version above.

**Adversarial validation.** A classifier separating bench frames from BAR-ATH frames went from AUC 0.92 to 0.84
over these steps. Its top features at each stage pointed to the next fix. What remains is mostly team-shape style
(Barça vs. Bundesliga), which is real domain difference rather than a simulation bug. The "kick" features never
fire on the smoothed BAR-ATH tracking, and the match clock was the main remaining real-vs-bench difference, so
both are excluded.

## 4. Model

- **Features (131 per frame).** Team centroids, spreads and extremes. Ball state and time since or until seen.
  Nearest players of each team to the ball. Multi-scale rolling means. Coordinates are normalised so home always
  attacks +x. Direction of play comes from the tracking, per half.
- **Set encoder.** Every tracked player is a token (position, offset to the ball, speed, team). Tokens are pooled
  per team (mean, max, attention) and concatenated with the hand features.
- **Sequence model.** Conv1d, then a 2-layer bidirectional GRU over 40 s windows at 25 fps, with overlapping
  windows averaged at inference. Five of the six networks also have an auxiliary head, "label change within
  ±0.24 s".
- **Ensemble.** 5 init seeds of the auxiliary-head encoder, 1 encoder without it, and LightGBM on the hand
  features. Probabilities are averaged. Each network is also run with home/away swapped and averaged back
  (+0.013 to +0.014 spellF1 per model).
- **Decoder.** Viterbi over {H, A, D}, cost 3 for a change of team and 20 for alive↔dead, chosen on
  out-of-fold bench probabilities only.

## 5. Possession rules implemented

- **Owner, not control.** The label is the team that owns the ball until the other team takes it or the ball
  dies. Loose balls and challenges keep the last owner until the other team clearly takes over. The decoder's
  team-switch cost implements this: a brief touch by an opponent must be supported by enough evidence to open a
  new spell.
- **Dead ball.** Out of play, fouls, and every restart until the ball is played. The D state with its own
  switch cost captures these. Restart realism (ball placed, players slow, players missing) is learned from the
  degraded bench.
- **Kickoff rule.** D from the start of each graded half up to that half's Smart Tagging "Kick Off (Start)" tag.
  This is a football fact, not a tuned value. It changes 709 frames before the second-half kickoff.
- **X** outside the two graded halves (published boundaries). Nothing is abstained inside them.

## 6. Assets used, tried and dropped

| Asset | Use |
|---|---|
| Tactical automatic tracking (ATD) | **Main input** |
| Smart Tagging | **Kickoff rule.** Also a label-free sanity proxy: set-piece windows called D 0.87; team-phase agreement 0.83 / 0.85. Tried as model input (simulated on the bench from DFL restart events): +0.002, dropped. |
| Public 5-minute ground truth | Wiring check only (frames aligned, H/A not swapped), never for choosing or tuning |
| Tactical video | Tried an off-the-shelf YOLO ball detector (40% agreement with the tracking ball) and a self-supervised one. Not used: it could not be validated on the bench, and a teammate's version showed the output file becomes sensitive to tiny float differences. |
| Broadcast video | Not used |

## 7. Training data

| Dataset | Source / licence | Labelled by | Amount | Use |
|---|---|---|---|---|
| IDSSE (DFL Bundesliga) | Bassek et al., figshare / Hugging Face `pysport/idsse-data`, CC-BY 4.0 | DFL's tracking provider (TRACAB `BallStatus`, `BallPossession` flags), automatic plus their QA | 7 matches, about 1.0 M frames per copy × 16 degraded copies | Training and validation |
| Metrica sample games 1-3, SkillCorner open data (20 A-League games) | Metrica Sports; SkillCorner, public GitHub | Event-derived / provider possession groups | 23 games | Independent test bench only (teammate's bench), never trained on |

Nothing from BAR-ATH is labelled or used as a label.

## 8. Results

Leave-one-match-out on the degraded IDSSE bench (seed 99), spellF1:

| Stage | spellF1 |
|---|---|
| Organisers' baseline on the bench | about 0.17 (0.24 on the real match, so the bench is realistic) |
| LightGBM + Viterbi | about 0.58 |
| BiGRU on hand features | 0.63 |
| Set encoder + TTA (single) | about 0.70 |
| Final ensemble, most realistic bench (degrade 6/7) | about 0.69 |

**Independent check on a second bench.** A teammate built a separate bench: 30 games from 3 sources (Metrica's 3
public sample games with event-derived labels, the 7 IDSSE matches, and 20 SkillCorner A-League broadcast games).
It uses a different, ball-only degradation and 5 folds × 3 test seeds. Our final models were frozen, and they never
saw the Metrica or SkillCorner games, so only those count. Its other systems are refitted per fold.

| System (mean spellF1) | Metrica (9 runs) | SkillCorner (60 runs) |
|---|---|---|
| Hand features + logistic regressions + hand-over-gated Viterbi | 0.503 | 0.337 |
| Set-encoder TCN trained on Metrica + IDSSE (per fold) | 0.677 | 0.359 |
| **Ours (submitted)** | **0.710** | **0.373** |

Ours wins 98-100% of paired runs against the first system and 78% against the second (+0.017). SkillCorner labels
follow a different possession definition (provider "possession groups", fewer and longer spells), so every system
scores lower there. Adding either other system to our ensemble as an eighth member (weight 1/8, fixed in advance)
gained only +0.003, so the submission stays as it is.

## 9. Limitations and what we would do next

- **Short possessions.** 30% of possessions last under 2 s. Recall is 0.15 for those under 1 s and 0.42 for
  1-2 s. Missed spells split into wrong team (46%), merged or split (44%) and inside dead time (10%). With the
  ball missing on 41% of frames, that is close to what tracking alone supports. The next lever is our own ball
  detection on the tactical video, validated on a labelled video dataset.
- **Domain style gap.** Barça's positional play differs from the Bundesliga. More varied labelled tracking
  would help.
- **Goal kicks** are called dead in about 67% of their tag windows, compared with 87% for all set pieces.

## 10. Reproduction

```
git clone https://github.com/metrica-sports/more-than-a-hack-2027 repo   # data bundle unpacked into repo/data
git clone REPO_URL possession                                           # next to repo/
cd possession && python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # (bin/ on Linux)
python make_submission.py --team bueno                              # -> out/bueno.csv
```

CPU only and deterministic: about 15-20 minutes on a laptop. Expected sha256 of the file: `b86c6d8d07c7a310ff4337e7dfe84de1d90fefac001ef4b3f7c2b85e2bb3effc`.
Retraining (optional, GPU): `hpc/` holds the exact commands (`seq_set.py --final ...`, `final_lgbm.py --train`),
with the degraded IDSSE copies built by `mtah/degrade.py`.
