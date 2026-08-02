# A–Z Build & Publication Plan
## Terrain-Aware Multimodal Flood GNN — Sri Lanka Early-Warning Benchmark (CS3631)

Every phase below follows the same structure: **what you do → why you do it → what it buys you.** The plan is
sequential — each phase assumes the previous ones are done — and it's designed so that by the time you reach the
research-paper phases (T onward), you're assembling a paper from evidence you already generated, not writing one
from scratch under deadline pressure.

**Context that shapes this plan:** Sri Lanka is currently under a Super El Niño advisory for 2026–2027 — drought
risk July–September 2026, followed by elevated flood risk (particularly eastern and northern river basins) from
around October 2026 through early 2027, per Sri Lanka's Department of Meteorology and multiple current reports. This
is real, current, and directly relevant to your motivation section: it is the actual reason a Kelani-focused
early-warning benchmark, built and validated *now*, is a genuinely useful prototype rather than a purely academic
exercise — your Phase 3 deadline lands well before the forecast flood-risk window opens, which is a legitimate,
citable argument for urgency in your paper's introduction. Be precise about this in writing: it is a *forecast risk
period*, not a confirmed event, so frame your motivation as "the elevated flood risk anticipated for late 2026" with
a citation, not as a specific predicted flood.

---

## PHASE A — Problem framing & research question lock-in

**What:** Write a one-paragraph, unambiguous statement of the research question before touching code: *"Does a
terrain-conditioned, optionally SAR-augmented spatiotemporal GNN improve probabilistic flood early-warning skill and
calibration over non-graph and non-multimodal baselines, for Sri Lankan river basins, under leakage-controlled
evaluation?"*

**Why:** CS3631 explicitly requires "problem motivation" as the first proposal section, and every later phase
(architecture choices, ablations, paper structure) needs a fixed target to be judged against. Without this locked
early, teams drift — someone adds a feature "because it might help," nobody can say whether it answered the actual
question.

**Benefit:** Every subsequent phase becomes a checkable box against this sentence. It also becomes your abstract's
first sentence almost verbatim, so this is not throwaway process — it's paper content, written first.

---

## PHASE B — Literature grounding (done)

**What:** The three core papers (FloodGNN-GRU, HydroGAT, Merit of River Network Topology) plus the bonus
physics-informed paper, already reviewed in depth in your earlier document.

**Why:** You cannot claim novelty without knowing what already exists, and CS3631's rubric explicitly grades
"Related Work" as a paper section — this isn't optional background reading, it's a deliverable.

**Benefit:** You already have this. It gives you (1) a defensible novelty statement, (2) a baseline architecture
ladder to imitate, (3) specific ablations to borrow (edge-weighting comparison, topology-value test, worst-node
diagnostic) instead of inventing your own from scratch.

---

## PHASE C — Dataset freeze (= your WP1)

**What:** Lock the exact row count, column set, and file versions you will train on. Remove the 9 incompatible
256×256 SAR frames. Decide and document the 2024-core-panel vs. 2025-SAR-only-edge treatment. Update `README.md` and
`IMAGE_DATASET.md` to remove superseded claims (per your feasibility report's Appendix B).

**Why:** Every number you report in the paper (410,931 rows, 1.907% prevalence, etc.) must be reproducible from a
frozen artifact. If the dataset silently changes mid-project (new API pull, re-run script), your Phase 2 and Phase 3
numbers won't match, and a reviewer/TA can catch this instantly — it's one of the fastest ways to lose credibility.

**Benefit:** A tagged, frozen dataset version (e.g., git tag `data-freeze-v1`) is what makes "Dataset Description" in
your paper a two-sentence, citable fact instead of a moving target you have to re-verify before every submission.

---

## PHASE D — Repository & environment setup

**What:** GitHub repo with the folder structure from `ARCHITECTURE_SPEC.md`, `requirements.txt`, `.gitignore`,
branch convention per `GIT_WORKFLOW.md`.

**Why:** Five people writing training code in five different Colab notebooks with no shared structure is how you end
up unable to reproduce your own Phase 2 numbers by Phase 3, and how individual contribution becomes impossible to
evidence (CS3631 §11 requirement).

**Benefit:** Clean commit history become your evidence trail for the colour-highlighted contribution requirement, and
a shared `configs/*.yaml` means "run the calibration experiment" is one command for anyone on the team, not
tribal knowledge held by whoever wrote it.

---

## PHASE E — Standalone architecture validation (the notebook you just got)

**What:** Run `flood_gnn_colab.ipynb` end-to-end on synthetic data. Confirm every module's output shape matches
spec, confirm the gradient-check warns about no dead branches, confirm loss decreases over the sanity training loop.

**Why:** This is the single cheapest bug-catching step in the entire plan. Shape mismatches, dead gradients (e.g. a
branch that's computed but never reaches the loss), and fusion-dimension ambiguities (the 320-vs-192 issue) are
*much* faster to find on random tensors that finish in 30 seconds than on your real 410,931-row dataset where a
training run takes an hour and the bug might be silent (wrong shape but broadcasts anyway, producing garbage instead
of an error).

**Benefit:** By the time you touch real data, the architecture itself is de-risked. Any remaining bugs are data-
pipeline bugs, which are a different, more tractable category to debug.

---

## PHASE F — WP2: Baseline ladder implementation

**What:** Implement, in order of increasing complexity: persistence → climatology → discharge-percentile rule →
gradient-boosted trees → per-node LSTM/GRU (no graph) → spatial-only GNN. Evaluate each under all three split
protocols (temporal, Gin basin holdout, event GroupKFold).

**Why:** CS3631 mandates a baseline model per data type (§4) before any "novel" architecture work is credited. More
importantly, the Kirschstein & Sun paper's central lesson is that a strong non-graph baseline can beat naive
graph models — if you don't have that baseline number, you cannot tell whether your final architecture is actually
adding value or just adding complexity.

**Benefit:** This produces the first real numbers in your paper's Experimental Setup / Baselines section, and it's
the safety net that lets you honestly report a null result on graph value later without the whole project looking
like a failure — a well-executed baseline ladder is itself evidence of rigor to a reviewer.

---

## PHASE G — Temporal encoder ablation (GRU vs. TCN vs. temporal-attention)

**What:** Train the per-node LSTM/GRU baseline three ways using the config-switchable `TemporalEncoder` from the
notebook, holding everything else fixed. Record PR-AUC, calibration, and training time for each.

**Why:** Your feasibility report leaves this as an open choice (k=7/14/30, GRU/TCN/attention) rather than a
decision. HydroGAT's own ablation found their Transformer temporal encoder beat naive multi-head attention by
double digits once positional encoding was added — so "which temporal module" is not a neutral choice, and CS3631
explicitly prefers Transformers over RNNs where feasible.

**Benefit:** You get to state in the paper, with a number attached, why you picked the temporal module you did —
"Hyper-parameter Tuning experiments" is an explicit required subsection of your Phase 3 paper (§9), so this ablation
directly fills that requirement rather than being extra work.

---

## PHASE H — Static/terrain FiLM branch, validated in isolation

**What:** Confirm FiLM modulation actually changes model behavior — e.g., feed two different terrain vectors for
the same temporal history and confirm the outputs differ meaningfully, not negligibly.

**Why:** FiLM layers can silently collapse to near-identity (gamma≈1, beta≈0) if terrain features are weakly
predictive or the learning rate on that branch is off, giving you a branch that looks present in the architecture
diagram but contributes nothing in practice. This is the kind of thing that only shows up if you specifically test
for it.

**Benefit:** Directly produces evidence for the FiLM-vs-concat ablation flagged in your earlier literature review
(§4 point 4) — "does the terrain conditioning genuinely help beyond just concatenating terrain features" is a clean,
citable ablation result for your paper.

---

## PHASE I — SAR CNN branch, trained on the 321-frame Kelani subset

**What:** Complete the frame cleanup (drop the 9 incompatible 256×256 leftovers, finish `image_dataset.csv`) and
train the SAR CNN branch as a **standalone** classifier first (SAR chip → flood/no-flood at that node/date), before
plugging it into the full fusion pipeline.

**Why:** With only 321 usable frames across 3 nodes, this branch is data-starved relative to the tabular branch's
409,350 samples. Testing it standalone first tells you whether it has *any* predictive signal on its own before you
bury that signal (or lack of it) inside a large multi-branch model where it's hard to diagnose.

**Benefit:** This is the evidence base for your WP5 "honest estimate of incremental SAR value" exit criterion, and
it's the core of your project's most genuinely novel contribution (no reviewed paper combines SAR with a river-graph
GNN) — so this phase deserves disproportionate care relative to its data size.

---

## PHASE J — Graph construction & spatial edge-weighting ablation

**What:** Build `nodes.csv`/`edges.csv` into the PyG graph object. Run the adjacency-type comparison borrowed from
Kirschstein & Sun: no-graph vs. binary adjacency vs. your fixed `exp(-d/40km)` decay vs. fully-learned edge weights.
Report Pearson correlation between learned attention weights and your distance-decay prior, as HydroGAT and
Kirschstein & Sun both did.

**Why:** This is the single most literature-contested design choice in your architecture (§4 of your earlier
review). Running this ablation is not optional polish — it's the difference between claiming "our GATv2 uses
distance-aware spatial edges" as an assumption versus as a tested, defensible design choice.

**Benefit:** Directly produces your WP3 deliverable ("Ablation showing whether topology adds value beyond node
histories") and gives you the specific evidence needed to either defend your graph design against Kirschstein &
Sun's negative result, or honestly report that it doesn't help in your setting — both are publishable outcomes.

---

## PHASE K — Full model integration & end-to-end sanity check on real data

**What:** Swap the notebook's `make_dummy_batch` for your real `DataLoader`. Run one full epoch on a small subset
(e.g., one basin) before attempting the full 51-node, 409,350-sample run. Re-run the Phase E gradient-check on real
data.

**Why:** Real data introduces failure modes synthetic data can't: NaNs from missing values, class-imbalance-induced
gradient explosions, memory overflow from the SAR branch at full batch size. Testing on one basin first means a
crash costs you two minutes, not the two hours a full-dataset run would take.

**Benefit:** By the end of this phase you have a model that trains stably end-to-end on real data — this is the
point where "we built an architecture" becomes "we have a trained model," which is the actual prerequisite for every
phase from here on.

---

## PHASE L — Full training run + checkpointing discipline

**What:** Train the full model under the temporal split protocol, with frequent checkpointing (every N steps, not
just end-of-epoch) given Colab's session-length limits, and with the three train-year cross-validation folds your
report's split protocol implies.

**Why:** Colab free-tier sessions can disconnect without warning. A training run that only checkpoints at epoch end
and disconnects at 90% of an epoch has wasted that entire epoch's compute — a real risk on a dataset this size.

**Benefit:** A completed full training run, with saved checkpoints and a training curve, is the artifact everything
else (calibration, event metrics, ablations) is computed *from* — this is the load-bearing phase of the whole
project timeline, so protecting it with good checkpointing is worth the extra 20 minutes of setup.

---

## PHASE M — Calibration (= your WP4)

**What:** Fit temperature scaling and isotonic regression on the validation block only (never the test block).
Compute Brier score, ECE (15 bins), reliability diagrams, and Brier decomposition. Run a 5-seed deep ensemble for
predictive variance, per your report's recommendation.

**Why:** Your evaluation design is already more rigorous here than any of the 4 papers reviewed in your literature
review — none of them report calibration at all. This is a genuine strength to lean into, not a checkbox.

**Benefit:** Calibrated, honestly-reported probabilities are what make "P(flood in next 24h) = 0.73" a meaningful
number rather than an arbitrary sigmoid output — this is what separates an early-warning *system* from a flood
*classifier*, and it's a clean, self-contained results subsection for your paper.

---

## PHASE N — Event-level and operational metrics (= part of your WP4/WP6)

**What:** Compute POD/recall, FAR, CSI, F1 at validation-selected thresholds; compute the share of your 1,469 flood
events detected with at least one day of lead time, and mean lead time on detected events. **Never report plain
accuracy as a headline result** — your own report already flags this correctly given the 1.9% positive rate.

**Why:** PR-AUC alone tells you ranking quality; it doesn't tell a reader "how many real floods would this system
have caught, and how much warning time would it have given." Those are the numbers that matter for the early-warning
framing your whole project is built around — and they're what a domain reader (irrigation department, disaster
management centre) would actually ask for.

**Benefit:** This produces the results your motivation section (Phase A, and the El Niño context above) needs to
pay off — "the system would have flagged X% of documented flood events Y days in advance" is the sentence your
introduction is building toward.

---

## PHASE O — SAR case study integration (= your WP5)

**What:** Compare no-imagery, SAR-scalar (simple summary statistics from chips), and full SAR-CNN variants,
restricted honestly to the KEL_HAN-focused case study your feasibility report scopes. Report the coverage/presence
mask explicitly alongside any performance numbers.

**Why:** With only 3 nodes having SAR coverage, any claim broader than "in this case study" would be an
overclaim your own report already warns against (§4.3). Being explicit about this scope is what keeps the claim
defensible.

**Benefit:** This becomes a self-contained, clearly-scoped results subsection — "SAR imagery provides [X]
incremental value in a controlled single-site case study" — that is honest, specific, and still genuinely novel
relative to the literature.

---

## PHASE P — External validity check (= your WP6)

**What:** Compare model predictions against the 11 documented flood episodes and their known misses (3 pre-2011
south-western flash floods). Run the Kirschstein-&-Sun-style worst-node investigation: identify your single
worst-performing node, characterize why (single-node basin? headwater? flash-flood-prone?).

**Why:** This is the phase that decides your paper's honest scope: "research and decision-support benchmark" vs.
"operationally deployable system." Your report already draws this line correctly (§2.3 boundary condition) — this
phase is where you generate the evidence that lets you keep defending that line rather than overclaiming under
pressure to show a "finished product."

**Benefit:** A clearly-scoped, evidence-backed limitations section is what reviewers reward, not penalize — papers
that overclaim get rejected on exactly this point far more often than papers that are precise about their own
boundaries.

---

## PHASE Q — Consolidate results into tables & figures

**What:** One master results table (all baselines + full model, all three split protocols, all headline metrics).
One calibration figure (reliability diagram). One ablation table (temporal encoder choice, edge-weighting choice,
graph-value ablation, FiLM ablation, SAR ablation). One event-detection table.

**Why:** CS3631's paper structure (§9) requires distinct "Ablation Studies," "Comparative Analysis," "Hyperparameter
Tuning," and "Computational Analysis" subsections — if your results are scattered across a dozen separate notebook
runs with no consolidation, writing these sections becomes a scramble at the deadline.

**Benefit:** Once this phase is done, writing the Experimental Setup and Discussion sections of the paper is
largely transcription from these tables, not fresh analysis — this is where the earlier phases' discipline pays
off directly in writing speed.

---

## PHASE R — Computational efficiency analysis

**What:** Report training time per epoch, inference time per prediction, model parameter count, and GPU memory
footprint — explicitly required by CS3631 §9 ("Computational Analysis").

**Why:** This is an easy, often-skipped requirement. It's also a genuine differentiator for an early-warning system
specifically — "how fast can this run when a storm is approaching" is operationally meaningful, not just an
academic formality.

**Benefit:** A short, factual subsection that's nearly free to produce (you already have training logs from Phase L)
but directly satisfies a graded rubric line item.

---

## PHASE S — Internal draft: Phase 1 Proposal (2 pages, due Aug 2, 2026)

**What:** Assemble the ACM template proposal: Problem Motivation (Phase A + El Niño context), Dataset overview
(Phase C), Baseline Model + results (Phase F), Proposed Contribution (your novelty statement from the earlier
literature review, §5), Experimental Plan (Phases J, M, N, O as your planned ablations).

**Why:** This is the actual near-term deadline. Everything above through Phase F is realistically what needs to be
*done* by Aug 2; Phases G onward can be described as planned work in the proposal and executed afterward.

**Benefit:** Because Phases A–F generate real content (not projections), your Phase 1 proposal is evidence-backed
rather than aspirational — which is exactly what "assess feasibility, scope, and suitability" (the proposal's stated
purpose per the module brief) rewards.

---

## PHASE T — Phase 2: Short paper (4 pages, due Aug 23, 2026)

**What:** Proposed architecture description (your full spec), initial evaluation (Phases K–N results, even if
partial), comparison with baseline (Phase F numbers).

**Why:** This is the checkpoint where "does the model actually train and produce sane numbers" gets externally
verified via TA/lecturer review — catching a fundamental problem here is vastly cheaper than catching it at Phase 3.

**Benefit:** Feedback from this submission is your last structured opportunity to redirect before the final paper —
treat any TA feedback here as a required-not-optional input to Phase U below.

---

## PHASE U — Phase 3: Complete paper, LaTeX, code (due per your module timeline)

**What:** Full paper with all required sections (Abstract, Introduction, Related Work, Proposed Framework,
Experimental Setup, Discussion, Conclusion, References), colour-highlighted per-author contribution version, clean
conference-ready version, complete code + LaTeX zip.

**Why:** This is the deliverable graded most heavily (34% of project marks) and the one with authorship-order and
acceptance-outcome stakes attached (100%/95%/≤84%/≤50% tiers depending on submission/acceptance outcome per §10).

**Benefit:** Because every prior phase produced a specific, reusable artifact (tables, ablation results, a locked
dataset, a defensible novelty statement), this phase is assembly and polish, not first-draft generation under
maximum deadline pressure — which is the entire point of front-loading the plan this way.

---

## PHASE V — Internal review & colour-highlighting for submission

**What:** Assign one colour per team member across the paper, add margin comments distinguishing "wrote this text"
from "implemented this method" where they differ, per CS3631 §11.

**Why:** Explicit module requirement, and it's also just good practice for a 5-author paper — reviewers (and your
lecturers) need to see individual contribution clearly, and Git history (Phase D) plus this colour-coding together
make that case robustly.

**Benefit:** Protects every team member's individual mark, independent of the paper's overall acceptance outcome.

---

## PHASE W — Conference selection & submission

**What:** Choose a target conference/venue matched to your paper's actual scope (a benchmark + methodology paper on
Sri Lankan flood forecasting fits regional/applied-ML or hydroinformatics venues well; be realistic about "high-rank
/ high-h-index" vs. "any recognised conference" per the grading tiers in §10 of your module brief). Format to that
venue's template. Submit.

**Why:** Venue choice affects both the page limit/template you need and, per §10, your maximum achievable grade —
this decision has real stakes beyond "which conference sounds good."

**Benefit:** A submission that matches its venue's actual scope and rigor bar has a materially better acceptance
chance than one submitted reflexively to the most prestigious-sounding option regardless of fit.

---

## PHASE X — Address reviewer feedback / notification

**What:** Respond to reviews if given the chance (major/minor revision), or, if rejected, revise per your
lecturers' rubric-based review as CS3631 §10 describes for the "Paper Rejected" tier.

**Why:** Both paths are explicitly built into your module's grading structure — neither outcome ends your grading
process, so treat this as a planned phase, not a contingency.

**Benefit:** Rejection with lecturer review still caps at 84% rather than failing — knowing this in advance should
reduce panic if it happens and keep the team focused on responding to feedback constructively rather than treating a
rejection as a dead end.

---

## PHASE Y — Post-acceptance: conference presentation / camera-ready

**What:** If accepted, prepare the camera-ready version and any required presentation materials by the notification
deadline (Nov 15, 2026 per your module brief).

**Why:** This is the final formal checkpoint in the CS3631 grading pipeline.

**Benefit:** Closes the loop — your project moves from "coursework" to "a published, citable research contribution,"
which is directly useful for every team member's academic record going forward.

---

## PHASE Z — Beyond the course: the actual early-warning use case

**What:** With the benchmark validated and the paper submitted, the natural next step — outside CS3631's scope, but
worth naming since it's the real motivation — is connecting your validated model to the operational gap your own
feasibility report identifies (§7 Roadmap): Irrigation Department gauge integration, observed-inundation
cross-checks, and the calibrated-alert-threshold work needed before any operational pilot is defensible.

**Why:** Your feasibility report is explicit and correct that this project supports a "go" decision for benchmark
completion and model validation, **not** a go decision for public operational deployment (§2.3). Given the current
Super El Niño advisory and its forecast elevated flood risk for late 2026 onward, there's genuine value in being
honest about exactly where this line sits — a validated benchmark is a real and useful step, and overstating it as
"ready to deploy" would be both inaccurate and potentially unsafe if anyone acted on that claim.

**Benefit:** Framed this way, your project is simultaneously (a) a rigorous, publishable academic contribution
and (b) an honest, responsibly-scoped first step toward something that could genuinely matter for Sri Lankan flood
risk management — which is a stronger, more credible story for your paper's Conclusion/Future Work section than an
inflated deployment claim would be.