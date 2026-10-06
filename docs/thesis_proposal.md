# Thesis Proposal — The Class-Level Cost of Protection and Exclusion in Federated Medical Imaging

*27 September 2026 · Umut Kızıltoprak, Şevval Gürler, Bülent Sürer*

> This file is the markdown version of the proposal PDF. Two corrections relevant to implementation are marked as `[Note: …]`.

## Problem and claim

**Claim:** The assumption in federated learning that "a deviating client is suspect" also punishes honest hospitals
that carry rare diseases in medical imaging; nobody currently sees this cost.

Cross-hospital federated learning is going into production in 2026: the NCI's FLAIMME consortium in the US, the FLIP
platform in London, a nationwide histopathology deployment in Germany. The common problem of the teams running these
networks is operational: when something goes wrong, understanding what happened without seeing the data.

Federation tools treat deviation as a fault. Robust aggregation (Trimmed Mean, Krum) discards deviating updates;
debugging tools flag a client that deviates from normal as disruptive. Yet in medical imaging deviation is often
information: a hospital that uses a different device or sees a rare disease more often naturally deviates. Excluding it
barely changes overall accuracy, but can collapse the detection rate of that disease.

This thesis develops a working system that measures, without touching the data, which disease class it costs to exclude a
hospital or to turn on a protection (robust aggregation, differential privacy), and shows this at decision time.

## Research questions

1. **RQ1 — Cost of protection.** How do robust aggregation and differential privacy affect per-class recall on
   Fed-ISIC2019? Is the cost concentrated in rare classes?
2. **RQ2 — False exclusion.** How often do deviation-based defenses and fault detection exclude honest but different
   hospitals (different device or class distribution)? What is the class-level cost of this false exclusion?
3. **RQ3 — Foresight.** Can the class-level effect of excluding a hospital be predicted accurately enough during
   training, without touching raw data, before the decision is made?

RQ1 and RQ2 measure the problem; RQ3 forms the basis of the application. Even if the answer to RQ3 is negative, the system
keeps working by measuring and reporting the real cost after exclusion.

## Place in the literature

The two halves of the problem are known separately, but have not been combined and brought to the operator's decision
moment. Our contribution is not a new method, but a system that measures and makes usable these two pieces of
knowledge on real federated medical data.

| Work | What it does | Difference from this thesis |
|---|---|---|
| Tolerating Outliers (IJCAI 2024) | Shows that Byzantine defenses rely on the IID assumption and, on non-IID data, even without an attack, lose performance by discarding honest outlier clients; proposes a new penalized aggregation | Proposes a new rule; does not address which class the loss falls on or how an operator would see it |
| GAS (ICML 2023) | Explains why robust aggregation degrades under non-IID (curse of dimensionality, gradient heterogeneity), proposes a method adapting existing rules | Method paper; synthetic CIFAR-like splits |
| Maverick-Aware Shapley / FedMS (2024) | Measures the value of clients holding rare classes ("Maverick") with per-class Shapley values and uses it in client selection | Measures contribution for selection and reward; does not audit the exclusion decisions of defenses |
| CSSV (IJCAI 2024) | Outputs client contribution as a heatmap with class-specific Shapley values | Contribution measurement; no defenses or DP |
| FedDebug (ICSE 2023) | Finds the deviating (faulty) client from neuron activations without seeing data or labels and continues training without it | The "who is faulty?" question; does not ask the cost of exclusion |

**Gap.** Per-class client valuation and robust aggregation's exclusion of honest outlier clients are known in separate
literatures. We found no study or tool that combines the two and answers "if this hospital leaves, which class loses
what" at the moment of the exclusion decision, on naturally federated medical data, with DP active. Our survey is not
exhaustive; the first task of the first month is to widen it.

**Practical advantage.** With only 6 clients in Fed-ISIC2019, all subsets make 2⁶ = 64 coalitions. Per-class Shapley values
can be computed exactly without approximation and used as an accuracy reference for the console's fast exclusion preview.

## Data and method

The main dataset is Fed-ISIC2019 (FLamby): dermoscopy images split by real centers.

| Property | Value |
|---|---|
| Clients | 6 centers; three from the same hospital (Vienna), with different devices |
| Images | 23,247 |
| Task | 8-class skin lesion classification |
| Class imbalance | Class frequency between 49% and under 1% |
| Size imbalance | The largest center holds more than half the data; the smallest has 351 training images |
| Standard metric | Balanced accuracy (mean of per-class recall) |
| License | CC BY-NC 4.0 (non-commercial) |

**Defenses.** Aggregation: FedAvg (baseline), Trimmed Mean, Krum. Privacy: record-level DP-SGD with Opacus,
ε ∈ {∞, 8, 3}. Attack: label flipping at one center; attack-free runs as controls.

**Metrics.** Per-class recall and PR-AUC; class-level cost caused by defense or exclusion (difference from the
undefended baseline); error of the exclusion preview against the exact per-class Shapley value (RQ3).

**Compute plan.** Features are extracted once with a pretrained image model; in federated training only the classifier
head (and if needed the last block) is trained. Runs are fast on a single GPU or even CPU, many seeds can be tried, and
DP-SGD's per-sample clipping becomes cheap. A few end-to-end (EfficientNet-B0) runs are added to validate the findings.

## Application: Federation Health Console

A working system for the team operating the federation, plugged into Flower as an extension.

1. **Privacy-friendly telemetry.** Each hospital sends only a summary each round: per-class recall on its own
   validation data, update norm, and, if DP is on, the per-class fraction of clipped samples. No images or labels are sent.
2. **Exclusion preview.** When a rule or operator wants to exclude a hospital, the server builds an alternative model without
   that hospital, has the other hospitals evaluate it, and shows the per-class loss before the decision. It runs only at
   the moment of exclusion.
3. **Deviation explanation.** When a client deviates, the console shows which classes the deviation comes from; if the
   deviation comes from rare classes that are over-represented at that hospital, it raises a "different but possibly valuable" warning.
4. **Protection cost panel.** With DP and robust aggregation on, shows the per-class cost round by round and the
   ε spent.
5. **Audit report.** Which hospital was excluded in which round and why, and what the class-level cost was.
   Usable as an audit trail for high-risk medical AI.

A fault detector such as FedDebug can be plugged into the console as input: it says "who deviates", the console says
"what it costs to drop them".

**Demo to be shown at the defense.** One center gets a label-flipping attack, another gets a natural device difference.
Robust aggregation excludes both. The console shows that the first harms all classes, while the second carries a rare
class; the operator excludes one and keeps the other.

> [Note: The original text says "Trimmed Mean excludes both". Trimmed Mean is coordinate-wise and does not drop a
> client entirely; it is Krum that excludes a whole client. The demo should be built with Krum, or via the fraction of
> trimmed coordinates for Trimmed Mean.]

## Schedule, work split and scope

**First term — design and prototype.** Widen the literature survey; freeze the experiment matrix; feature extraction
pipeline; an end-to-end working federation on Flower with FedAvg and per-class metric output; exact Shapley computation
for a single exclusion scenario. By the end of the term the first RQ1 results are in hand.

**Second term — system.** Defenses and DP; telemetry agent; exclusion preview and deviation explanation; console and
audit report; full experiment matrix; writing and open-source release. The console comes last: a thesis without an
interface is still complete, a thesis without results is not.

> [Note: The schedule was moved forward at the advisor's request; see `prototype_plan.md` for a one-week prototype plan.]

| Member | Responsibility |
|---|---|
| Umut Kızıltoprak | Federation, defenses, DP, attack scenarios |
| Şevval Gürler | Telemetry, exclusion preview, Shapley reference, analysis |
| Bülent Sürer | Console, alerts, audit report, demo |

**Out of scope:** proposing a new aggregation rule or a new DP mechanism; deployment to a real hospital network
(Flower's distributed mode is shown on a local network); additional datasets; user accounts and multi-tenant infrastructure.

## Risks and open questions

- **The phenomenon itself is not new.** It is known that robust aggregation excludes honest outliers. Our contribution is
  to measure it at class level, on natural federated medical data, and to make it visible at decision time.
- **Few clients.** With 6 clients, Krum tolerates at most 1 malicious client under its guarantee (n > 2f + 2). This will
  be written up as a finding.
- **Compute.** End-to-end image training is heavy; the feature-extraction approach solves this, but applying DP only to the
  trained part is a limitation.
- **Cost of the exclusion preview.** Each alternative model requires an extra evaluation round; so it runs only at decision time.
- **License.** CC BY-NC 4.0; suitable for the thesis and the open-source tool, not for commercial use of the trained
  model. The tool is independent of the dataset.

## References

- Industry: FLAIMME, NCI (NVIDIA FLARE Day 2026); FLARE Day 2026 program (FLIP); nationwide federated histopathology in Germany
- Data: FLamby (NeurIPS 2022); Fed-ISIC2019 data card
