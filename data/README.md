# Data files

These files hold the measurements that the thesis tables and figures are built
from. `scripts/export_data_package_v2.py` wrote them from the scored answers on
the cluster.

## `measurements.csv`

One row per measurement, 10,350 rows in long format.

| Column | Meaning |
|---|---|
| `arm` | `short_answer` or `long_form` (the biography study) |
| `condition` | prompting condition: `chat_0shot`, `default_0shot`, `chat_5shot`, `default_5shot`, or `paragraph` for biographies |
| `instruction` | `chat` (brief sentence) or `default` (as briefly as possible) |
| `n_demonstrations` | number of worked examples, 0 or 5 |
| `split_role` | `eval` for short answers. In the biography rows it is not a split and can be ignored |
| `dataset` | `triviaqa`, `nqopen`, `svamp` or `bio` |
| `model` | the generator |
| `seed` | 42, 43 or 44 |
| `temperature` | sampling temperature of the ten samples |
| `entailment_backend` | the model that groups answers by meaning, or `exact-match` |
| `grader` | correctness rule: `squad_token_f1`, `llm_qwen2.5-72b` or `llm_llama-3.1-70b` |
| `method` | the uncertainty score |
| `n_records` | questions in the cell |
| `mean_answer_chars` | mean length of the graded answer. Only comparable between short-answer rows at temperature 1.0 |
| `accuracy` | fraction of answers graded correct |
| `auroc` | AUROC of the uncertainty score against incorrect answers |
| `aurac_paper`, `aurac_mean_retained` | rejection-accuracy summaries, not used in the thesis |

The `method` values are `discrete_semantic_entropy`, `semantic_entropy_full`
(likelihood-weighted), `surface_entropy` (normalized strings),
`naive_sample_entropy` (raw strings), `naive_entropy` (sequence likelihoods),
`ptrue_uncertainty`, `probe_uncertainty` (semantic entropy probe) and
`accuracy_probe_uncertainty`. The biography rows also contain
`correctness_score`, which is the judge's own score and not an uncertainty
score.

Not every combination exists. The two added prompting conditions have one
temperature and one entailment model. Temperatures below 1.0 cover only the
entropy scores. The probes are stored under `nli-deberta-v3-large` and P(True)
under `deberta-v2-xlarge-mnli`. The biography rows use only the Qwen grader.

## Other files

- `seed_variability.csv`: the AUROC range across the three seeds, averaged
  over dataset and generator.
- `label_agreement.csv`: accuracy under the two LLM judges and how often they
  agree, per cell. It was computed before the TriviaQA duplicates were removed,
  so use it for the agreement rate only.
- `dedup_triviaqa.json`: the TriviaQA questions removed because they appear in
  both splits or more than once.
- `raw/`: two tiny example datasets used by the tests.
