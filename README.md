# Selecting What to Remember

**Validation-Gated Memory for LLM Review of Reinforcement Learning Trading Decisions**

Nidhish Gautam, Quanyan Zhu, and Mirela M. Ivan  
New York University, Tandon School of Engineering

[Open the experiment notebook](notebookb9d80604df%20%283%29.ipynb)

Can a frozen language model improve an existing trading policy by learning which experiences to remember, without updating any model weights? This repository contains the experimental notebook accompanying our study of a memory-augmented LLM reviewer for daily Ethereum trading.

A trained MLP-PPO ensemble proposes an exposure. The reviewer can keep, halve, or flatten that proposal. It learns through reusable text instructions generated from historical review outcomes, screened for consistency, and tested on a later validation period. The selected instruction is then frozen for final evaluation.

**Research status:** this is an exploratory pipeline prototype. The notebook explicitly records known future-label leakage in the inherited base model and prior inspection of the original TEST period. Reported returns are prototype replay measurements, not evidence of an executable, leakage-free trading strategy. See [limitations](#limitations).

## How it works

### Review an existing proposal

The base system combines an MLP's signed market-direction signal with the mean exposure from three PPO policies. The fixed ensemble weights are approximately 0.2792 for the MLP and 0.7208 for PPO.

The frozen reviewer receives model outputs, selected market features, its own portfolio state, and Python-computed costs for each permitted action. It returns an action, an explanation, and references to supplied evidence and memory.

| Action | Target exposure |
| --- | --- |
| `KEEP` | 100% of the new base proposal |
| `HALVE` | 50% of the new base proposal |
| `FLAT` | Zero |

`KEEP` means following the new proposal; it can require a trade. The reviewer cannot increase the proposal's magnitude or reverse its direction. All conditions receive the same reconstructed base proposals, while each maintains an independent portfolio.

### Generate, screen, and validate memory

1. **Collect experience:** run the reviewer with empty memory during `MEMORY_BUILD`. Save the decision-time state, action, explanation, realized outcome, and one-step counterfactual returns for all three actions.
2. **Generate candidates:** reflect on chronological batches of 14 decisions, with a final batch of seven. The recorded run produced 20 candidate instructions from 273 decisions.
3. **Screen the evidence:** an offline, Codex-assisted review admitted five candidates, excluded eight for semantic issues and three for redundancy, and deferred four. This was an assisted research step, not an automated validator.
4. **Run the validation gate:** replay each admitted instruction separately on `MEMORY_GATE`, using the same proposals and accounting rules.
5. **Freeze the selected memory:** retain at most one qualifying instruction, then evaluate it without further memory updates on `FROZEN_EVAL`.

Counterfactual feedback compares the next interval's actions from the same observed account state. It does not simulate the complete future trajectory of every alternative action. Future outcomes belong to the evaluator and reflection stages, not the decision-time reviewer inputs.

### Validation rule

Let `S0`, `D0`, and `G0` be the no-memory reviewer's validation Sharpe ratio, maximum drawdown as a fraction, and total log growth. A candidate qualifies only if all three conditions hold:

| Metric | Requirement |
| --- | --- |
| Annualized Sharpe | `S >= S0 + 0.10` |
| Maximum drawdown | `D <= D0 + 0.01` |
| Total log growth | `G >= 0.90 * G0` when `G0 > 0`; otherwise `G >= G0` |

Among qualifying candidates, select the highest Sharpe, breaking exact ties by lower drawdown and then candidate identifier. If none qualifies, retain empty memory. These are selection thresholds, not statistical significance tests; the reference is the **no-memory reviewer**, not the unreviewed base model.

The recorded run selected **B013: "Cross-check model agreement before full exposure."** B013 asks the reviewer to consider agreement among the MLP and PPO components alongside proposal magnitude and the supplied action costs. Its wording remained unchanged during final evaluation.

## Experimental periods

The export contains 3,243 daily feature rows spanning November 9, 2017 through September 25, 2026. The memory experiment uses the following inclusive decision-date ranges:

| Stage | Decision dates | Decisions | Purpose |
| --- | --- | ---: | --- |
| `MEMORY_BUILD` | 2023-10-01 to 2024-06-29 | 273 | Collect outcomes and generate candidates |
| `MEMORY_GATE` | 2024-07-01 to 2024-12-30 | 183 | Validate and select memory |
| `FROZEN_EVAL` | 2025-01-01 to 2026-09-24 | 632 | Evaluate frozen conditions |

June 30 and December 31, 2024 are buffer dates with no new decisions. September 25, 2026 is excluded as a decision because the exact next calendar day's price is missing. These splits separate memory development and selection; they do not resolve the inherited base model's data-validity issues.

## Reported prototype results

The following six conditions appear in both the manuscript and the notebook's saved final-evaluation output. Net return is cumulative over the 632-decision period; Sharpe is annualized using 365 days.

| Condition | Memories | Sharpe | Net return (%) | Maximum drawdown (%) |
| --- | ---: | ---: | ---: | ---: |
| Base model, full exposure | 0 | 5.638 | 10,009.31 | 18.59 |
| Base model, half exposure | 0 | 5.556 | 959.25 | 9.61 |
| LLM without memory | 0 | 5.003 | 4,292.64 | 17.49 |
| LLM with raw memory | 20 | 5.254 | 1,019.82 | 10.89 |
| LLM with screened memory | 5 | 4.996 | 1,530.85 | 9.61 |
| **LLM with gate-selected memory** | **1** | **5.763** | **10,762.61** | **18.59** |

In this replay, gate-selected memory improved Sharpe by approximately **2.2%** relative to the full-exposure base model. Raw and screened memory reduced exposure much more frequently and underperformed the base model in Sharpe. The selected-memory reviewer chose `KEEP` on 552 days and `HALVE` on 80; none of the four LLM conditions chose `FLAT`.

The manuscript additionally reports model-disagreement and high-volatility rules introduced after inspecting the original final-evaluation results. Those exploratory benchmarks are not part of the uploaded notebook's six-condition evaluation.

### Replay settings

- Each account starts flat with USD 100,000 and is liquidated at the end of its replay period.
- Daily close-to-next-calendar-day-close returns; fee of 0.10% plus slippage of 0.05% on traded notional.
- Recorded LLM configuration: DeepSeek API identifier `deepseek-flash`, thinking disabled, temperature `0`, and up to 512 completion tokens per review.
- Decisions remain chronological within each account; independent accounts can run concurrently.
- Requests, responses, protocols, memory artifacts, and daily account states are checkpointed, with resumable runs and ZIP backups.

## Repository contents and running the notebook

The repository contains this README and [the original notebook with saved outputs](notebookb9d80604df%20%283%29.ipynb). The source dataset, trained weights, full API-response cache, and experiment backup ZIP are **not included**.

### Required inputs

The notebook is written for Kaggle and discovers `llm_export_summary.json` beneath `/kaggle/input`. Attach the original `models-and-data` export, preserving its directory layout. Required files include:

| Location | Required content |
| --- | --- |
| `llm_export/llm_export_summary.json` | Feature order, ensemble weights, and export metadata |
| `llm_export/archive/` | `model_inputs_raw.csv`, `model_signals_saved.csv`, `existing_splits.csv` |
| `llm_export/reviewer_candidates/` | `features.csv` |
| `llm_export/evaluator_only/` | `prices.csv`, `outcomes.csv` |
| `llm_export/models/` | `mlp.pt`, `normalization.npz`, and exported PPO checkpoint directories |
| `setup/config.json` beside `llm_export/` | Replay configuration, including initial cash, fee, and slippage |

The recorded environment uses Python 3.12, NumPy, pandas, PyTorch, Gymnasium, and cloudpickle. Block 6 pins `stable-baselines3==2.9.0` and loads the frozen models on CPU.

### Workflow

1. Import the notebook into Kaggle and attach the required export. Blocks 1-11 inspect inputs, audit timing and labels, reconstruct proposals, and define replay periods.
2. Enable internet access and provide a Kaggle secret named `DEEPSEEK_API_KEY`, or the same environment variable. Block 12 checks model access; subsequent review and reflection cells make paid API calls. The archived request configuration uses a USD 10 accounting guard.
3. Follow the numbered blocks and their prerequisites. Blocks 14-17 contain early pilots; Blocks 18-20 establish cost-aware review and collect the full memory-development experience. Blocks 21-22 generate and inspect candidate memories.
4. Block 23 records the assisted screen and freezes the validation protocol. Blocks 24-25 run the gate and freeze the selected instruction. Block 26 runs the final evaluation; Block 26A is a recovery helper for interrupted requests.
5. Preserve generated artifacts under `/kaggle/working/llm_memory_experiment` and download the `eth_llm_backup_*.zip` archives for later restoration.

This is a saved research workflow, not an unattended end-to-end package. In particular, **Block 23 is bound to the exact evidence-packet hash and candidate IDs from the original assisted review**. A new generation run needs its own evidence review and recorded screening decisions before validation. Do not bypass the check or reuse the original annotations for different candidates. The final download cell also names one historical backup ZIP; use the archive produced by the current run.

## Limitations

- **Inherited label leakage and timing:** the notebook retains the original `prev_lbl` model input and explicitly sets `LLM_CAUSALITY_VERIFIED = False` and `LLM_BASELINE_VALID_FOR_PERFORMANCE_CLAIMS = False`. Keeping labels out of the reviewer's direct inputs does not repair the proposals of a model trained with leaked information. The notebook also records prior TEST inspection and assumes a same-close fill after observing that row's features.
- **Limited evidence:** one asset, one historical path, one validation window, and one retained instruction. The small Sharpe difference is a point estimate without a significance test or multiple-testing correction.
- **Simplified execution:** funding, borrowing costs, interest on idle cash, and the original take-profit/stop-loss mechanics are omitted. PPO proposal reconstruction uses each policy's previous proposal and a recent-return input fixed at zero.
- **Commercial-model uncertainty:** the LLM's training exposure to evaluation-period history cannot be verified, and fresh API responses may differ from the archived run.
- **Assisted selection:** semantic screening was an interactive research step. Its reproducibility and reliability as an automated process remain untested.

Further work includes repairing the base model's causal inputs and execution timing, evaluating multiple assets and rolling windows, and testing whether small sets of complementary memories improve on a single retained instruction.

## Citation

```bibtex
@misc{gautam2026selecting,
  title = {Selecting What to Remember: Validation-Gated Memory for LLM Review of Reinforcement Learning Trading Decisions},
  author = {Gautam, Nidhish and Zhu, Quanyan and Ivan, Mirela M.},
  year = {2026},
  note = {Research manuscript and experimental notebook},
  url = {https://github.com/nidhish1/validated-memory-rl-trading}
}
```
