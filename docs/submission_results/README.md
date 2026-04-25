# ReproPilot Submission Artifacts

Generate the held-out evaluation reports and plot PNGs with:

```bash
python scripts/generate_committed_submission_plots.py
```

Expected judge-facing files:

- `total_reward_before_after.png`
- `verdict_accuracy_before_after.png`
- `failure_type_accuracy.png`
- `evidence_validity_rate.png`
- `fabricated_evidence_rate.png`
- `reward_channels.png`
- `scenario_type_breakdown.png`
- `novelty_calibration.png`
- `checker_usage_rate.png`

The plot writer falls back to a pure-Python PNG renderer if Matplotlib or its
native dependencies are unavailable.
