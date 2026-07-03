# How To Use Codex Agents

Use the main thread for decisions, requirements, and final review. Use parallel agents only for independent read-heavy work such as source research, metric auditing, public notebook triage, or reviewing a diff.

## Good Agent Splits

- Metric auditor: compare `src/biohub_ct/metrics/` against organizer code.
- Data/EDA: inspect Zarr/GEFF metadata and summarize sample statistics.
- Classical baseline: tune detection/linking parameters.
- Notebook packager: verify offline Kaggle path.
- Reviewer: look for schema, runtime, and metric regressions.

## Avoid Context Rot

- Give each agent one bounded task and expected output.
- Ask for file/line evidence and commands run.
- Do not let multiple agents edit the same files concurrently.
- Merge only after tests and validator pass.

## Worktrees

Codex docs describe worktrees as isolated copies for parallel repo work. Use worktrees for competing implementations or risky model experiments. Keep the local checkout for the validated baseline path.

Sources: OpenAI Codex docs and app announcement, plus AGENTS.md convention.

