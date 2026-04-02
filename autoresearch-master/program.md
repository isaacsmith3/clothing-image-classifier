# autoresearch — clothing condition + fraud detection

This is an experiment to have the LLM autonomously research better models for clothing image classification.

## Problem

Multi-task classification from front + back images of second-hand clothing:
- **Condition** (5-class): grades 1–5 representing clothing quality
- **Fraud detection** (binary): flagging items where condition rating may be fraudulent

**Combined metric:** `combined_score = 0.6 * condition_acc + 0.4 * fraud_f1` — **higher is better**.

## Dataset

- **~31,936 items** in `data/cleaned_metadata.csv` (inside this directory)
- **Images** in the parent project's `data/clothing_v3/` (~32 GB, referenced by absolute paths in the CSV)
- **Split**: 80/20 train/test (pre-split in the `split` column)
- **Columns**: `front_path`, `back_path`, `condition` (1–5), `stains` (0–2), `holes` (0–2), `is_fraud_candidate` (True/False), `split`
- **Class imbalance**: fraud is extremely rare (~0.3% of items). Condition classes are also imbalanced (class 3 has ~9,700 items vs class 1 with ~1,100).

## Hardware

- Apple M3 Max, 36 GB unified memory
- **MPS device** (Metal Performance Shaders) — there is NO CUDA
- MPS constraints: `torch.compile` is risky (may crash), no flash attention kernels, use `torch.float16` (not bfloat16), use `torch.mps.synchronize()` for timing

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar30`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current branch.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data loading, evaluation. **Do not modify.**
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `data/cleaned_metadata.csv` exists and that a sample image path from the CSV is accessible. If not, tell the human.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on MPS. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time). You launch it simply as: `python3 train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, image size, augmentation, loss functions, multi-task weighting, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, and training constants.
- Install new packages or add dependencies. You can only use what's already in `requirements.txt`.
- Modify the evaluation harness. The `evaluate()` function in `prepare.py` is the ground truth metric.

**Model contract:** Your model's `forward(front, back)` must return a dict: `{"condition": tensor(B, 5), "fraud": tensor(B, 1)}`. The `evaluate()` function in `prepare.py` depends on this interface.

**The goal is simple: get the highest combined_score.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the image size, the augmentation pipeline. The only constraint is that the code runs without crashing and finishes within the time budget.

**Memory** is a soft constraint. Some increase is acceptable for meaningful score gains, but it should not blow up dramatically (36 GB unified memory shared with the OS).

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

**Domain hints** (things worth trying):
- Fraud class is extremely rare (~0.3%). Consider focal loss, class weights, oversampling, or threshold tuning.
- Condition class 3 dominates. Consider balanced sampling or class-weighted loss.
- Auxiliary tasks (stains, holes) could improve feature learning — the data is in the CSV.
- Front + back images together matter more than either alone.
- Lighter backbones may allow more epochs in 5 minutes; heavier backbones may extract better features in fewer epochs.

## Output format

Once the script finishes it prints a summary like this:

```
=== RESULTS ===
condition_acc: 0.4600
fraud_f1: 0.3200
combined_score: 0.4040
peak_memory_mb: 8192.0
training_seconds: 300.1
total_seconds: 325.9
num_epochs_completed: 3
=== END RESULTS ===
```

You can extract the key metric from the log file:

```
grep "^combined_score:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	combined_score	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. combined_score achieved (e.g. 0.4040) — use 0.0000 for crashes
3. peak memory in GB, round to .1f (e.g. 8.0 — divide peak_memory_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	combined_score	memory_gb	status	description
a1b2c3d	0.2760	8.0	keep	baseline dual-stream EfficientNet-B2
b2c3d4e	0.3150	8.2	keep	add focal loss for fraud + class weights for condition
c3d4e5f	0.2500	8.0	discard	switch to ResNet50 backbone
d4e5f6g	0.0000	0.0	crash	double batch size (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar30`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python3 train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^combined_score:\|^peak_memory_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If combined_score improved (higher), you "advance" the branch, keeping the git commit
9. If combined_score is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
