# Running DarkTrace Phase 1 on AWS Linux (step-by-step)

> **Prefer a script?** Everything below is automated in `deploy/aws_phase1.sh`.
> Run `bash deploy/aws_phase1.sh setup` then `bash deploy/aws_phase1.sh all`
> (place datasets first; see step 5). The manual steps remain here for reference.


This is a concrete, copy-paste runbook for **Amazon Linux 2023** and **Ubuntu on
EC2**. It assumes you have already unzipped the package on the instance and that
`darktrace_phase1/` is your current directory. Commands that differ by distro are
labelled; run the block matching your AMI.

> Phase 1 is CPU-only. A `t3.large` (2 vCPU / 8 GB) is comfortable; `t3.medium`
> works for smaller runs. No GPU required.

---

## 0. (If not already on the instance) connect via SSH

From your local machine:

```bash
# Amazon Linux 2023
ssh -i /path/to/your-key.pem ec2-user@<INSTANCE_PUBLIC_IP>

# Ubuntu
ssh -i /path/to/your-key.pem ubuntu@<INSTANCE_PUBLIC_IP>
```

Then `cd` into the unzipped package:

```bash
cd ~/darktrace_phase1     # adjust if you unzipped elsewhere
```

---

## 1. Install Python, pip, venv, and unzip

**Amazon Linux 2023 (dnf):**
```bash
sudo dnf -y update
sudo dnf -y install python3 python3-pip git unzip
```

**Ubuntu (apt):**
```bash
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv git unzip
```

Check the Python version (3.10+ expected):
```bash
python3 --version
```

---

## 2. Create and activate the virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt should now show `(.venv)`. (On Amazon Linux, if `venv` is missing,
run `sudo dnf -y install python3-venv` or use `python3 -m pip install --user
virtualenv && python3 -m virtualenv .venv`.)

---

## 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Verify the pipeline before touching real data (synthetic, non-reportable)

```bash
python tests/test_smoke.py
```

Expect `ALL SMOKE TESTS PASSED`. This confirms the environment is correct and
writes nothing reportable.

---

## 5. Place the datasets

Target paths: `data/raw/CIC-Darknet2020.csv` and `data/raw/coda.csv`.

### 5a. CIC-Darknet2020 (traffic)

The UNB CIC page requires a browser form, so the simplest path is to download it
on your **local machine**, then copy it up to the instance with `scp`.

Run this **on your local machine** (not the instance):
```bash
# Amazon Linux instance (user ec2-user)
scp -i /path/to/your-key.pem ~/Downloads/Darknet.csv \
    ec2-user@<INSTANCE_PUBLIC_IP>:~/darktrace_phase1/data/raw/CIC-Darknet2020.csv

# Ubuntu instance (user ubuntu)
scp -i /path/to/your-key.pem ~/Downloads/Darknet.csv \
    ubuntu@<INSTANCE_PUBLIC_IP>:~/darktrace_phase1/data/raw/CIC-Darknet2020.csv
```

(The downloaded filename varies; rename to `CIC-Darknet2020.csv` as shown.)

If you have an S3 bucket with the file, you can instead pull it directly on the
instance:
```bash
# requires an IAM role on the instance OR `aws configure`
aws s3 cp s3://your-bucket/CIC-Darknet2020.csv data/raw/CIC-Darknet2020.csv
```

### 5b. CoDA (text) — export from Hugging Face on the instance

```bash
pip install datasets
python3 - <<'PY'
from datasets import load_dataset
ds = load_dataset("s2w-ai/CoDA")
split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
df = split.to_pandas()
print("columns:", list(df.columns))      # note text/label column names
df.to_csv("data/raw/coda.csv", index=False)
print("wrote data/raw/coda.csv", df.shape)
PY
```

Read the printed `columns:` line. The loader auto-detects common names
(text/content/body and label/category/class). If CoDA uses different names, edit
`configs/text.json` and set `text_col` and `label_col` (use a terminal editor):
```bash
nano configs/text.json      # or: vi configs/text.json
```

### 5c. (Alternative) DUTA-10K instead of CoDA

Request DUTA-10K from the GVIS/authors' page, export it to a CSV with a text and a
label column, copy it up with `scp` (as in 5a), save as `data/raw/coda.csv`, or
point `configs/text.json:data.csv_path` at your DUTA file and set
`data.name` to `DUTA-10K`.

### Confirm both files are present
```bash
ls -la data/raw/
```
You should see `CIC-Darknet2020.csv` and `coda.csv` next to `.gitkeep`.

---

## 6. Run the experiments in real-data mode

No `--smoke-test` flag — that is what makes the outputs reportable.

```bash
python -m src.exp_traffic --config configs/traffic.json
python -m src.exp_text    --config configs/text.json
```

Real-data runs log lines such as:
```
[INFO] Loading real CIC-Darknet2020 from data/raw/CIC-Darknet2020.csv
[INFO] Dropping N identifier/leakage column(s): [...]
[INFO] Real-data run complete — Table 6 traffic fragment written.
```

If you instead see `[ERROR] Real ... not found` and the process exits with code 2,
the file path is wrong — recheck step 5. (Hard failure is intentional; the package
never silently substitutes synthetic data in normal mode.)

> Tip: GradientBoosting on the full ~158k-row CIC-Darknet2020 is the slowest step
> and is CPU-bound. To keep it running after you disconnect SSH, use the driver
> script under `nohup` or `tmux` (see "Long runs" below).

---

## 7. Build Table 6 and figures

```bash
python -m src.build_table6
python -m src.make_figures
```

`build_table6` uses only real-data fragments; "No real-data Table 6 fragments
found" means step 6 did not produce real outputs.

---

## 8. Locate the outputs

```bash
ls -la results/tables/      # CSV + JSON
ls -la results/figures/     # PNG
cat results/tables/table6_full.csv
```

Key files: `results/tables/table6_full.csv` (assembled table),
`table6_traffic.csv` / `table6_text.csv` (fragments),
`traffic_results.json` / `text_results.json` (full metrics, CV, CIs, McNemar,
`class_report`), and `results/figures/perclass_f1_*.png`.

---

## 9. Verify outputs are REAL, not SMOKETEST

Run all three checks; each should print nothing (or the expected real-data values):

```bash
# (a) no smoke-test files present
ls results/tables/ | grep SMOKETEST

# (b) no non-reportable markers in the assembled table
grep -i "SMOKE-TEST\|NON-REPORTABLE\|SYNTHETIC" results/tables/table6_full.csv

# (c) provenance flags say real data
grep -E '"synthetic"|"reportable"' \
     results/tables/traffic_results.json results/tables/text_results.json
```

For (c) you want `"synthetic": false` and `"reportable": true`. If any check
flags a file, delete smoke-test artefacts and rerun step 6:
```bash
rm -f results/tables/*SMOKETEST*
```

---

## 10. Copy results back to your local machine

Run **on your local machine**:
```bash
# Amazon Linux
scp -i /path/to/your-key.pem -r \
    ec2-user@<INSTANCE_PUBLIC_IP>:~/darktrace_phase1/results ./results

# Ubuntu
scp -i /path/to/your-key.pem -r \
    ubuntu@<INSTANCE_PUBLIC_IP>:~/darktrace_phase1/results ./results
```

---

## Long runs (survive SSH disconnects)

Use the provided driver script with `nohup` so it continues if your session drops:
```bash
nohup bash deploy/run_all.sh > results/logs/run_all.out 2>&1 &
tail -f results/logs/run_all.out      # watch progress; Ctrl-C to stop watching
```

Or use `tmux`:
```bash
sudo dnf -y install tmux     # Amazon Linux   (Ubuntu: sudo apt-get install -y tmux)
tmux new -s darktrace
# inside tmux: run the experiments; detach with Ctrl-b then d; reattach: tmux attach -t darktrace
```

---

## Cost reminder

`t3.large` is inexpensive per hour, but **stop or terminate the instance when you
are done** to avoid ongoing charges:
```bash
# from your local machine, with AWS CLI configured
aws ec2 stop-instances --instance-ids <INSTANCE_ID>
```

## Security reminders

- Restrict the instance's inbound SSH (port 22) to **your IP only**, never
  `0.0.0.0/0`.
- Do not store AWS keys or GitHub tokens in the repo; use an IAM role for S3
  access.
