# Deploying DarkTrace Phase 1 on AWS Linux

Phase 1 is CPU-only and lightweight, so a small EC2 instance is sufficient. No GPU
is required (that's Phase 2+). This guide covers a minimal, reproducible setup.

## 1. Launch an EC2 instance

- **AMI:** Amazon Linux 2023 or Ubuntu 22.04 LTS.
- **Instance type:** `t3.large` (2 vCPU / 8 GB) is comfortable; `t3.medium` works for
  smaller runs. GradientBoosting on the full CIC-Darknet2020 is the heaviest step.
- **Storage:** 20–30 GB gp3 (datasets + results).
- **Security group:** you only need outbound internet (to clone the repo and pip
  install) and inbound SSH (port 22) **restricted to your own IP**. No other ports.
- **Key pair:** create/download an SSH key for access.

## 2. Connect and clone

```bash
ssh -i your-key.pem ec2-user@<INSTANCE_PUBLIC_IP>     # Amazon Linux
# or: ssh -i your-key.pem ubuntu@<INSTANCE_PUBLIC_IP> # Ubuntu

git clone https://github.com/coderbpl/DTPaper.git
cd DTPaper
```

## 3. Provision the environment

```bash
bash deploy/setup_aws.sh
```

This installs Python/pip/git, creates a virtualenv, installs `requirements.txt`,
and runs the smoke test to confirm everything works.

## 4. Run the pipeline

```bash
bash deploy/run_all.sh
```

Outputs land in `results/tables/` (CSV + JSON) and `results/figures/` (PNG).
Until you add real data to `data/raw/`, results are synthetic and tagged
`(SYNTHETIC)` — see `DATASETS.md`.

## 5. Retrieve results

```bash
# from your local machine
scp -i your-key.pem -r ec2-user@<INSTANCE_PUBLIC_IP>:~/DTPaper/results ./results
```

## Optional: scheduled runs with systemd timer

Create `/etc/systemd/system/darktrace.service`:

```ini
[Unit]
Description=DarkTrace Phase 1 pipeline
[Service]
Type=oneshot
WorkingDirectory=/home/ec2-user/DTPaper
ExecStart=/bin/bash deploy/run_all.sh
User=ec2-user
```

Create `/etc/systemd/system/darktrace.timer`:

```ini
[Unit]
Description=Run DarkTrace Phase 1 daily
[Timer]
OnCalendar=daily
Persistent=true
[Install]
WantedBy=timers.target
```

Enable:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now darktrace.timer
sudo systemctl list-timers | grep darktrace
```

(Or use a simple `cron` entry: `0 3 * * * cd /home/ec2-user/DTPaper && bash deploy/run_all.sh`.)

## Cost note

A `t3.large` is inexpensive per hour; **stop or terminate the instance when idle** to
avoid ongoing charges. Phase 1 runs in minutes on real data, so an on-demand instance
that you stop afterward is the cheapest pattern.

## Security reminders

- Restrict SSH to your IP; never open it to `0.0.0.0/0`.
- Do not store GitHub tokens or AWS keys in the repo. Use IAM roles for AWS access
  and the GitHub CLI/SSH keys for git (see `PUSH_INSTRUCTIONS.md`).
- The shared chat token must be revoked and regenerated.
