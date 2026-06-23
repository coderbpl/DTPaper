# Pushing DarkTrace Phase 1 to GitHub

> **Security first:** the token you pasted in chat should be **revoked and regenerated**
> (GitHub → Settings → Developer settings → Personal access tokens). Never commit a
> token or paste it into a file. The steps below read it from an environment variable
> for a single push and never store it.

Target repo: `https://github.com/coderbpl/DTPaper`

## 1. One-time: put the (fresh) token in your shell, not in any file

```bash
# generate a NEW token first; do not reuse the one shared in chat
export GH_TOKEN="ghp_your_fresh_token_here"
```

## 2. Initialise and commit (from the project root)

```bash
cd darktrace_phase1
git init -b main
git add .
git status                      # confirm no token/secret is staged
git commit -m "DarkTrace Phase 1: traffic + text classification pipeline"
```

## 3. Push using the token via env var (single use)

```bash
git remote add origin "https://${GH_TOKEN}@github.com/coderbpl/DTPaper.git"
git push -u origin main

# immediately scrub the token from the remote URL so it isn't saved in .git/config
git remote set-url origin "https://github.com/coderbpl/DTPaper.git"
```

After this, `git pull`/`git push` will prompt for credentials normally (or use the
GitHub CLI / SSH keys, which are safer than embedding tokens).

## 4. Clear the token from your shell session

```bash
unset GH_TOKEN
```

## Safer alternatives to a PAT-in-URL

- **GitHub CLI:** `gh auth login` then `gh repo create coderbpl/DTPaper --source=. --push`
- **SSH:** add an SSH key to GitHub and use `git@github.com:coderbpl/DTPaper.git`

Either avoids putting a token on the command line at all.
