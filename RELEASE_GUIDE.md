# FIDES Release & Zenodo Archiving Guide

This guide walks you through creating the GitHub Release v1.0.0 and obtaining
a permanent DOI from Zenodo, as required by the BSPC journal's software
citation policy (Author Guide, pages 23-24).

---

## Step 1: Prepare the Repository

Before creating the release, ensure the following are committed and pushed:

- [x] `README.md` (with version badge, DOI placeholder, new sections)
- [x] `CHANGELOG.md`
- [x] `LICENSE` (MIT)
- [x] `requirements.txt`
- [x] `.gitignore`
- [x] `run_fides.py` (with random seed and updated checkpoint default)
- [x] All `fides/`, `experiments/`, `clinical/`, `benchmark/`, `data/`, `utils/` modules

```bash
# Stage all changes
git add -A

# Commit
git commit -m "Release v1.0.0: Initial public release for BSPC submission"

# Push to GitHub
git push origin main
```

---

## Step 2: Create Git Tag

Create an annotated tag for version 1.0.0:

```bash
git tag -a v1.0.0 -m "FIDES v1.0.0 - Initial release for BSPC submission"

# Push the tag to GitHub
git push origin v1.0.0
```

---

## Step 3: Create GitHub Release

1. Go to https://github.com/tale-bonbom/FIDES/releases/new
2. **Choose a tag**: Select `v1.0.0` (created in Step 2)
3. **Release title**: `FIDES v1.0.0 - Initial Release`
4. **Description** (copy from `CHANGELOG.md`):

```markdown
## FIDES v1.0.0 - Initial Release

This is the initial public release of the FIDES inference pipeline, accompanying
the paper submitted to Biomedical Signal Processing and Control.

### Highlights
- Full implementation of the Training-Inference Consistency Guideline (TICP)
- 38 pipeline configurations across 5 phases
- Phase VI cross-architecture validation (4 architectures)
- Phase VII 2×2 cross-ablation (decisive causal evidence)
- Clinical measurement and statistical analysis utilities
- Error Cancellation Theorem verification code
- FLOPs and latency benchmarking scripts

### Installation
```bash
pip install -r requirements.txt
python run_fides.py --config fides_optimal --input /path/to/acdc --output ./results
```

### Citation
@software{bao2026fides, ...}
```

5. **Attach binaries** (optional): Upload the pre-trained checkpoints as release assets:
   - `mednext_l.pth` (Model-A, CorSeg pre-trained)
   - `mednext_l_acdc_noarp.pth` (Model-B, ACDC-only)
   - `unet.pth`, `basicunet.pth`, `segresnet.pth`
6. **Publish release**

---

## Step 4: Connect Zenodo to GitHub

### One-time setup (if not already done):

1. Go to https://zenodo.org
2. Sign in / create an account
3. Click your profile → "Settings" → "GitHub"
4. Click "Connect GitHub account"
5. Authorize Zenodo to access your GitHub repositories
6. Find `tale-bonbom/FIDES` in the repository list and toggle it **ON**

---

## Step 5: Trigger Zenodo Archive

Once Zenodo integration is enabled, **creating a new GitHub Release will
automatically trigger Zenodo to archive the repository** and assign a DOI.

1. After publishing the release in Step 3, wait ~2-5 minutes
2. Check your email for a Zenodo confirmation
3. Go to https://zenodo.org/account/settings/github/
4. You should see a new deposit for `FIDES v1.0.0`
5. The DOI will be in the format `10.5281/zenodo.XXXXXXX`

---

## Step 6: Replace DOI Placeholder

Once you have the Zenodo DOI, you must replace the placeholder
`10.5281/zenodo.XXXXXXX` in the following files:

| File | Location | Replace |
|------|----------|---------|
| `README.md` | DOI badge (line 8) | `10.5281/zenodo.XXXXXXX` → your DOI |
| `README.md` | BibTeX block | `10.5281/zenodo.XXXXXXX` → your DOI |
| `README.md` | BSPC reference format | `10.5281/zenodo.XXXXXXX` → your DOI |
| `FIDES_BSPC_Submission.md` | Line 247 (Data Availability) | `10.5281/zenodo.XXXXXXX` → your DOI |
| `FIDES_BSPC_Submission.md` | Line 367 (Section S4) | `10.5281/zenodo.XXXXXXX` → your DOI |

**Automated replacement script**:

```python
# Run from the repository root after obtaining your DOI
import os

YOUR_DOI = "10.5281/zenodo.1234567"  # REPLACE with your actual DOI
PLACEHOLDER = "10.5281/zenodo.XXXXXXX"

files_to_update = [
    "README.md",
    "FIDES_BSPC_Submission.md",
]

for filepath in files_to_update:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace(PLACEHOLDER, YOUR_DOI)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Updated: {filepath}")

print(f"\nAll DOI placeholders replaced with {YOUR_DOI}")
```

After replacing, commit and push:

```bash
git add -A
git commit -m "Update DOI placeholder with Zenodo DOI"
git push origin main
```

---

## Step 7: Update the Release (Optional)

If you need to update the code after publishing the release (e.g., to fix the
DOI placeholder), create a new tag and release:

```bash
git tag -a v1.0.1 -m "Update DOI in README and paper"
git push origin v1.0.1
```

Then create a new release on GitHub for v1.0.1. Zenodo will create a new
versioned DOI that conceptually links to the original v1.0.0 DOI.

---

## Verification Checklist

After completing all steps, verify:

- [ ] GitHub Release v1.0.0 exists at https://github.com/tale-bonbom/FIDES/releases/tag/v1.0.0
- [ ] Zenodo deposit exists with a DOI of format `10.5281/zenodo.XXXXXXX`
- [ ] `README.md` has the DOI badge with the actual DOI
- [ ] `README.md` BibTeX block has the actual DOI in the `doi` field
- [ ] `FIDES_BSPC_Submission.md` Data Availability Statement contains the actual DOI
- [ ] `FIDES_BSPC_Submission.md` Supplementary Material S4 contains the actual DOI
- [ ] All checkpoint files are attached as release assets
- [ ] `CHANGELOG.md` is up to date

---

## BSPC Compliance Notes

This process satisfies the following BSPC Author Guide requirements:

- **Page 23-24 (Software Citation)**: Permanent DOI via Zenodo
- **Page 18 (Research Data Policy, Option C)**: Code deposited in a recognized
  data repository (Zenodo) with DOI
- **Page 18 (Data Availability Statement)**: Clear statement of code access
- **Page 16-17 (Supplementary Materials)**: Code linked in Supplementary Material S4

For questions about this process, consult the
[Zenodo-GitHub integration guide](https://help.zenodo.org/).
