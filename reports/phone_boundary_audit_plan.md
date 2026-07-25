# Phone Boundary Manual Audit Plan

The audit set contains **40** sentence recordings sampled deterministically
from the direct phone-CTC alignment. It covers all **10**
sentence classes (4 recordings each) and all **20**
speakers (2-2 recordings each).

## Sampling

- High-confidence alignments: **25**
- Primary-gate alignments: **8**
- Borderline alignments: **3**
- Alignments containing uniform word fallback: **4**

The tracked manifest is `metadata/phone_boundary_audit_set.csv`. Audio symlinks, editable TextGrids,
review JSON, and browser data remain under ignored `artifacts/phone_boundary_audit` because they contain
local paths or generated data.

## Workflow

1. Run `make phone-boundary-audit`.
2. Listen to each clip and drag any incorrect phone boundaries.
3. Mark every recording accepted, corrected, or excluded.
4. Run `make phone-boundary-import`; incomplete audits are rejected.

## Decision Rule

Listen to every clip and mark it accepted, corrected, or excluded. Corrected boundaries
must remain ordered, non-overlapping, and inside the recording. The automated phonetic
claims should only be rerun after all 40 rows have a manual decision. This audit tests
boundary validity; it does not turn canonical prompt pronunciations into observed
speaker-specific phonetic transcriptions.
