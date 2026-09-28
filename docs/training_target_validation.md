# Training-target validation (TASK 5)

Automated verification of the segmentation training targets built from the development pool (train + internal_val). No models are run.

## Status counts

| target_status | pairs |
| --- | --- |
| VALID_POSITIVE | 643 |
| VALID_NEGATIVE | 1696 |
| IGNORE | 1 |
| **total** | 2340 |

## Checks

| Check | Result |
| --- | --- |
| Image/mask dimensional consistency | ✅ PASS |
| No patient leakage (train ↔ internal_val) | ✅ PASS |
| No test-set training targets | ✅ PASS |
| Correct target_status assignment | ✅ PASS |
| No positive-unlocalized converted to negative mask | ✅ PASS |
| No duplicate (image_id, pathology) pairs | ✅ PASS |

**Overall: PASS**

## Detail counts

- dimension mismatches: 0
- status assignment errors: 0
- IGNORE pairs with a mask file (must be 0): 0
- duplicate pairs: 0
- test rows (must be 0): 0
- train / internal_val patients: 161 / 39