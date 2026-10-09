# Official-test results

The official test annotations had previously been inspected for dataset-level mask-overlap/schema analysis, but test performance was not used for model training, checkpoint selection, threshold selection, early stopping, or hyperparameter tuning. Model checkpoints and evaluation procedures were frozen before this final performance evaluation.

Scored population: 1534 VALID_POSITIVE official-test pairs from `results/test/test_eval_manifest.csv`.
Bootstrap: 5,000 resamples, numpy Generator seed 42, percentile 2.5/97.5, pairs in manifest order. The same index matrix is used for every overall Dice, IoU, and paired-improvement interval.
nnU-Net is unprompted. SAM 2 and MedSAM receive a ground-truth-derived box. These are different inference settings.
Nothing was retrained. Thresholds and preprocessing match the internal-validation procedure. Checkpoints were read only.

The first MedSAM pretrained launch raised `NameError: name 'torch' is not defined` before any test case was scored. The resumed MedSAM run then completed all 1,534 pairs for both MedSAM conditions. That name error did not change weights, thresholds, or the scored predictions.

| Model | Setting | Mean Dice | 95% CI | Mean IoU | Median Dice | vs internal val |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| nnU-Net v2 | Supervised / None / unprompted | 0.309919 | [0.293625, 0.326164] | 0.234052 | 0.161641 | -0.019131 |
| SAM 2.1 Hiera Tiny | Pretrained / Oracle GT box | 0.645995 | [0.635310, 0.656836] | 0.511932 | 0.702857 | -0.016724 |
| SAM 2.1 Hiera Tiny | Fine-tuned / Oracle GT box | 0.742136 | [0.731198, 0.752546] | 0.630198 | 0.825773 | -0.019766 |
| MedSAM ViT-B | Pretrained / Oracle GT box | 0.607481 | [0.596912, 0.617513] | 0.466588 | 0.644142 | -0.021275 |
| MedSAM ViT-B | Fine-tuned / Oracle GT box | 0.772297 | [0.762626, 0.781434] | 0.663990 | 0.849549 | -0.015498 |

## Fine-tuning gains

- **SAM 2.1 Hiera Tiny:** 0.096141 95% CI [0.090745, 0.101624] (includes zero: false)
- **MedSAM ViT-B:** 0.164816 95% CI [0.157842, 0.172124] (includes zero: false)

## Internal validation vs official test

Difference is official-test mean Dice minus the frozen internal-validation mean Dice.

- **nnU-Net v2 (Supervised):** 0.329050 → 0.309919 (-0.019131)
- **SAM 2.1 Hiera Tiny (Pretrained):** 0.662719 → 0.645995 (-0.016724)
- **SAM 2.1 Hiera Tiny (Fine-tuned):** 0.761902 → 0.742136 (-0.019766)
- **MedSAM ViT-B (Pretrained):** 0.628756 → 0.607481 (-0.021275)
- **MedSAM ViT-B (Fine-tuned):** 0.787795 → 0.772297 (-0.015498)
