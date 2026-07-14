# Official-weight ML preprocessing benchmark

## Conclusion

No tested generic pretrained ML enhancer improves over the raw-RGB control or
the lightweight CLAHE method on the locked test set. Zero-DCE and native-input
Restormer improve over the training-time preprocessing baseline, but that gain
is already explained by removing the baseline preprocessing: raw RGB performs
slightly better than both learned methods.

The recommended inference preprocessing remains `clahe_clip1_tile4`. Do not add
Real-ESRGAN, Restormer, or Zero-DCE globally to the production path for this
checkpoint.

## Leakage-safe protocol

- PARSeq checkpoint: epoch 26, refinement iterations 2.
- Validation: 397 images; all classical anchors and all six ML/input-order
  variants were evaluated here.
- Test: 411 images; only the two validation-selected ML variants plus the fixed
  anchors were evaluated.
- Ranking: exact match first, character accuracy second.
- Two preprocessing orders were tested: standardized 32x128 input and native
  crop enhancement followed by resize.
- Paired 2,000-sample bootstrap intervals compare each method with
  `train_baseline`.

## Validation results

| Method | Input order | Exact match | Character accuracy | Delta exact vs baseline | Delta char vs baseline |
| --- | --- | ---: | ---: | ---: | ---: |
| `clahe_clip1_tile4` | native then resize | 93.1990% | 98.7443% | +0.5038 pp | +0.0919 pp |
| `zero_dce` | resize then enhance | 92.9471% | 98.7443% | +0.2519 pp | +0.0919 pp |
| `restormer_motion_deblur_native` | enhance then resize | 92.9471% | 98.6524% | +0.2519 pp | +0.0000 pp |
| `raw_rgb` | resize only | 92.9471% | 98.5911% | +0.2519 pp | -0.0613 pp |
| `train_baseline` | native then resize | 92.6952% | 98.6524% | 0 | 0 |
| `zero_dce_native` | enhance then resize | 92.4433% | 98.7443% | -0.2519 pp | +0.0919 pp |
| `restormer_motion_deblur` | resize then enhance | 91.6877% | 98.5299% | -1.0076 pp | -0.1225 pp |
| `realesrgan_x2plus_native` | enhance then resize | 89.6725% | 97.5191% | -3.0227 pp | -1.1332 pp |
| `realesrgan_x2plus` | resize, enhance, resize | 89.4207% | 97.6110% | -3.2746 pp | -1.0413 pp |

## Locked test confirmation

| Method | Exact match | Character accuracy | Delta exact vs baseline | Delta char vs baseline | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| `clahe_clip1_tile4` | 93.1873% | 99.0768% | +1.2165 pp | +0.2085 pp | 243.1 img/s |
| `raw_rgb` | 93.1873% | 98.9875% | +1.2165 pp | +0.1191 pp | 307.3 img/s |
| `zero_dce` | 92.9440% | 98.9577% | +0.9732 pp | +0.0893 pp | 105.0 img/s |
| `restormer_motion_deblur_native` | 92.9440% | 98.9577% | +0.9732 pp | +0.0893 pp | 5.4 img/s |
| `train_baseline` | 91.9708% | 98.8684% | 0 | 0 | 206.1 img/s |

Against `raw_rgb`, each selected ML method is -0.2433 pp exact match and
-0.0298 pp character accuracy. Against the training baseline, the 95% paired
bootstrap intervals still cross zero:

- Zero-DCE exact: [-0.9732, +2.6764] pp; character: [-0.2085, +0.3604] pp.
- Restormer-native exact: [-0.7299, +2.6764] pp; character: [-0.2069, +0.3592] pp.

Real-ESRGAN is the only clear negative result: on validation, its exact-match
drop versus raw RGB is about 3.53 pp for standardized input and 3.27 pp for
native input, with both paired confidence intervals entirely below zero.

## Official model provenance

| Model | Official repository commit | Weight SHA-256 |
| --- | --- | --- |
| Real-ESRGAN x2plus | `a4abfb2979a7bbff3f69f58f58ae324608821e27` | `49FAFD45F8FD7AA8D31AB2A22D14D91B536C34494A5CFE31EB5D89C2FA266ABB` |
| Restormer motion deblur | `68dc6ac472db26f16361150cb7a96a1bc87da93f` | `194E38FB5B607C9DC5A5B3E08E65B2E79EE2BF0EF5048E0612F6B2FF2F79DA31` |
| Zero-DCE Epoch99 | `e0f4adc54d0f23348c4a9b84acc08fe8778d5bfd` | `A4395ACB874F320375D9704997CEF874EAAAAA26A1777CEB29A92B70F74C3612` |

The script verifies every hash before loading a model. TSRN/TextZoom was not
benchmarked because the authors' repository provides training/evaluation code
but no directly released pretrained checkpoint; unofficial mirrors were not
used.

## Recommended next ML experiment

The next defensible ML direction is not another global generic enhancer. Train
a plate-specific restoration model on the training split using synthetic blur,
noise, compression, glare, and low-light degradations, then select its checkpoint
only on validation. A second option is a quality router trained on training data
to apply specialist enhancement only to crops classified as degraded. Both must
retain the same locked-test protocol used here.

