# Held-Out Sparse Feature Exemplars

The three highest-ranked fold-local SAE features were evaluated on speakers excluded
from both student and SAE training. Rankings come from training speakers; all metrics
and utterance exemplars below come from held-out test speakers.

## Held-Out Confirmation

| Fold | Feature | Train Class | Train Type | Test Frequency | Test Class Eta2 | Test Type Eta2 | Top-10 Class Match | Top-10 Type Match | Top-10 Speakers |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 434 | 6 | Sentence | 16.2% | 0.911 | 0.284 | 30.0% | 100.0% | 4 |
| 0 | 237 | 8 | Sentence | 23.9% | 0.855 | 0.292 | 40.0% | 100.0% | 4 |
| 0 | 285 | 24 | Sentence | 26.5% | 0.643 | 0.071 | 30.0% | 60.0% | 4 |
| 1 | 165 | 10 | Vowel | 37.8% | 0.657 | 0.368 | 20.0% | 60.0% | 3 |
| 1 | 73 | 28 | Word | 25.6% | 0.332 | 0.098 | 10.0% | 80.0% | 3 |
| 1 | 311 | 2 | Sentence | 21.1% | 0.799 | 0.225 | 30.0% | 90.0% | 3 |
| 2 | 502 | 0 | Sentence | 18.6% | 0.709 | 0.210 | 40.0% | 90.0% | 4 |
| 2 | 106 | 26 | Word | 25.5% | 0.557 | 0.158 | 20.0% | 100.0% | 3 |
| 2 | 299 | 6 | Sentence | 20.6% | 0.530 | 0.242 | 30.0% | 90.0% | 3 |
| 3 | 175 | 12 | Vowel | 18.3% | 0.742 | 0.213 | 20.0% | 50.0% | 4 |
| 3 | 236 | 28 | Sentence | 24.3% | 0.860 | 0.252 | 0.0% | 100.0% | 4 |
| 3 | 507 | 10 | Vowel | 12.2% | 0.548 | 0.117 | 10.0% | 40.0% | 4 |
| 4 | 90 | 13 | Vowel | 14.8% | 0.596 | 0.152 | 40.0% | 60.0% | 4 |
| 4 | 471 | 14 | Vowel | 21.7% | 0.710 | 0.095 | 20.0% | 50.0% | 4 |
| 4 | 336 | 12 | Vowel | 14.8% | 0.459 | 0.313 | 20.0% | 60.0% | 3 |

Across the 15 features, mean held-out class selectivity is
**0.660** and type selectivity is
**0.206**. Their top-10 activations match the
training-selected class **24.0%** of the time
and coarse type **75.3%** of the time. The top active utterances cover
**3.6 distinct held-out speakers** on average,
which argues against single-speaker exemplars driving the ranking.

## Strongest Held-Out Utterances

| Fold | Feature | Exemplar | Speaker | Utterance | Type | Activation |
|---:|---:|---:|---|---|---|---:|
| 0 | 434 | 1 | 15 | sentences6 | Sentence | 2.523 |
| 0 | 434 | 2 | 16 | sentences6 | Sentence | 2.318 |
| 0 | 434 | 3 | 8 | sentences6 | Sentence | 2.203 |
| 0 | 237 | 1 | 15 | sentences8 | Sentence | 2.889 |
| 0 | 237 | 2 | 16 | sentences8 | Sentence | 2.871 |
| 0 | 237 | 3 | 8 | sentences4 | Sentence | 2.689 |
| 0 | 285 | 1 | 10 | word4 | Word | 2.278 |
| 0 | 285 | 2 | 15 | sentences7 | Sentence | 2.260 |
| 0 | 285 | 3 | 16 | sentences10 | Sentence | 2.245 |
| 1 | 165 | 1 | 13 | vowel1 | Vowel | 2.124 |
| 1 | 165 | 2 | 20 | vowel1 | Vowel | 2.122 |
| 1 | 165 | 3 | 13 | word8 | Word | 2.112 |
| 1 | 73 | 1 | 20 | word1 | Word | 1.875 |
| 1 | 73 | 2 | 13 | word8 | Word | 1.750 |
| 1 | 73 | 3 | 13 | word14 | Word | 1.546 |
| 1 | 311 | 1 | 20 | sentences2 | Sentence | 2.411 |
| 1 | 311 | 2 | 7 | sentences5 | Sentence | 2.331 |
| 1 | 311 | 3 | 13 | sentences5 | Sentence | 2.135 |
| 2 | 502 | 1 | 1 | sentences8 | Sentence | 2.171 |
| 2 | 502 | 2 | 17 | sentences3 | Sentence | 2.151 |
| 2 | 502 | 3 | 6 | sentences1 | Sentence | 1.921 |
| 2 | 106 | 1 | 6 | word6 | Word | 2.395 |
| 2 | 106 | 2 | 6 | word15 | Word | 2.269 |
| 2 | 106 | 3 | 17 | word6 | Word | 1.766 |
| 2 | 299 | 1 | 6 | sentences6 | Sentence | 2.641 |
| 2 | 299 | 2 | 17 | sentences7 | Sentence | 2.622 |
| 2 | 299 | 3 | 1 | sentences6 | Sentence | 2.522 |
| 3 | 175 | 1 | 19 | vowel1 | Vowel | 2.652 |
| 3 | 175 | 2 | 3 | word3 | Word | 2.544 |
| 3 | 175 | 3 | 19 | vowel3 | Vowel | 2.435 |
| 3 | 236 | 1 | 3 | sentences4 | Sentence | 2.730 |
| 3 | 236 | 2 | 5 | sentences4 | Sentence | 2.593 |
| 3 | 236 | 3 | 12 | sentences3 | Sentence | 2.553 |
| 3 | 507 | 1 | 19 | vowel3 | Vowel | 2.254 |
| 3 | 507 | 2 | 19 | vowel1 | Vowel | 1.729 |
| 3 | 507 | 3 | 12 | word13 | Word | 1.550 |
| 4 | 90 | 1 | 2 | vowel4 | Vowel | 1.782 |
| 4 | 90 | 2 | 18 | vowel4 | Vowel | 1.757 |
| 4 | 90 | 3 | 2 | word2 | Word | 1.654 |
| 4 | 471 | 1 | 2 | word6 | Word | 2.831 |
| 4 | 471 | 2 | 14 | vowel4 | Vowel | 2.462 |
| 4 | 471 | 3 | 18 | word6 | Word | 2.394 |
| 4 | 336 | 1 | 14 | vowel3 | Vowel | 2.047 |
| 4 | 336 | 2 | 14 | vowel1 | Vowel | 1.519 |
| 4 | 336 | 3 | 2 | vowel3 | Vowel | 1.192 |

## Interpretation Boundary

Group names make the class and coarse utterance type inspectable, but they do not provide
phoneme timestamps. These examples validate repeatable utterance-level preferences; they
do not justify assigning a phoneme or articulator name to an individual feature.
