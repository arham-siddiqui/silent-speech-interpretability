# Phone-CTC Target Audit

- Prompt subset: **sentence**.
- Quality-controlled recordings: **178 / 200**.
- Speakers retained: **20**.
- Classes retained: **10**.
- Minimum recordings per retained class: **14**.
- Minimum recordings per retained speaker: **2**.
- Median phone-CTC probability: **0.451**.
- Median unconstrained phone error rate: **0.238**.

## Quality Gate

- Forced-path probability >= 0.05
- Phone error rate <= 0.75
- Phone emissions inside their word anchors >= 0.50
- Uniform fallback words <= 0.25 of the utterance

Intervals are derived from direct IPA phone-CTC emission centers and midpoint boundaries
inside independently aligned words. Interval methods represented: {'phone_ctc_midpoints_with_word_anchors': 2814, 'uniform_collapsed_word_fallback': 49}.
