# Gecko: The Pitch

**We built a story-writing AI from scratch, then built a fair way to tell whether it was actually getting better.**

---

## The problem

Everyone wants better language models, but "better" is slippery. The usual yardstick, validation loss, is a single average number. In our project, models with a **lower** loss wrote **worse** stories more than once. If you can't measure progress honestly, you can't trust the model, or the claims made about it.

## What we did

1. **Built four generations of language model from the ground up**, following Andrej Karpathy's *Zero to Hero* course: a bigram lookup table, an MLP, a GPT-style transformer and an encoder-decoder. All of them write children's stories one character at a time, trained on the TinyStories dataset.
2. **Built a blind benchmark.** Each model writes 24 stories from familiar, altered and unfamiliar openings. An AI judge scores every story on seven criteria without knowing which model wrote it. Real stories set 100 and random letters set 0.
3. **Ran careful experiments**, changing one factor at a time (data, context, size, batch, training length), and tested whether each difference was real or just luck.

## What we found

| | Score (0-100) |
|---|---|
| Bigram | 0 |
| MLP | 18 |
| First GPT | 45 |
| **Best GPT** | **53** |

- **10× more data: no gain.** 45 → 46.
- **A bigger batch was the biggest win**, 40 → 51, our clearest result (p = 0.002).
- **Lower loss, worse stories.** We traced a surprising drop to a subtle training choice, not to the idea being tested.
- **Fluent is not the same as sensible.** Spelling reached 4.25 out of 5, but meaning stayed stuck near 2.
- **Train too long and it starts copying.** Our best model reuses longer stretches of training text than real stories do.

## Why it matters

These are the same trust questions big AI systems face, at a scale we can fully inspect:

- **Evaluation:** a single number can point the wrong way, so test models the way they will actually be used.
- **Hallucination:** text can look right without making sense.
- **Fair judging:** AI judges need blinding, rubrics and anchors.
- **Memorisation:** copying grows quietly with more training.

## What's next

- **Word-piece tokens (BPE)**, our best bet for moving meaning past 2 out of 5.
- **The encoder-decoder** at full size: it's built and tested small, and ready to train.
- **Checks on our own method:** a second training seed and a re-judging pass to measure noise and judge drift.

## In one line

**Building the model taught us how language models work. Measuring it honestly taught us why trusting them is hard.**

---

Code and data: [github.com/focuzd/gecko](https://github.com/focuzd/gecko) · Milestones: [focuzd.github.io/gecko](https://focuzd.github.io/gecko/) · Full write-up: [report.md](report.md)
