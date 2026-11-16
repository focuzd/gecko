# Gecko: Building a Small Language Model from Scratch, and Measuring What Actually Makes It Better

**Course:** Trustworthy AI
**Project repository:** [github.com/focuzd/gecko](https://github.com/focuzd/gecko)
**Project website:** [focuzd.github.io/gecko](https://focuzd.github.io/gecko/)
**Period covered:** October to November 2026

Companion documents: [explanation.md](explanation.md) (full technical detail and every number), [glossary.md](glossary.md) (meaning of every term used here), [pitch.md](pitch.md) (short summary).

---

## Executive summary

We built a family of small language models that write children's stories, one character at a time. We started from the simplest possible model and worked our way up to a transformer, following Andrej Karpathy's *Neural Networks: Zero to Hero* course. All models were trained on TinyStories, a dataset of short, simple stories.

Building the models was only half the job. The other half was finding a fair way to say whether one model is *better* than another. Validation loss, the standard number, turned out to be misleading: more than once, a model with lower loss wrote worse stories. So we built a benchmark in which an AI judge, which never knows which model wrote a story, scores stories on seven criteria, placed on a scale from 0 (random letters) to 100 (real TinyStories stories).

Then we changed one thing at a time and measured each change.

**Main results**

- Our best model scores **53 out of 100**. The bigram baseline scores **0**, the MLP **18**, and our first GPT **45**.
- **More data did not help.** Training on 10 times more text left the score unchanged (45 vs 46).
- **A bigger batch helped the most.** Training the same model on 64 sequences per step instead of 32 raised the score from 40 to 51. This was our clearest result (p = 0.002).
- **Lower loss did not always mean better stories.** One model had a lower loss but wrote worse stories. We tracked down why.
- **Meaning and consistency are stuck at about 2 out of 5** for every GPT we trained. The models now spell and write grammatically, but their stories still don't hold together.
- **Longer training starts to cause copying.** Our best model reuses longer word-for-word stretches of its training text than real stories do.

---

## 1. Introduction

### 1.1 What we set out to do

A language model has one job: given some text, guess what comes next. Repeat that guess many times and you get a story. Big models such as GPT-4 do this with billions of parameters. We wanted to understand the idea from the ground up, by building small models ourselves and seeing what each improvement actually buys.

We had three goals:

1. **Build** a sequence of models, each a step up from the last: bigram, MLP, GPT, and an encoder-decoder.
2. **Measure** story quality in a way we could trust, not only with loss numbers.
3. **Find out what matters most**, by changing one factor at a time: data size, context length, batch size, model size and training length.

### 1.2 Why this is a Trustworthy AI project

Trustworthiness questions show up even at this small scale:

- **Fluent is not the same as correct.** Our models write stories that look fine at a glance but mix up who is who and lose track of what happened. This is a small-scale version of what larger models do when they "hallucinate".
- **Evaluation can mislead.** A single summary number (validation loss) gave the wrong answer more than once. Choosing how to evaluate a model is itself a trust decision.
- **Judges need checking.** We used an AI model as a judge, so we had to design the test so the judge could not play favourites, and keep its scores open to checking.
- **Memorisation.** We checked whether our models copy their training data word for word, and found that copying grows with longer training.

---

## 2. Data

We used **TinyStories** (Eldan and Li, 2023), a collection of short stories written in very simple English, the kind a 3 or 4 year old would understand. The stories are separated by a marker, `<|endoftext|>`.

| Setting | Source | Size used | Distinct characters |
|---|---|---|---|
| "d5m" models | TinyStories validation file | first 5 million characters | 84 |
| "d50m" models | TinyStories training file | first 50 million characters | 96 |

In both cases the first 90% of the text is used for training and the last 10% is held back for validation, so the model never trains on it.

Our models work with **characters**, not words. Every distinct character (letters, digits, punctuation, space, newline) is one token. This keeps things simple, but it means the model has to learn spelling before it can learn anything else.

---

## 3. The models

Each model takes the characters it can see and gives a probability for every possible next character. Generation samples one character from those probabilities, adds it to the text, and repeats.

| Model | What it looks at | Size | Idea |
|---|---|---|---|
| **Bigram** | the last 1 character | 7 thousand parameters | A lookup table: "after this letter, how likely is each letter next?" |
| **MLP** | the last 32 characters | 0.44 million | Turns each character into a small vector, joins them, and passes them through one hidden layer (Bengio et al., 2003). |
| **GPT** | the last 256 or 512 characters | 10.8 to 19.3 million | A decoder-only transformer. Self-attention lets each position look back at any earlier character. |
| **Encoder-decoder** | 256 older + 256 recent characters | 11.2 million | The original 2017 transformer layout: an encoder reads older text all at once, a decoder writes new text and reads the encoder. *Built and tested at small scale; not yet trained at full size.* |

The GPT follows Karpathy's "Let's build GPT" lecture: 6 layers, 6 attention heads (8 for the 19.3M model), learned position embeddings, layer normalisation, residual connections and dropout of 0.2.

### How we name models

Every model's name records how it was made, for example `gpt-10.9m-c512-d50m-b64`:

| Part | Meaning |
|---|---|
| `gpt` | model type |
| `10.9m` | 10.9 million parameters |
| `c512` | context of 512 characters |
| `d50m` | trained on 50 million characters of data |
| `b64` | 64 sequences per training step (batch size) |

---

## 4. How we measured quality

### 4.1 Validation loss

Validation loss measures how surprised the model is by text it has never seen. Lower is better. It is cheap and exact, but it averages over every position and every kind of text, so it can hide what matters for writing whole stories. Section 6 shows where it went wrong.

### 4.2 The story benchmark

Every model writes **24 stories**: 4 stories for each of 6 opening lines. The openings come in three groups, so we can see how each model copes with less familiar starts:

| Group | Openings |
|---|---|
| **Familiar** (like the training data) | "Once upon a time", "One day, a little girl named Lily" |
| **Altered** (familiar pattern, new details) | "Once upon a time, a robot named Zed", "One day, a grumpy dragon wanted to" |
| **Unfamiliar** (unlike the training data) | "The scientist looked at the data and", "In the year 3000, the city" |

Every model uses the same random seeds, the same temperature (0.8) and the same limit of 600 new characters, so the comparison is fair.

### 4.3 The judge

Each story is scored by **Claude Haiku 5.5**, a separate AI model. To keep the judging fair:

- **One story per judge.** Each story gets a fresh judge that sees only that one story, so it can't compare stories or drift over a long session.
- **No labels.** The judge never knows which model wrote the story.
- **A written rubric.** The judge scores 7 criteria from 1 to 5, each with a written description of what a 1, 3 and 5 mean, and writes a short justification first.

The seven criteria are **spelling, grammar, consistency, prompt following, structure, meaning** and **style**. Consistency, structure and meaning count 1.5 times as much in the average, because they are what separates a real story from fluent nonsense.

We first tried Google's Gemini Flash on its free tier, but it ran out of requests partway through, so we switched to Claude Haiku.

### 4.4 A scale from 0 to 100

Raw scores out of 5 are hard to read, so we also had the judge score two control groups, mixed in with the model stories:

- **12 real TinyStories stories**, cut to the same length. This sets **100**.
- **6 strings of random characters.** This sets **0**.

A score of 45 means a model closes 45% of the gap between random noise and real stories.

### 4.5 Automatic checks (no AI involved)

| Check | What it tells us |
|---|---|
| **Real-word rate** | share of generated words that appear in the training data |
| **Distinct-2** | how much the text repeats itself (low means looping) |
| **Ended on its own** | how often the model finished a story instead of hitting the length limit |
| **Longest copied span** | the longest run of words copied exactly from the training data; much higher than real stories would suggest memorisation |

### 4.6 Is a difference real?

With 24 stories per model, scores move by a few points just by chance. So for every important comparison we paired up stories written from the same opening with the same seed, counted how many pairs each model won, and ran a significance test. As a rule of thumb, gaps of about 5 points or less are not trustworthy on their own.

---

## 5. Results

### 5.1 From bigram to GPT

| Model | Validation loss | Score (0-100) | What its stories look like |
|---|---|---|---|
| Bigram | 2.29 | **0** | Common letter pairs, almost no real words: *"Once upon a timey wanthe er sackeaplithe we..."* |
| MLP | 1.13 | **18** | Real words and names, but sentences don't connect: *"...there was a big, red and wanted to lit home, in a big party."* |
| GPT (first version) | about 0.68 | **45** | Grammatical, readable stories that lose track of who is who: *"Blue said, 'Thank you, Blue!' Blue and Blue played..."* |

*Note: bigram, MLP and the first GPT were trained on the 5M-character set. Their losses are on that set's held-out text and are not directly comparable with the 50M models below.*

Each step up the course made a clear difference. The bigram is no better than random characters to the judge, even though about a third of its "words" are real. The MLP learns words. The GPT learns sentences.

### 5.2 Changing one thing at a time

All of these models are GPTs trained on the same 50 million characters.

| # | Model | What changed | Validation loss | Score | Familiar | Altered | Unfamiliar |
|---|---|---|---|---|---|---|---|
| A | `gpt-10.8m-c256-d50m-b64` | baseline | 0.661 | 46 | 49 | 39 | 49 |
| B | `gpt-10.9m-c512-d50m-b32` | context 256 → 512 | 0.638 | 40 | 49 | 33 | 37 |
| C | `gpt-19.3m-c512-d50m-b32` | size 10.9M → 19.3M (vs B) | 0.613 | 44 | 48 | 43 | 42 |
| D | `gpt-10.9m-c512-d50m-b64` | batch 32 → 64 (vs B) | 0.588 | 51 | 56 | 45 | 51 |
| E | `gpt-19.3m-c512-d50m-b64` | bigger, batch 64, 10k steps | **0.516** | **53** | **58** | **48** | **52** |

What we learned, factor by factor:

**1. Data size: no effect.** The first GPT trained on 5M characters scored 45; the same model trained on 50M scored 46. The model had already seen its 5M characters about 18 times with only a small gap between training and validation loss, so it was not short of data. Its limit was model size, training time and character-level tokens.

**2. Longer context: lower loss, worse stories.** Doubling the context to 512 characters lowered the loss but dropped the score from 46 to 40. We investigated and ruled out bugs, data differences and generation errors. The real cause: to keep the amount of text per step the same, this run used half as many sequences per step (32 instead of 64). The model got less practice at the start of sequences, which is exactly where most of a story is written. Its lower loss came only from positions beyond 256 characters, which the benchmark barely uses. Its stories also latched onto unusual words in the opening and mangled them ("scien", "scient", "scissient").

**3. Model size: helps, but only partly.** Nearly doubling the parameters at the same settings raised the score from 40 to 44 (won 15 of 24 pairs, p ≈ 0.08), and improved every criterion. It still didn't beat the 256-character baseline.

**4. Batch size: the biggest win.** Going back to 64 sequences per step at the 512 context raised the score from 40 to **51**, the first clearly significant result (p = 0.002). This confirmed the explanation in point 2: the longer context was never the problem, the smaller batch was. Note that this run also saw twice as many characters in total.

**5. Training longer: diminishing returns.** Our "how far can we get" run (19.3M parameters, batch 64, 10,000 steps, about 4 times the compute of model D) reached the lowest loss by far (0.516), but only 2 more points (53 vs 51, not a significant difference, p = 0.39). The extra training improved spelling and grammar, not meaning.

### 5.3 What the judge saw, criterion by criterion

| Model | Spelling | Grammar | Consistency | Prompt following | Structure | Meaning | Style |
|---|---|---|---|---|---|---|---|
| Bigram | 1.00 | 1.00 | 1.00 | 1.92 | 1.00 | 1.00 | 1.08 |
| MLP | 1.58 | 1.79 | 1.62 | 2.12 | 1.75 | 1.12 | 2.00 |
| First GPT (5M data) | 4.04 | 2.62 | 2.08 | 2.67 | 2.04 | 1.83 | 3.29 |
| Best GPT (E) | **4.25** | **2.96** | 2.08 | 3.12 | **2.38** | **2.04** | 3.46 |
| Real TinyStories | 5.00 | 4.42 | 4.58 | 4.75 | 3.50 | 3.75 | 4.25 |

Spelling is nearly solved. Consistency and meaning have barely moved from about 2 across every GPT we trained. This is the most important pattern in the whole project.

### 5.4 Copying from the training data

| Model | Longest copied span (words) |
|---|---|
| Real TinyStories stories | 8.75 |
| First GPT | 8.00 |
| `gpt-10.9m-c512-d50m-b64` (D) | 7.92 |
| `gpt-19.3m-c512-d50m-b32` (C) | 9.79 |
| `gpt-19.3m-c512-d50m-b64` (E) | **10.58** |

Real stories share stock phrases such as "Once upon a time, there was a", so some copying is normal. Our best model, after about 6.5 passes over the same 50 million characters, copies longer stretches than real stories do. It is not reciting whole stories, but the trend is worth watching.

### 5.5 The encoder-decoder (early results)

We built a small encoder-decoder in which the encoder reads the older half of the text and the decoder writes the newer half. A 1-million-parameter test version trained on a laptop CPU learned normally (validation loss 1.17) and wrote clean story openings. However, it barely used its encoder: giving the decoder 64 extra characters through the encoder improved its loss by only 0.005. The full 11-million-parameter versions, with three different splits of layers between encoder and decoder, are ready to train but have not been run yet.

---

## 6. Discussion

### 6.1 Loss is not the same as quality

Across models A, B and C, the loss went steadily down (0.661 → 0.638 → 0.613) while the story score went 46 → 40 → 44. Ranking by loss alone would have put these models in the wrong order. Loss is an average, and an average can hide *where* a model improved. In our case, the improvement was in a part of the text the stories rarely reach. The lesson is simple: **test a model the way it will actually be used.**

### 6.2 "The same amount of training" can mean two things

When we doubled the context, we had to choose what to keep equal: characters per step, or sequences per step. We can't keep both equal. We kept characters equal, which halved the sequences, and that mistake explained the whole drop. Fair comparisons need a clear statement of what is held fixed.

### 6.3 Why meaning is stuck

Every improvement we made helped spelling, grammar and structure. None of them moved meaning or consistency much. Our best guess is the character-level tokens: the model spends most of its capacity learning how to spell, and a 512-character window holds only about 100 words. Word-piece tokens (BPE) would let the model work with whole words and see more of the story at once. This is our top next step.

### 6.4 Can we trust the judge?

We took several steps to keep the judge honest: one story per judge, no model names, a written rubric, written justifications, and real and random stories as anchors. The anchors behaved as expected: real stories scored highest, random characters lowest, and each model sat in a sensible place between them. What we have *not* yet done is re-judge old stories later to check that the judge's standards didn't drift between runs, or compare against a second judge model.

---

## 7. Limitations

- **One training run per setting.** We never trained the same setup twice with a different random seed, so we don't know exactly how much scores vary between runs.
- **Only 24 stories per model.** Small differences (about 5 points) can be chance.
- **One judge model.** Another judge, or human readers, might rank the models differently.
- **Not every comparison is one factor.** Model D also saw twice as much text as model A, and model E changed size and training length together.
- **Different data splits.** The 5M models used TinyStories' validation file and the 50M models its training file, so the data-size comparison is not perfectly clean.
- **Character-level only.** All results are for character tokens. They may not carry over to word-piece models.
- **The encoder-decoder is untested at full size.**

---

## 8. Next steps

In order of priority:

1. **Word-piece tokens (BPE)**, compared against model D at the same settings. This is the change most likely to improve meaning and consistency.
2. **Sampling settings** (temperature, top-k, repetition penalty) on our best model. This needs no training and could fix some of the looping.
3. **Half-precision (fp16) training**, as its own experiment, to make future runs about twice as fast.
4. **Full encoder-decoder runs** for the three layer splits, and later a version trained on TinyStories-Instruct, where the encoder reads instructions and the decoder writes the story.
5. **Checks on our own method:** a second training seed for one setup, and a re-judging of old stories to test judge drift.
6. **More data instead of more passes**, if we train longer, to avoid memorisation.

---

## 9. Conclusion

We built four kinds of language model from scratch and a benchmark we can trust more than the loss curve. Going from bigram to GPT took the score from 0 to 45, and careful, one-factor-at-a-time experiments took it to 53. The biggest gain came from something as ordinary as batch size, and the most useful lesson came from a result that first looked like a failure: lower loss, worse stories.

Our models now write stories that look right: real words, correct spelling, mostly sound grammar. They still don't fully make sense. Closing that gap between looking right and being right is the same challenge, at a much smaller scale, that makes trust in large language models hard.

---

## References

1. Karpathy, A. *Neural Networks: Zero to Hero* (video lecture series), including "Let's build GPT: from scratch, in code, spelled out."
2. Eldan, R. and Li, Y. (2023). *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* arXiv:2305.07759.
3. Bengio, Y., Ducharme, R., Vincent, P. and Jauvin, C. (2003). *A Neural Probabilistic Language Model.* Journal of Machine Learning Research.
4. Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.
5. Raffel, C. et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5). JMLR.
6. Zheng, L. et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS.
7. Hoffmann, J. et al. (2022). *Training Compute-Optimal Large Language Models* (Chinchilla). NeurIPS.

---

## Appendix: training setup of every model

| Model | Parameters | Context | Batch | Characters per step | Steps | Characters seen | Data |
|---|---|---|---|---|---|---|---|
| `bigram-7k-c1-d5m-b32` | 0.01M | 1 (trained on 8) | 32 | 256 | 3,000 | 1M | 5M |
| `mlp-439k-c32-d5m-b64` | 0.44M | 32 | 64 | 2,048 | 10,000 | 20M | 5M |
| `gpt-10.8m-c256-d5m-b64` | 10.80M | 256 | 64 | 16,384 | 5,000 | 82M | 5M |
| `gpt-10.8m-c256-d50m-b64` | 10.81M | 256 | 64 | 16,384 | 5,000 | 82M | 50M |
| `gpt-10.9m-c512-d50m-b32` | 10.91M | 512 | 32 | 16,384 | 5,000 | 82M | 50M |
| `gpt-19.3m-c512-d50m-b32` | 19.27M | 512 | 32 | 16,384 | 5,000 | 82M | 50M |
| `gpt-10.9m-c512-d50m-b64` | 10.91M | 512 | 64 | 32,768 | 5,000 | 164M | 50M |
| `gpt-19.3m-c512-d50m-b64` | 19.27M | 512 | 64 | 32,768 | 10,000 | 328M | 50M |

All GPTs were trained in full 32-bit precision with the AdamW optimiser at a learning rate of 0.0003, on a free Kaggle T4 GPU. Full results, stories and judge scores are in [`benchmarks/`](benchmarks/).
