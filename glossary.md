# Glossary

Plain-language meanings of the terms used in [report.md](report.md) and [explanation.md](explanation.md), in alphabetical order.

---

**AdamW**
The optimiser we used: the rule that decides how to nudge each parameter after every training step. It adapts the step size for each parameter separately and adds a small pull towards zero (weight decay).

**Altered prompts**
Benchmark openings that follow a familiar pattern but add new details, for example "Once upon a time, a robot named Zed". They test whether a model can handle a small step away from its training data.

**Attention (self-attention)**
The mechanism inside a transformer that lets each position in the text look back at earlier positions and decide which ones matter for its prediction. For example, when writing a pronoun, attention can look back at the character's name.

**Attention head**
One independent attention pattern. A layer with 6 heads can track 6 different kinds of relationships at once, such as "the previous few letters" and "a name from earlier".

**Batch size**
How many separate text sequences the model learns from in one training step. In our names it is the `b` part, for example `b64`.

**Benchmark**
Our fixed test: every model writes 24 stories from the same 6 openings with the same seeds, and each story is scored by a judge and by automatic checks.

**BPE (byte pair encoding)**
A way to split text into word pieces instead of single characters. Common words become one token and rare words are split into a few pieces. Not used yet; it is our top next step.

**Bigram model**
The simplest language model: it predicts the next character from only the current character, using a lookup table of letter pairs.

**Bits per character**
A way of measuring loss that does not depend on the tokenizer, so a character model and a word-piece model can be compared fairly.

**Block size**
See *context length*.

**Character-level model**
A model whose tokens are single characters (letters, digits, punctuation, space, newline). All our models are character-level.

**Characters per step**
Batch size times context length: how much text the model processes in one training step. For example, 64 × 512 = 32,768.

**Characters seen**
Characters per step times the number of steps: the total amount of text a model processed during training.

**Checkpoint**
A saved copy of a model's parameters (and sometimes the optimiser state) during or after training, so it can be reloaded later.

**Consistency (judge criterion)**
Whether characters, names and facts stay the same from start to end of a story.

**Context length (context window, block size)**
The maximum number of earlier characters a model can see when predicting the next one. In our names it is the `c` part, for example `c512`.

**Control stories**
Stories we mix into the benchmark to anchor the scale: real TinyStories stories (the top) and random characters (the bottom).

**Cross-attention**
Attention where the decoder looks at the encoder's output instead of at its own earlier positions. It is the one new piece an encoder-decoder adds to a GPT.

**Cross-entropy loss**
The standard loss for predicting the next token: how much probability the model gave to the character that actually came next. Lower means the model was less surprised.

**Decoder**
The part of a transformer that writes text one token at a time, looking only backwards. A GPT is a decoder on its own ("decoder-only").

**Distinct-2**
An automatic check for repetition: unique word pairs divided by all word pairs in a story. Low values mean the story loops. Random text scores 1.0, real stories about 0.9.

**Dropout**
A training trick that randomly switches off some connections during training so the model can't rely too heavily on any one of them. We used 0.2 (20%).

**Embedding**
A list of numbers (a vector) that represents a token. The model learns these, so similar characters end up with similar vectors.

**Encoder**
The part of a transformer that reads its whole input at once, in both directions, and builds a representation for the decoder to use.

**Encoder-decoder**
The original 2017 transformer layout, with an encoder that reads an input and a decoder that writes an output. Usually used for tasks like translation.

**Encoder fill (empty / partly filled / full)**
For our encoder-decoder: how much of the encoder's 256-character side contains real text. At the start of a story the encoder is empty.

**Ended on its own**
An automatic check: the share of stories where the model wrote the end-of-story marker before reaching the 600-character limit.

**Epoch / pass**
One full trip through the training data. Our best model made about 6.5 passes over its 50 million characters.

**Familiar prompts**
Benchmark openings typical of TinyStories, such as "Once upon a time".

**fp16 / fp32 (precision)**
How many bits each number uses. fp32 (32-bit) is the default and what we used. fp16 (16-bit) "mixed precision" is faster on GPUs; we plan to test it as its own experiment.

**Fused attention**
Computing attention in one combined GPU operation instead of several steps. Same maths, faster and lighter on memory. We checked that it gives the same outputs as the original code.

**GPT (decoder-only transformer)**
A transformer made only of decoder blocks, trained to predict the next token. Ours follows Karpathy's "Let's build GPT" lecture.

**Hallucination**
When a model produces fluent text that is wrong or doesn't make sense. Our models' name swaps and broken cause-and-effect are a small-scale version.

**Held-out data**
See *validation set*.

**Kaggle T4**
The free cloud GPU (an NVIDIA T4) we trained on, through Kaggle notebooks.

**Layer**
One transformer block, made of attention plus a feed-forward network. Our GPTs have 6.

**Layer normalisation**
A step that rescales the numbers inside the model to keep them in a stable range, which makes training smoother.

**Layer-split ablation**
Our planned encoder-decoder experiment: keep the total size about the same, but move layers between the encoder and decoder (2+4, 3+3, 4+2) to see which side matters more.

**Learning rate**
How big each training step's adjustment is. We used 0.0003 for the GPTs.

**LLM-as-a-judge**
Using a language model to score the output of other models, guided by a rubric.

**Longest copied span**
An automatic check: the longest run of words in a story that appears exactly in the model's training data. A rough test for memorisation.

**Loss**
A number that measures how wrong the model's predictions are. Training tries to make it smaller.

**Meaning (judge criterion)**
Whether a story makes sense as a whole, with clear cause and effect.

**Memorisation**
When a model reproduces its training text word for word instead of writing new text. A privacy and originality concern.

**MLP (multi-layer perceptron)**
A basic neural network of stacked layers. Our MLP language model joins the embeddings of the last 32 characters and passes them through one hidden layer (Bengio et al., 2003).

**Normalised score (0-100)**
Our main score. The judge's weighted average rescaled so random characters are 0 and real TinyStories stories are 100.

**One factor at a time**
Our experiment rule: change exactly one setting between two runs, so any difference has one clear cause.

**Overfitting**
When a model gets very good at its training data but worse on new text. A growing gap between training and validation loss is the warning sign.

**p-value**
The chance of seeing a difference at least this big if the two models were really equally good. Below 0.05 is usually treated as a real difference; we treated 0.05 to 0.10 as "a hint".

**Paired comparison (same-seed pairs)**
Comparing two models story by story, where both stories come from the same opening and the same random seed, and counting which model wins each pair.

**Parameters**
The numbers inside a model that training adjusts. More parameters means more capacity. In our names it is the size part, for example `19.3m`.

**Perplexity**
The exponential of the loss. Roughly "how many characters the model is choosing between" at each step. Random guessing over 84 characters gives 84.

**Position embedding**
A learned vector added to each token's embedding to tell the model where in the sequence the token is.

**Prefix language model**
A training setup (used by T5) where the first part of the text is read by an encoder and the rest is predicted by a decoder. Our small encoder-decoder works this way.

**Prompt**
The opening text a model is given to continue.

**Prompt following (judge criterion)**
Whether the story actually continues from the opening it was given.

**Real-word rate**
An automatic check: the share of generated words that appear in the model's training data.

**Residual connection**
A shortcut that adds a layer's input to its output, which makes deep models much easier to train.

**Rubric**
The written scoring guide given to the judge, describing what 1, 3 and 5 mean for each criterion.

**Run name**
Our naming scheme, `<model>-<parameters>-c<context>-d<data>-b<batch>`, for example `gpt-10.9m-c512-d50m-b64`.

**Sampling**
Picking the next character at random according to the model's probabilities, rather than always taking the most likely one.

**Seed (random seed)**
A starting number for the random number generator. Using the same seed makes runs and samples repeatable.

**Sequences per step**
See *batch size*. We learned that holding characters per step fixed while doubling the context halves the sequences per step, which hurt quality.

**Significance test (sign-flip test)**
The test we used on paired scores: randomly flip the sign of each pair's difference many times and see how often a gap as large as ours appears by chance.

**Spelling (judge criterion)**
Whether the words are real English words.

**Steps (training steps, iterations)**
How many times the model's parameters were updated during training.

**Structure (judge criterion)**
Whether the story has a setup, some events and an ending.

**Style (judge criterion)**
Whether the story reads like a simple TinyStories children's story.

**Temperature**
A sampling setting. Below 1 makes the model safer but more repetitive; above 1 makes it more random. We used 0.8.

**TinyStories**
A dataset of short, simple English stories (Eldan and Li, 2023), designed for training small language models.

**TinyStories-Instruct**
A version of TinyStories where each story comes with instructions, such as words it must use or a summary. A natural task for an encoder-decoder.

**Token / tokenizer**
A token is the unit of text the model works with. The tokenizer turns text into token numbers and back. For us, one token is one character.

**Top-k**
A sampling setting that only lets the model pick from its k most likely next characters.

**Transformer**
The neural network design behind GPT, built from attention and feed-forward layers (Vaswani et al., 2017).

**Unfamiliar prompts**
Benchmark openings unlike the training data, for example "The scientist looked at the data and".

**Validation loss**
The loss on text the model never trained on. It shows how well the model generalises.

**Validation set (held-out data)**
The last 10% of the text, kept aside and never used for training.

**Vocabulary**
The set of all tokens a model knows. Ours has 84 characters (5M data) or 96 characters (50M data).

**Weighted score**
The average of the 7 judge criteria, with consistency, structure and meaning counted 1.5 times.

**Width (embedding size, n_embd)**
How many numbers represent each position inside the model. 384 for our 10.9M GPTs, 512 for the 19.3M ones.
