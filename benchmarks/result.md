# Benchmark results

4 stories per prompt, 6 prompts, max 600 new characters, temperature 0.8.

Model names are `<model>-<parameters>-c<context length>-d<training characters>-b<batch size>`, e.g. `gpt-10.8m-c256-d50m-b64` is a 10.8M parameter GPT that sees 256 characters at a time, trained on 50 million characters with 64 sequences per step. d5m models used the TinyStories validation split, d50m models the train split.

## How to read this

**Judge scores** are 1 (worst) to 5 (best) per criterion, ± is the standard deviation over the stories.

- **spelling**: are the words real English words.
- **grammar**: are the sentences well formed and fluent.
- **consistency**: do characters, names and facts stay the same throughout.
- **prompt_following**: does the story actually continue the opening text it was given.
- **structure**: is there a setup, some events and an ending.
- **meaning**: does the story make sense as a whole, with cause and effect.
- **style**: does it read like a simple TinyStories children's story.
- **weighted** (1-5): mean of the seven, with consistency, structure and meaning counted 1.5x.
- **normalised** (0-100): the weighted score rescaled so random characters are 0 and real TinyStories stories
  are 100. 45 means the model closes 45% of the gap between noise and real stories.

**Automatic metrics** are computed in Python, no LLM involved.

- **real word rate** (0-1): share of generated words that appear in that model's training data. Higher is better.
- **distinct-2** (0-1): unique word pairs / all word pairs. Low means looping and repetition, but noise also
  scores 1.0, so compare against the reference row (~0.9).
- **ended on its own** (0-1): share of stories where the model wrote the end-of-story marker within the length
  limit. For real stories it only says how many were shorter than the limit.
- **longest copied span** (words): longest run of words copied verbatim from that model's training data. Around the
  reference value means stock phrases ("Once upon a time, there was a"); much higher would suggest memorisation.

## Training setup

| model | parameters | context | batch size | chars per step | steps | chars seen | dataset size |
|---|---|---|---|---|---|---|---|
| bigram-7k-c1-d5m-b32 | 0.01M | 8 | 32 | 256 | 3000 | 1M | 5M |
| mlp-439k-c32-d5m-b64 | 0.44M | 32 | 64 | 2,048 | 10000 | 20M | 5M |
| gpt-10.8m-c256-d5m-b64 | 10.80M | 256 | 64 | 16,384 | 5000 | 82M | 5M |
| gpt-10.8m-c256-d50m-b64 | 10.81M | 256 | 64 | 16,384 | 5000 | 82M | 50M |
| gpt-10.9m-c512-d50m-b32 | 10.91M | 512 | 32 | 16,384 | 5000 | 82M | 50M |
| gpt-19.3m-c512-d50m-b32 | 19.27M | 512 | 32 | 16,384 | 5000 | 82M | 50M |
| gpt-10.9m-c512-d50m-b64 | 10.91M | 512 | 64 | 32,768 | 5000 | 164M | 50M |

## Automatic metrics

| source | real word rate | distinct-2 | ended on its own | longest copied span (words) |
|---|---|---|---|---|
| bigram-7k-c1-d5m-b32 | 0.36 ± 0.04 | 1.00 ± 0.01 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| mlp-439k-c32-d5m-b64 | 0.90 ± 0.03 | 0.97 ± 0.02 | 0.46 ± 0.51 | 1.29 ± 3.62 |
| gpt-10.8m-c256-d5m-b64 | 0.99 ± 0.01 | 0.91 ± 0.06 | 0.71 ± 0.46 | 8.00 ± 5.81 |
| gpt-10.8m-c256-d50m-b64 | 0.99 ± 0.01 | 0.91 ± 0.06 | 0.71 ± 0.46 | 7.79 ± 4.64 |
| gpt-10.9m-c512-d50m-b32 | 0.98 ± 0.03 | 0.83 ± 0.10 | 0.58 ± 0.50 | 7.29 ± 4.56 |
| gpt-19.3m-c512-d50m-b32 | 0.99 ± 0.02 | 0.86 ± 0.07 | 0.54 ± 0.51 | 9.79 ± 3.61 |
| gpt-10.9m-c512-d50m-b64 | 0.98 ± 0.04 | 0.87 ± 0.06 | 0.75 ± 0.44 | 7.92 ± 5.85 |
| reference | 1.00 ± 0.01 | 0.91 ± 0.05 | 0.33 ± 0.49 | 8.75 ± 4.92 |
| random | 0.08 ± 0.02 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |

## Judge: claude-haiku-5-5

Claude Haiku 5.5, run as Claude Code subagents: one fresh agent per story, same rubric as judge.py, never told which model wrote the story.

| source | n | spelling | grammar | consistency | prompt_following | structure | meaning | style | weighted | normalised |
|---|---|---|---|---|---|---|---|---|---|---|
| bigram-7k-c1-d5m-b32 | 24 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.92 ± 0.28 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.08 ± 0.28 | 1.12 ± 0.05 | 0 |
| mlp-439k-c32-d5m-b64 | 24 | 1.58 ± 0.50 | 1.79 ± 0.41 | 1.62 ± 0.49 | 2.12 ± 0.34 | 1.75 ± 0.44 | 1.12 ± 0.34 | 2.00 ± 0.00 | 1.68 ± 0.22 | 18 |
| gpt-10.8m-c256-d5m-b64 | 24 | 4.04 ± 0.86 | 2.62 ± 0.71 | 2.08 ± 0.72 | 2.67 ± 0.76 | 2.04 ± 0.46 | 1.83 ± 0.48 | 3.29 ± 0.69 | 2.54 ± 0.43 | 45 |
| gpt-10.8m-c256-d50m-b64 | 24 | 4.04 ± 1.00 | 2.67 ± 0.56 | 1.96 ± 0.62 | 2.75 ± 0.85 | 2.12 ± 0.45 | 1.88 ± 0.34 | 3.29 ± 0.55 | 2.55 ± 0.36 | 46 |
| gpt-10.9m-c512-d50m-b32 | 24 | 3.50 ± 1.22 | 2.42 ± 0.72 | 1.71 ± 0.46 | 3.00 ± 0.78 | 1.96 ± 0.36 | 1.79 ± 0.51 | 2.96 ± 0.46 | 2.36 ± 0.41 | 40 |
| gpt-19.3m-c512-d50m-b32 | 24 | 3.79 ± 0.93 | 2.50 ± 0.51 | 1.92 ± 0.41 | 3.04 ± 0.75 | 2.12 ± 0.34 | 1.92 ± 0.28 | 3.04 ± 0.46 | 2.51 ± 0.26 | 44 |
| gpt-10.9m-c512-d50m-b64 | 24 | 4.04 ± 1.00 | 2.88 ± 0.61 | 2.17 ± 0.56 | 3.12 ± 0.74 | 2.17 ± 0.38 | 2.00 ± 0.00 | 3.50 ± 0.51 | 2.71 ± 0.34 | 51 |
| reference | 12 | 5.00 ± 0.00 | 4.42 ± 0.51 | 4.58 ± 0.51 | 4.75 ± 0.45 | 3.50 ± 0.52 | 3.75 ± 0.45 | 4.25 ± 0.45 | 4.25 ± 0.29 | 100 |
| random | 6 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 2.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.12 ± 0.00 | 0 |

### Normalised score by prompt type (claude-haiku-5-5)

How well each model holds up as the opening text gets less like the training data (0 = random characters, 100 = real TinyStories).

- **familiar (in-distribution)**: 'Once upon a time', 'One day, a little girl named Lily'
- **altered (near)**: 'Once upon a time, a robot named Zed', 'One day, a grumpy dragon wanted to'
- **unfamiliar (out-of-distribution)**: 'The scientist looked at the data and', 'In the year 3000, the city'

| model | overall | familiar (in-distribution) | altered (near) | unfamiliar (out-of-distribution) |
|---|---|---|---|---|
| bigram-7k-c1-d5m-b32 | 0 | 0 | 0 | 0 |
| mlp-439k-c32-d5m-b64 | 18 | 14 | 18 | 21 |
| gpt-10.8m-c256-d5m-b64 | 45 | 52 | 38 | 46 |
| gpt-10.8m-c256-d50m-b64 | 46 | 49 | 39 | 49 |
| gpt-10.9m-c512-d50m-b32 | 40 | 49 | 33 | 37 |
| gpt-19.3m-c512-d50m-b32 | 44 | 48 | 43 | 42 |
| gpt-10.9m-c512-d50m-b64 | 51 | 56 | 45 | 51 |
