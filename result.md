# Benchmark results

4 stories per prompt, 6 prompts, max 600 new characters, temperature 0.8. All three models were trained on the first 5 million characters of TinyStories; the GPT is labelled gpt-5m so models trained on more data can be added next to it.

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

- **real word rate** (0-1): share of generated words that appear in the training data. Higher is better.
- **distinct-2** (0-1): unique word pairs / all word pairs. Low means looping and repetition, but noise also
  scores 1.0, so compare against the reference row (~0.9).
- **ended on its own** (0-1): share of stories where the model wrote the end-of-story marker within the length
  limit. For real stories it only says how many were shorter than the limit.
- **longest copied span** (words): longest run of words copied verbatim from the training data. Around the
  reference value means stock phrases ("Once upon a time, there was a"); much higher would suggest memorisation.

## Automatic metrics

| source | real word rate | distinct-2 | ended on its own | longest copied span (words) |
|---|---|---|---|---|
| bigram | 0.36 ± 0.04 | 1.00 ± 0.01 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| mlp | 0.90 ± 0.03 | 0.97 ± 0.02 | 0.46 ± 0.51 | 1.29 ± 3.62 |
| gpt-5m | 0.99 ± 0.01 | 0.91 ± 0.06 | 0.71 ± 0.46 | 8.00 ± 5.81 |
| reference | 1.00 ± 0.01 | 0.91 ± 0.05 | 0.33 ± 0.49 | 8.75 ± 4.92 |
| random | 0.08 ± 0.02 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |

## Judge: claude-haiku-5-5

Claude Haiku 5.5, run as Claude Code subagents: one fresh agent per story, same rubric as judge.py, never told which model wrote the story.

| source | n | spelling | grammar | consistency | prompt_following | structure | meaning | style | weighted | normalised |
|---|---|---|---|---|---|---|---|---|---|---|
| bigram | 24 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.92 ± 0.28 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.08 ± 0.28 | 1.12 ± 0.05 | 0 |
| mlp | 24 | 1.58 ± 0.50 | 1.79 ± 0.41 | 1.62 ± 0.49 | 2.12 ± 0.34 | 1.75 ± 0.44 | 1.12 ± 0.34 | 2.00 ± 0.00 | 1.68 ± 0.22 | 18 |
| gpt-5m | 24 | 4.04 ± 0.86 | 2.62 ± 0.71 | 2.08 ± 0.72 | 2.67 ± 0.76 | 2.04 ± 0.46 | 1.83 ± 0.48 | 3.29 ± 0.69 | 2.54 ± 0.43 | 45 |
| reference | 12 | 5.00 ± 0.00 | 4.42 ± 0.51 | 4.58 ± 0.51 | 4.75 ± 0.45 | 3.50 ± 0.52 | 3.75 ± 0.45 | 4.25 ± 0.45 | 4.25 ± 0.29 | 100 |
| random | 6 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 2.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.12 ± 0.00 | 0 |

### Normalised score by prompt type (claude-haiku-5-5)

How well each model holds up as the opening text gets less like the training data (0 = random characters, 100 = real TinyStories).

- **familiar (in-distribution)**: 'Once upon a time', 'One day, a little girl named Lily'
- **altered (near)**: 'Once upon a time, a robot named Zed', 'One day, a grumpy dragon wanted to'
- **unfamiliar (out-of-distribution)**: 'The scientist looked at the data and', 'In the year 3000, the city'

| model | overall | familiar (in-distribution) | altered (near) | unfamiliar (out-of-distribution) |
|---|---|---|---|---|
| bigram | 0 | 0 | 0 | 0 |
| mlp | 18 | 14 | 18 | 21 |
| gpt-5m | 45 | 52 | 38 | 46 |
