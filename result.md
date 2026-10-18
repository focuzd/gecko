# Benchmark results


4 stories per prompt, 6 prompts, max 600 new characters, temperature 0.8.


## Automatic metrics

| source | real word rate | distinct-2 | ended on its own | longest copied span (words) |
|---|---|---|---|---|
| bigram | 0.36 ± 0.04 | 1.00 ± 0.01 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| mlp | 0.90 ± 0.03 | 0.97 ± 0.02 | 0.46 ± 0.51 | 1.29 ± 3.62 |
| gpt | 0.99 ± 0.01 | 0.91 ± 0.06 | 0.71 ± 0.46 | 8.00 ± 5.81 |
| reference | 1.00 ± 0.01 | 0.91 ± 0.05 | 0.33 ± 0.49 | 8.75 ± 4.92 |
| random | 0.08 ± 0.02 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |

## Judge: claude-haiku-5-5

Scores 1-5. The normalised score puts random characters at 0 and real TinyStories at 100.

| source | n | spelling | grammar | consistency | prompt_following | structure | meaning | style | weighted | normalised |
|---|---|---|---|---|---|---|---|---|---|---|
| bigram | 24 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.92 ± 0.28 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.08 ± 0.28 | 1.12 ± 0.05 | 0 |
| mlp | 24 | 1.58 ± 0.50 | 1.79 ± 0.41 | 1.62 ± 0.49 | 2.12 ± 0.34 | 1.75 ± 0.44 | 1.12 ± 0.34 | 2.00 ± 0.00 | 1.68 ± 0.22 | 18 |
| gpt | 24 | 4.04 ± 0.86 | 2.62 ± 0.71 | 2.08 ± 0.72 | 2.67 ± 0.76 | 2.04 ± 0.46 | 1.83 ± 0.48 | 3.29 ± 0.69 | 2.54 ± 0.43 | 45 |
| reference | 12 | 5.00 ± 0.00 | 4.42 ± 0.51 | 4.58 ± 0.51 | 4.75 ± 0.45 | 3.50 ± 0.52 | 3.75 ± 0.45 | 4.25 ± 0.45 | 4.25 ± 0.29 | 100 |
| random | 6 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 2.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.12 ± 0.00 | 0 |

### Normalised score by prompt type (claude-haiku-5-5)

| model | in-distribution | near | out-of-distribution |
|---|---|---|---|
| bigram | 0 | 0 | 0 |
| mlp | 14 | 18 | 21 |
| gpt | 52 | 38 | 46 |
