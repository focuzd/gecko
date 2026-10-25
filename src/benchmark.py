import argparse
import json
import random
import re
import statistics
from collections import Counter, defaultdict

import torch

from data import DATASETS, CharTokenizer, load_text
from judge import CRITERIA, Judge, QuotaExhausted
from sample_stories import END_MARKER, generate_story
from utils import ROOT, WEIGHTS_DIR, get_device, load_checkpoint, resolve_path

BENCH_DIR = ROOT / "benchmarks"
RESULT_FILE = BENCH_DIR / "result.md"
STORY_DIR = BENCH_DIR / "stories"
STORIES_FILE = BENCH_DIR / "stories.jsonl"

# run names as written by train.py (<model>-<params>-c<context>-d<training chars>), each in weights/<name>.pt
MODELS = {name: f"{name}.pt" for name in [
    "bigram-7k-c1-d5m",
    "mlp-439k-c32-d5m",
    "gpt-10.8m-c256-d5m",
    "gpt-10.8m-c256-d50m",
    "gpt-10.9m-c512-d50m",
    "gpt-19.3m-c512-d50m",
]}

GROUP_NAMES = {
    "in": "familiar (in-distribution)",
    "near": "altered (near)",
    "out": "unfamiliar (out-of-distribution)",
}

JUDGE_NOTES = {
    "claude-haiku-5-5": "Claude Haiku 5.5, run as Claude Code subagents: one fresh agent per story, "
                        "same rubric as judge.py, never told which model wrote the story.",
}

HOW_TO_READ = """## How to read this

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
"""

PROMPTS = {
    "in": ["Once upon a time", "One day, a little girl named Lily"],
    "near": ["Once upon a time, a robot named Zed", "One day, a grumpy dragon wanted to"],
    "out": ["The scientist looked at the data and", "In the year 3000, the city"],
}

STORIES_PER_PROMPT = 4
MAX_NEW_CHARS = 600
TEMPERATURE = 0.8
N_REFERENCE = 12
N_RANDOM = 6

# meaning, consistency and structure are what separate a story from fluent-looking text
WEIGHTS = {name: 1.0 for name in CRITERIA}
WEIGHTS.update(meaning=1.5, consistency=1.5, structure=1.5)


def all_prompts():
    return [(group, prompt) for group, prompts in PROMPTS.items() for prompt in prompts]


def train_split_text():
    text = load_text("tinystories")
    n = int(0.9 * len(text))
    return text[:n], text[n:]


def model_stories(name, weights_path, device):
    model, ckpt = load_checkpoint(weights_path, device)
    tok = CharTokenizer(ckpt["chars"])
    # stored with every story so the report knows the training data without needing the weights
    data = dict(dataset=ckpt["dataset"], max_chars=ckpt.get("max_chars") or DATASETS[ckpt["dataset"]]["max_chars"])
    stories = []
    for p, (group, prompt) in enumerate(all_prompts()):
        dropped = set(prompt) - set(tok.chars)
        if dropped:
            print(f"warning: {prompt!r} has characters the model doesn't know: {dropped}")
        for k in range(STORIES_PER_PROMPT):
            torch.manual_seed(1000 * p + k)  # same seeds for every model
            text, ended = generate_story(model, tok, ckpt["block_size"], prompt, MAX_NEW_CHARS, TEMPERATURE)
            stories.append(dict(id=f"{name}-{p}-{k}", source=name, group=group,
                                prompt=prompt, text=text, ended=ended, **data))
        print(f"{name}: done with {prompt!r}")
    return stories


def generate(weights=None, out=None):
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()

    if weights:
        # a single model, e.g. on Kaggle right after training; bring the file back and run `import`
        path = resolve_path(weights)
        out = resolve_path(out) if out else BENCH_DIR / f"stories_{path.stem}.jsonl"
        write_jsonl(out, model_stories(path.stem, path, device))
        print(f"saved to {out}, run `python src/benchmark.py import {out}` where the benchmark lives")
        return

    stories = load_jsonl(STORIES_FILE) if STORIES_FILE.exists() else []
    have = {s["source"] for s in stories}
    for name, weights_file in MODELS.items():
        if name not in have:
            stories += model_stories(name, WEIGHTS_DIR / weights_file, device)

    if "reference" in have:
        save_stories(stories)
        return

    # real TinyStories stories from the held out split, cut to the same length
    _, val_text = train_split_text()
    real = [s.strip() for s in val_text.split(END_MARKER) if len(s.strip()) > 200]
    rng = random.Random(0)
    for i, story in enumerate(rng.sample(real[1:-1], N_REFERENCE)):  # first/last may be partial
        prompt = " ".join(story.split()[:4])
        stories.append(dict(id=f"reference-{i}", source="reference", group="reference",
                            prompt=prompt, text=story[:len(prompt) + MAX_NEW_CHARS],
                            ended=len(story) <= len(prompt) + MAX_NEW_CHARS))

    # random characters as the floor
    chars = sorted(set(val_text) - {"\n"})
    for i in range(N_RANDOM):
        group, prompt = all_prompts()[i % len(all_prompts())]
        noise = "".join(rng.choice(chars) for _ in range(MAX_NEW_CHARS))
        stories.append(dict(id=f"random-{i}", source="random", group="random",
                            prompt=prompt, text=prompt + noise, ended=False))

    save_stories(stories)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def save_stories(stories):
    write_jsonl(STORIES_FILE, stories)
    print(f"saved {len(stories)} stories to {STORIES_FILE}")


def import_stories(path):
    stories = load_jsonl(STORIES_FILE)
    have = {s["id"] for s in stories}
    new = [s for s in load_jsonl(resolve_path(path)) if s["id"] not in have]
    print(f"importing {len(new)} stories from {sorted({s['source'] for s in new})}")
    save_stories(stories + new)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def judgments_file(judge_model):
    return BENCH_DIR / f"judgments_{judge_model}.jsonl"


def judge(judge_model, provider, rpm):
    stories = load_jsonl(STORIES_FILE)
    out = judgments_file(judge_model)
    done = {j["id"] for j in load_jsonl(out)} if out.exists() else set()

    # shuffled so the judge never sees one model's stories in a row
    todo = [s for s in stories if s["id"] not in done]
    random.Random(1).shuffle(todo)
    print(f"{len(done)} already judged, {len(todo)} to go")

    j = Judge(judge_model, provider, rpm)
    with open(out, "a", encoding="utf-8") as f:
        for i, s in enumerate(todo, 1):
            try:
                result = j.score(s["prompt"], s["text"])
            except QuotaExhausted as e:
                print(f"daily quota used up, stopping. rerun later to continue.\n{e}")
                return
            except RuntimeError as e:
                print(f"[{i}/{len(todo)}] {s['id']}: skipped ({e}), rerun to retry")
                continue
            f.write(json.dumps({"id": s["id"], **result}) + "\n")
            f.flush()
            print(f"[{i}/{len(todo)}] {s['id']}: " + " ".join(str(result[c]) for c in CRITERIA))


def weighted_score(j):
    return sum(WEIGHTS[c] * j[c] for c in CRITERIA) / sum(WEIGHTS.values())


def words(text):
    return re.findall(r"[a-z']+", text.lower())


class TrainingData:
    """Words of the text a model was trained on, for the real-word and copying checks."""

    def __init__(self, text):
        w = words(text)
        self.known_words = {x for x, c in Counter(w).items() if c >= 3}
        # one big string instead of a set of n-grams, which would not fit in memory for 50M chars
        self.joined = " " + " ".join(w) + " "

    def contains(self, seq):
        return " " + " ".join(seq) + " " in self.joined


def data_key(story):
    if "dataset" in story:
        return story["dataset"], story["max_chars"]
    # stories from before the dataset was stored with them: look it up in the weights
    ckpt = torch.load(WEIGHTS_DIR / MODELS[story["source"]], map_location="cpu", weights_only=True)
    return ckpt["dataset"], ckpt.get("max_chars") or DATASETS[ckpt["dataset"]]["max_chars"]


def training_data_for(key, cache={}):
    if key not in cache:
        text = load_text(*key)
        cache[key] = TrainingData(text[:int(0.9 * len(text))])
    return cache[key]


def auto_metrics(story, train, n=8):
    w = words(story["text"][len(story["prompt"]):])
    bigrams = list(zip(w, w[1:]))
    known_words = train.known_words

    # longest run of consecutive n-grams that appear verbatim in the training data
    longest = run = 0
    for i in range(len(w) - n + 1):
        run = run + 1 if train.contains(w[i:i + n]) else 0
        longest = max(longest, run)

    return {
        "real_words": sum(x in known_words for x in w) / max(len(w), 1),
        "distinct_2": len(set(bigrams)) / max(len(bigrams), 1),
        "ended": float(story["ended"]),
        "longest_copy": longest + n - 1 if longest else 0,
    }


def mean_std(xs):
    if not xs:
        return "-"
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return f"{statistics.mean(xs):.2f} ± {sd:.2f}"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    return "\n".join(lines)


def model_sources(stories):
    # every model in the stories file, in the order it was added (includes imported ones)
    return [s for s in dict.fromkeys(st["source"] for st in stories.values()) if s not in ("reference", "random")]


def write_story_files(stories):
    # one readable file per model with the exact stories that were judged
    STORY_DIR.mkdir(parents=True, exist_ok=True)
    for src in model_sources(stories):
        lines = [f"{src}: {STORIES_PER_PROMPT} stories per prompt, max {MAX_NEW_CHARS} new characters, "
                 f"temperature {TEMPERATURE}", ""]
        for p, (group, prompt) in enumerate(all_prompts()):
            lines += ["=" * 80, f"{GROUP_NAMES[group]}", f"prompt: {prompt!r}", "=" * 80, ""]
            for k in range(STORIES_PER_PROMPT):
                s = stories[f"{src}-{p}-{k}"]
                ending = "ended on its own" if s["ended"] else f"stopped at {MAX_NEW_CHARS} characters"
                lines += [f"--- story {k + 1} ({ending}) ---", s["text"].strip(), ""]
        (STORY_DIR / f"{src}.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"saved story files to {STORY_DIR}")


def report():
    stories = {s["id"]: s for s in load_jsonl(STORIES_FILE)}
    write_story_files(stories)
    models = model_sources(stories)
    sources = models + ["reference", "random"]

    # each model is checked against its own training data, the controls against the 5m data
    first = {s["source"]: s for s in reversed(list(stories.values()))}
    train = {src: training_data_for(data_key(first[src])) for src in models}
    train["reference"] = train["random"] = training_data_for(("tinystories", 5_000_000))
    metrics = {sid: auto_metrics(s, train[s["source"]]) for sid, s in stories.items()}

    sections = ["# Benchmark results",
                f"{STORIES_PER_PROMPT} stories per prompt, {len(all_prompts())} prompts, "
                f"max {MAX_NEW_CHARS} new characters, temperature {TEMPERATURE}.",
                "Model names are `<model>-<parameters>-c<context length>-d<training characters>`, e.g. "
                "`gpt-10.8m-c256-d50m` is a 10.8M parameter GPT that sees 256 characters at a time, trained "
                "on 50 million characters. d5m models used the TinyStories validation split, d50m models the "
                "train split. All GPTs were trained for 5000 steps with 16,384 characters per step (64 x 256 or "
                "32 x 512), except names ending in `-b<N>`, which used batch N (e.g. -b64 at c512 = 32,768 per step).",
                HOW_TO_READ.strip()]

    by_source = defaultdict(list)
    for sid, m in metrics.items():
        by_source[stories[sid]["source"]].append(m)
    rows = [[src] + [mean_std([m[k] for m in by_source[src]]) for k in
                     ["real_words", "distinct_2", "ended", "longest_copy"]] for src in sources]
    sections.append("## Automatic metrics\n\n" + table(
        ["source", "real word rate", "distinct-2", "ended on its own", "longest copied span (words)"], rows))

    judge_files = sorted(BENCH_DIR.glob("judgments_*.jsonl"))
    all_judgments = {}
    for path in judge_files:
        name = path.stem.removeprefix("judgments_")
        judgments = {j["id"]: j for j in load_jsonl(path) if j["id"] in stories}
        controls = [sid for sid, s in stories.items() if s["source"] in ("reference", "random")]
        if any(sid not in judgments for sid in controls):
            sections.append(f"## Judge: {name}\n\nThe reference and random stories aren't all judged yet, left out.")
            continue
        unjudged = sorted({stories[sid]["source"] for sid in stories if sid not in judgments})
        if unjudged:
            print(f"{name}: not judged yet, left out of its tables: {unjudged}")
        all_judgments[name] = judgments
        for j in judgments.values():
            j["weighted"] = weighted_score(j)

        groups = defaultdict(list)
        for sid, j in judgments.items():
            groups[stories[sid]["source"]].append(j)

        floor = statistics.mean(j["weighted"] for j in groups["random"])
        ceiling = statistics.mean(j["weighted"] for j in groups["reference"])

        def normalised(js):
            return 100 * (statistics.mean(j["weighted"] for j in js) - floor) / (ceiling - floor)

        rows = [[src, len(groups[src])] + [mean_std([j[c] for j in groups[src]]) for c in CRITERIA]
                + [mean_std([j["weighted"] for j in groups[src]]), str(round(normalised(groups[src])))]
                for src in sources if groups[src]]
        sections.append(f"## Judge: {name}\n\n{JUDGE_NOTES.get(name, name)}\n\n" + table(
            ["source", "n", *CRITERIA, "weighted", "normalised"], rows))

        rows = []
        for src in [m for m in models if groups[m]]:
            row = [src, str(round(normalised(groups[src])))]
            for g in PROMPTS:
                js = [j for j in groups[src] if stories[j["id"]]["group"] == g]
                row.append(str(round(normalised(js))) if js else "-")
            rows.append(row)
        prompt_list = "\n".join(f"- **{GROUP_NAMES[g]}**: " + ", ".join(repr(p) for p in ps)
                                for g, ps in PROMPTS.items())
        sections.append(f"### Normalised score by prompt type ({name})\n\n"
                        "How well each model holds up as the opening text gets less like the training data "
                        "(0 = random characters, 100 = real TinyStories).\n\n" + prompt_list + "\n\n"
                        + table(["model", "overall", *GROUP_NAMES.values()], rows))

    if len(all_judgments) >= 2:
        names = list(all_judgments)
        a, b = all_judgments[names[0]], all_judgments[names[1]]
        common = a.keys() & b.keys()
        rows = []
        for c in [*CRITERIA, "weighted"]:
            diffs = [abs(a[i][c] - b[i][c]) for i in common]
            within = sum(d <= 1 for d in diffs) / len(diffs)
            rows.append([c, f"{statistics.mean(diffs):.2f}", f"{100 * within:.0f}%"])
        sections.append(f"## Judge agreement: {names[0]} vs {names[1]} ({len(common)} stories)\n\n" + table(
            ["criterion", "mean abs difference", "within 1 point"], rows))

    summary = "\n\n".join(sections) + "\n"
    RESULT_FILE.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"saved {RESULT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["generate", "import", "judge", "report"])
    parser.add_argument("file", nargs="?", help="for import: a stories file made with generate --weights")
    parser.add_argument("--weights", help="generate: stories for just this weights file (e.g. on Kaggle)")
    parser.add_argument("--out", help="generate --weights: where to save the stories")
    parser.add_argument("--judge-model", default="gemini-3.8-flash")
    parser.add_argument("--provider", default="gemini", choices=["gemini", "openrouter"])
    parser.add_argument("--rpm", type=float, default=10, help="max judge requests per minute")
    args = parser.parse_args()

    if args.step == "generate":
        generate(args.weights, args.out)
    elif args.step == "import":
        import_stories(args.file)
    elif args.step == "judge":
        judge(args.judge_model, args.provider, args.rpm)
    else:
        report()
