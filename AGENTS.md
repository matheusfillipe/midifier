# AGENTS.md

Guidance for agents working in this repo. `CLAUDE.md` is a symlink to this file, so there is
one set of instructions rather than two that drift.

## What this is

An API that turns a recording into a multi-track General MIDI file: it identifies which
instruments are playing, transcribes each, and names and assigns the tracks. It feeds
[kinesthesia](https://github.com/h4ks-com/kinesthesia), which renders MIDI as falling notes to
practise against.

## Toolchain

`uv` for everything. Never call `pip` or a bare `python`.

```bash
uv sync --extra dev          # install
uv run pytest                # tests
uv run mypy src tests        # types, strict
uv run ruff check --fix .    # lint
uv run ruff format .         # format
uv run pre-commit run -a     # everything CI runs
uv run python -m midifier    # serve the API
```

`uvx` for one-off tools that are not dependencies.

## Non-negotiables

- **mypy strict.** No `Any`, no untyped defs, no `# type: ignore` without a reason on the same
  line. If typing something is hard, that usually means the design is unclear.
- **One import per line.** `force-single-line = true`; do not group with commas or parentheses.
- **Coverage stays at or above 80%.** Add tests with the code, not afterwards.
- **Every check runs in pre-commit**, and CI runs pre-commit. Adding a check anywhere else
  splits the definition of "clean" in two.
- **No bare `except Exception`.** Catch what can actually be raised.

## Style

Comments are rare and earn their place. The code says what it does; a comment exists only for a
non-obvious *why*, a constraint, or a trap that would invite a wrong "simplification". Never
narrate a change ("was X, now Y") — write code as though it always looked this way.

Prefer deleting to adding. The pipeline reached its current quality mostly by removing stages
that sounded clever and measured worse; treat new stages with the same suspicion.

## Where things are

```
src/midifier/
  config.py        settings, all from the environment, prefix MIDIFIER_
  api.py           REST surface, async job pattern
  mcp.py           MCP tools over the same job store
  storage.py       local filesystem or S3/MinIO behind one protocol
  fetch.py         SSRF-guarded URL fetching
  jobs.py          job model and in-memory store
  midi/
    cleanup.py     repairs the decoder's characteristic defects
    segments.py    decodes long audio in overlapping pieces and stitches them
    consolidate.py folds lanes that are one part under two names
tests/             mirrors src; e2e/ needs the model and is opt-in
```

## Documentation

Keep it current as part of the change, not as a follow-up.

- `README.md` — brief and human. What it is, how to run it, how to configure it. No architecture
  essays, no feature lists that go stale.
- `docs/` — anything longer, and only if it is real. Never document intended behaviour.
- Docstrings carry the *why*. The signature already carries the what.

When behaviour changes, update the affected docs in the same commit. If a doc describes
something that no longer exists, delete it rather than softening it.

## Testing

Unit tests are pure and fast. Anything needing the transcription model or real audio is marked
`e2e` and skipped unless `MIDIFIER_E2E=1`, because the model is multi-gigabyte and gated.

Test the behaviour that was hard to get right — the cleanup thresholds, the seam repair,
the SSRF guards — not the framework. A test that only proves FastAPI routes are
plumbed is noise.

## Things that are settled

Decided by measurement and listening; see `claudedocs/PLAN.md` for the evidence. Do not
reintroduce these without new evidence:

- No source separation. Transcribing the full mix beats separating first.
- No quantization. Snapping to a detected beat grid sounded worse at every strength. The
  decoder's own tempo detection is off for the same reason, and because stitching builds a new
  file and would discard a per-segment guess anyway.
- No beam search. On `large` it returns byte-identical output for a multiple of the compute.
- No classifier-free guidance. It splits one song into more instrument classes, not fewer.
- Cleanup never judges a repeat by note *length*. Note lengths are note values at the song's
  tempo, so any fixed threshold is a tempo threshold and decides differently per song.
- No synthesised velocity. Measuring loudness from a full mix produces incoherent dynamics.
- Greedy decoding, not temperature sampling. Sampling fixes the ending and degrades the rest;
  the ending is repaired in post instead.
