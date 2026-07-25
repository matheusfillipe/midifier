# midifier

Turns a song into a multi-track General MIDI file. It works out which instruments are playing,
transcribes each one, and names and assigns the tracks, so the result can be practised against
in [kinesthesia](https://github.com/h4ks-com/kinesthesia).

Available as a REST API and as MCP tools.

## Run it

```bash
uv sync --extra dev
uv run python -m midifier          # API on :8000, docs at /docs
uv run python -m midifier mcp      # MCP server
```

No configuration is needed to start; results are written to `./data`.

## Use it

```bash
# start a transcription
curl -F 'file=@song.mp3' http://localhost:8000/v1/jobs
# {"id":"...","state":"queued"}

# poll it
curl http://localhost:8000/v1/jobs/<id>
```

Transcription takes roughly twice the length of the song, so jobs are polled rather than waited
on. The OpenAPI schema is at `/openapi.json`.

## Configure it

Every setting is an environment variable prefixed `MIDIFIER_`. See
[`.env.example`](.env.example) for the full list. The ones that matter:

| variable | default | |
|---|---|---|
| `MIDIFIER_API_KEY` | unset | when set, callers must send `X-API-Key` |
| `MIDIFIER_STORAGE_BACKEND` | `local` | `local` or `s3` |
| `MIDIFIER_MINIO_BUCKET` | — | with the other `MINIO_*` values, for `s3` |
| `MIDIFIER_MODEL_SIZE` | `medium` | `small`, `medium` or `large` |
| `MIDIFIER_HF_TOKEN` | — | needed to download the transcription weights |

## Develop it

```bash
uv run pre-commit run --all-files   # everything CI runs
uv run pytest --cov                 # tests, 80% gate
```

Conventions and the reasoning behind the pipeline are in [AGENTS.md](AGENTS.md).

## Licence

MIT. The transcription weights it downloads are licensed separately and are not redistributed
here.
