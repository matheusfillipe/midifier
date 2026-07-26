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

One job runs at a time, and each takes roughly three times the length of the song, so jobs are
polled rather than waited on. While a job waits, `GET /v1/jobs/<id>` reports `queue_ahead` and
`eta_seconds`, and `GET /v1/queue` reports how busy the service is. The estimate is measured from
completed jobs rather than configured, so it reflects whatever hardware this runs on.

The OpenAPI schema is at `/openapi.json`.

## Secure it

The service holds only the *hash* of an API key, so reading the deployed secret does not
let anyone call it. Generate a pair:

```bash
uv run python -m midifier keygen
```

Deploy the hash as `MIDIFIER_API_KEY_HASH`, and give the key to callers. With no hash
configured the service is open, which suits local use.

Callers present the key one of three ways, whichever their client makes easiest:

| caller | how |
|---|---|
| REST | `X-API-Key: <key>` |
| MCP over HTTP | `X-API-Key: <key>` or `Authorization: Bearer <key>` |
| MCP over stdio | the `api_key` tool argument |

The MCP server is served from the same app at `/mcp`, so one URL and one key cover both
surfaces, and a job started over MCP is visible over REST.

## Configure it

Every setting is an environment variable prefixed `MIDIFIER_`. See
[`.env.example`](.env.example) for the full list. The ones that matter:

| variable | default | |
|---|---|---|
| `MIDIFIER_API_KEY_HASH` | unset | when set, callers must present the key |
| `MIDIFIER_STORAGE_BACKEND` | `local` | `local` or `s3` |
| `MIDIFIER_MINIO_BUCKET` | — | with the other `MINIO_*` values, for `s3` |
| `MIDIFIER_MODEL_SIZE` | `medium` | `small`, `medium` or `large` |
| `MIDIFIER_HF_TOKEN` | — | needed to download the transcription weights |
| `MIDIFIER_MAX_DURATION_SECONDS` | `360` | longest song accepted |
| `MIDIFIER_MAX_CONCURRENT_JOBS` | `1` | transcriptions run at a time |

## Develop it

```bash
uv run pre-commit run --all-files   # everything CI runs
uv run pytest --cov                 # tests, 80% gate
```

Conventions and the reasoning behind the pipeline are in [AGENTS.md](AGENTS.md).

## Licence

MIT. The transcription weights it downloads are licensed separately and are not redistributed
here.
