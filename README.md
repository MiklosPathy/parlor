# Parlor

Fork of [fikrikarim/parlor](https://github.com/fikrikarim/parlor) with the following changes:

- Switched to **Gemma 4 E4B** model (from E2B)
- **Hungarian TTS** via Piper (`hu_HU-anna-medium`)
- **GPU backend** enabled (ROCm/OpenCL)
- **MODEL env var** to switch between `4b` and `12b` at startup
- Camera/vision disabled when using the 12B model
- System prompt defaults to Hungarian

## Running

```bash
PIPER_MODEL=hu_HU-anna-medium uv run python -u -m uvicorn server:app \
  --host 0.0.0.0 --port 8443 \
  --ssl-keyfile key.pem --ssl-certfile cert.pem
```

Switch to 12B model:

```bash
MODEL=12b PIPER_MODEL=hu_HU-anna-medium uv run python -u -m uvicorn server:app \
  --host 0.0.0.0 --port 8443 \
  --ssl-keyfile key.pem --ssl-certfile cert.pem
```

SSL cert generálása (első indítás előtt):

```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
```
