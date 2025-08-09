# alpacium
Alpaca Client, but a bit sexy ; )

## Run API

Install dependencies and run FastAPI via Uvicorn:

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

VS Code launch configs are provided under `.vscode/launch.json`:
- API: Uvicorn (reload)
- API: Uvicorn (prod)

Environment variables (.env):
- `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`
- `SUPABASE_URL`, `SUPABASE_KEY`
- optional `TRANSFORMERS_CACHE`