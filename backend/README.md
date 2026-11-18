# TruthLens Backend

Start locally:

```bash
python -m pip install -r requirements.txt
python -m uvicorn backend.app.main:app --reload --port 8000
```

API documentation: http://127.0.0.1:8000/docs

Copy `.env.example` to `.env` and add optional API keys. The app uses SQLite by default. Set `DATABASE_URL=postgresql://...` for PostgreSQL with SQLAlchemy/Alembic-ready models.
