from fastapi import FastAPI

app = FastAPI(title="AI Data Extraction API")


@app.get("/health")
def health():
    return {"status": "ok"}
