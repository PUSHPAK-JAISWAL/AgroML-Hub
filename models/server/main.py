# main.py
from fastapi import FastAPI
from routers.forest_router import router as forest_router
from routers.seeds_router import router as seeds_router

app = FastAPI(title="Forest Cover Prediction API")
app.include_router(forest_router)
app.include_router(seeds_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)