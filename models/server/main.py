# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.forest_router import router as forest_router
from routers.seeds_router import router as seeds_router
from routers.mushroom_router import router as mushroom_router

app = FastAPI(title="Forest Cover Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],           
    allow_headers=["*"],            
)


app.include_router(forest_router)
app.include_router(seeds_router)
app.include_router(mushroom_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)