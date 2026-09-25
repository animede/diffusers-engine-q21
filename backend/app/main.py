from __future__ import annotations

import io
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from .service import service
from .settings import ACCELERATION, CORS_ORIGINS, OUTPUT_DIR


app = FastAPI(
    title="Diffusers Engine Q21 API",
    version="1.0.0",
    description="Local Diffusers inference API for Qwen/Qwen-Image-2.1",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


@app.get("/")
def root() -> dict[str, str]:
    return {"name": "Diffusers Engine Q21 API", "docs": "/docs", "health": "/api/health"}


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "model": service.model_info()}


@app.post("/api/model/load", status_code=202)
def load_model() -> dict[str, str]:
    if not service.model_downloaded:
        raise HTTPException(status_code=409, detail="モデルがダウンロードされていません。")
    return {"status": service.request_load()}


@app.post("/api/model/unload")
def unload_model() -> dict[str, str]:
    try:
        service.unload()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "not_loaded"}


@app.post("/api/jobs", status_code=202)
async def create_job(
    prompt: Annotated[str, Form(min_length=1, max_length=4000)],
    negative_prompt: Annotated[str, Form(max_length=4000)] = "",
    width: Annotated[int, Form(ge=256, le=2752)] = 1024,
    height: Annotated[int, Form(ge=256, le=2752)] = 1024,
    steps: Annotated[int, Form(ge=2, le=100)] = 40,
    guidance_scale: Annotated[float, Form(ge=1, le=10)] = 1.0,
    seed: Annotated[int, Form(ge=-1, le=2**63 - 1)] = -1,
    images: Annotated[list[UploadFile] | None, File()] = None,
) -> JSONResponse:
    if not prompt.strip():
        raise HTTPException(status_code=422, detail="プロンプトを入力してください。")
    if width % 16 or height % 16:
        raise HTTPException(status_code=422, detail="幅と高さは16の倍数にしてください。")
    uploads = images or []
    if len(uploads) > 10:
        raise HTTPException(status_code=422, detail="参照画像は最大10枚です。")
    if ACCELERATION == "viggle-r128" and len(uploads) > 3:
        raise HTTPException(
            status_code=422,
            detail="Viggle Turboは学習範囲に合わせて参照画像を最大3枚に制限しています。",
        )

    decoded: list[Image.Image] = []
    try:
        for upload in uploads:
            data = await upload.read(50 * 1024 * 1024 + 1)
            if len(data) > 50 * 1024 * 1024:
                raise HTTPException(status_code=413, detail=f"{upload.filename}: 50MBを超えています。")
            image = Image.open(io.BytesIO(data))
            image.load()
            image = ImageOps.exif_transpose(image)
            decoded.append(image.convert("RGBA" if "A" in image.getbands() else "RGB"))
    except (UnidentifiedImageError, OSError) as exc:
        for image in decoded:
            image.close()
        raise HTTPException(status_code=422, detail="読み込めない参照画像が含まれています。") from exc

    params = {
        "prompt": prompt.strip(),
        "negative_prompt": negative_prompt.strip(),
        "width": width,
        "height": height,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
    }
    job = service.submit(params, decoded)
    return JSONResponse(status_code=202, content={"id": job.id, "status": job.status})


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません。")
    return job
