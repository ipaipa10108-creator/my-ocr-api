import base64
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from rapidocr_onnxruntime import RapidOCR

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = RapidOCR()

@app.get("/")
def health_check():
    return {"status": "running"}

@app.post("/api/ocr")
async def ocr_endpoint(req: Request):
    try:
        payload = await req.json()
        img_b64 = payload.get("base64", "")
        if not img_b64:
            return {"code": 101, "data": "Empty image payload"}

        img_bytes = base64.b64decode(img_b64)
        result, _ = engine(img_bytes)

        if not result:
            return {"code": 100, "data": []}

        formatted_data = [
            {"box": box, "score": float(score), "text": str(text)}
            for box, text, score in result
        ]
        return {"code": 100, "data": formatted_data}
    except Exception as e:
        return {"code": 500, "data": str(e)}
