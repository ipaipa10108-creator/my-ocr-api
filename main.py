import base64
import io
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
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
        img = Image.open(io.BytesIO(img_bytes))

        # 1. 攔截 PR_Remover 的 1x1 連線測試探針，直接回傳成功
        if img.width < 10 or img.height < 10:
            return {"code": 100, "data": []}

        # 2. 自動處理透明通道/單雙通道，轉為標準 RGB 3 通道
        if img.mode != "RGB":
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                img = img.convert("RGBA")
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg
            else:
                img = img.convert("RGB")

        img_np = np.array(img)
        result, _ = engine(img_np)

        if not result:
            return {"code": 100, "data": []}

        formatted_data = [
            {"box": box, "score": float(score), "text": str(text)}
            for box, text, score in result
        ]
        return {"code": 100, "data": formatted_data}
    except Exception as e:
        return {"code": 500, "data": str(e)}
