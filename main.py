from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from retinaface import RetinaFace
import cv2
import numpy as np
import io
import base64
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/detectar_rostos")
async def detectar_rostos(file: UploadFile = File(...)):
    # Ler a imagem do arquivo enviado
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Detectar rostos na imagem
    faces = RetinaFace.detect_faces(img)

    # Contar a quantidade de rostos
    quantidade_rostos = len(faces)

    # Desenhar retângulos ao redor dos rostos detectados
    for face in faces.values():
        facial_area = face["facial_area"]
        x, y, w, h = facial_area
        cv2.rectangle(img, (x, y), (w, h), (255, 0, 0), 2)

    # Converter a imagem para base64
    _, img_encoded = cv2.imencode(".png", img)
    img_base64 = base64.b64encode(img_encoded).decode("utf-8")

    # Retornar a imagem processada e a quantidade de rostos
    return JSONResponse(
        content={
            "quantidade_rostos": quantidade_rostos,
            "imagem_processada": img_base64,
        }
    )
