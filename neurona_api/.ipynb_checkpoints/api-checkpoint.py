from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI()
PAGINA = Path(__file__).parent / "index.html"


class Datos(BaseModel):
    # cada entrada es 0 o 1, como en la neurona M-P
    x1: int = Field(ge=0, le=1)
    x2: int = Field(ge=0, le=1)
    x3: int = Field(ge=0, le=1)
    x4: int = Field(ge=0, le=1)
    umbral: int = Field(ge=0, le=4)


class MPNeuron:
    def __init__(self, threshold):
        self.threshold = threshold

    def predict(self, x):
        # suma los 1 y responde True solo si llega al umbral
        return sum(x) >= self.threshold


@app.get("/")
def pagina():
    # el navegador pide esta ruta y recibe el formulario
    return FileResponse(PAGINA)


@app.post("/predecir")
def predecir(datos: Datos):
    # el html manda el json y aqui sale la prediccion
    entradas = [datos.x1, datos.x2, datos.x3, datos.x4]
    neurona = MPNeuron(datos.umbral)
    suma = sum(entradas)
    return {"suma": suma, "prediccion": neurona.predict(entradas)}
