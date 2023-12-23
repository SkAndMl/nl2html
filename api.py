from model import load_model
import contextlib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio
from pyngrok import ngrok
import uvicorn

text_gen = load_model()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get('/query/{text}')
def get_course(text: str):
    output = text_gen(f"<s>[INST] {text} [/INST]")
    response = {
        'value' : output[0]["generated_text"]
    }
    return response

if __name__ == "__main__":

    ngrok_tunnel = ngrok.connect(8000)
    print('Public URL:', ngrok_tunnel.public_url)
    nest_asyncio.apply()
    uvicorn.run(app, port=8000)