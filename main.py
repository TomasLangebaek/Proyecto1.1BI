from fastapi import Request, FastAPI
import pandas as pd
import Models
import json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8080",
    "http://localhost:8000",

]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ML_model = Models.SvmModel()


@app.post("/predict")
def make_predictions(prediction_model: Models.DataModel):
    df = pd.DataFrame(prediction_model.dict(),
                      columns=prediction_model.dict().keys(), index=[0])
    result = ML_model.make_predictions(df)
    return result.tolist()


@app.post("/predictAll")
async def predictAll(request_body: Request):
    req_info = await request_body.json()
    json_string = json.dumps([ob for ob in req_info])
    df = pd.read_json(json_string)
    df.columns = Models.columns()

    result = ML_model.make_predictions(df)
    return result.tolist()


@app.get("/")
def read_root():
    return {"Hello": "World"}


# if __name__ == '__main__':
#    uvicorn.run(app, host='0.0.0.0', port=8000)
