from fastapi import FastAPI
import pandas as pd
import Models


app = FastAPI()
ML_model = Models.SvmModel()


@app.post("/predict")
def make_predictions(prediction_model: Models.DataModel):
    df = pd.DataFrame(prediction_model.dict(), columns=prediction_model.dict().keys(), index=[0])
    result = ML_model.make_predictions(df)
    return result.tolist()


@app.get("/")
def read_root():
    return {"Hello": "World"}


# if __name__ == '__main__':
#    uvicorn.run(app, host='0.0.0.0', port=8000)
