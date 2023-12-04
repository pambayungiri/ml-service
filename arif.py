
from typing import Union
from typing import List
from fastapi import FastAPI
import pickle
app = FastAPI()


@app.get("/")
def read_root():
    return {"project": "Mpaas Engine Demo"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/")
async def pred_value(value_list : List[float]):
    with open('classifier_rf.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    y_pred = loaded_model.predict([value_list])
    print(int(y_pred[0]))
    if int(y_pred[0]) == 0:
        return "hadron"
    else:
        return "gamma"
    
