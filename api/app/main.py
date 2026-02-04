from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Training Track API")

class Item(BaseModel):
    text: str | None = None
    is_done: bool = False


items: list[Item] = []


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/items", response_model=list[Item])
def create_item(item: Item):
    items.append(item)
    return items

@app.get("/items", response_model=list[Item])
def list_items(limit: int = 10):
    return items[0:limit]

@app.get("/items/{item_id}", response_model=Item)
def get_item(item_id: int) -> Item:
    if 0 <= item_id < len(items):
        return items[item_id]
    raise HTTPException(status_code=404, detail=f"Item {item_id} not found")