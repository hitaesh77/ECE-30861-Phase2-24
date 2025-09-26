# name.py

def compute(payload: dict) -> str:
    """
    Returns the name of the model.
    """
    name = payload.get("name")
    return name
