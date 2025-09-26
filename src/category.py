# category.py

from run import UrlCategory

def compute(payload: dict) -> str:
    """
    Returns the category of model.
    """
    category = payload.get("category")
    if not category:
        return UrlCategory.OTHER.value
    return category
