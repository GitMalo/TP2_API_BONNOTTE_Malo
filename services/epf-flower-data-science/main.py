import uvicorn
from fastapi.responses import RedirectResponse
import kaggle

from src.app import get_application

app = get_application()

@app.get("/")
def redirect_to_doc():
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    uvicorn.run("main:app", debug=True, reload=True, port=8080)
