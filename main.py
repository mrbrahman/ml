import uvicorn
from src.api.routes import app
from src.infrastructure.config import HOST, PORT

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)