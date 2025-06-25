from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.routes.routes import router


app = FastAPI()

app.include_router(router)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    custom_errors = []

    for error in errors:
        loc = error["loc"]
        msg = error["msg"]
        if error["type"] == "missing":
            custom_msg = f"Le champ '{loc[-1]}' est requis mais manquant dans la requête."
        else:
            custom_msg = msg
        custom_errors.append({
            "champ": loc[-1],
            "message": custom_msg
        })

    return JSONResponse(
        status_code=422,
        content={"erreurs": custom_errors}
    )