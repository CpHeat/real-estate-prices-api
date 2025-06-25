from typing import List

from fastapi import APIRouter, Query, HTTPException, Depends

from app.models.models import UserSummary, User, users

router = APIRouter()


@router.get("/users/",
    response_model=List[UserSummary],
    response_description="A list of users",
    tags=["users"])
def get_users(skip:int = Query(0, ge=0), limit: int = Query(None, ge=1)) -> list[UserSummary]:
    """
    Returns a list of users.

    - **skip**: How many users to skip (optional - default = 0).
    - **limit**: How many users to return (optional - minimum = 1).
    """
    if limit is None:
        selected_users = users[skip:]
    else:
        selected_users = users[skip:skip + limit]
    return [UserSummary(id = u.id, login=u.login) for u in selected_users]

@router.get("/users/search",
    response_model=List[UserSummary],
    response_description="A list of users",
    tags=["users"])
def search_users(q: str = Query(..., min_length=1)) -> list[UserSummary] | str:
    """
    Returns a list of users whose login contains the specified string.

    - **q**: The string to search for.
    - **username**: An authenticated user's username.
    """
    found_users = []
    for u in users:
        if q.lower() in u.login.lower():
            found_users.append(u)
    return [UserSummary(id = u.id, login=u.login) for u in found_users]

@router.get("/users/{user_login}",
    response_description="The user's details",
    tags=["users"])
def get_user(user_login: str) -> User:
    """
    Returns details about a specific user.

    - **user_login**: The login to search for.
    - **username**: An authenticated user's username.
    """
    for u in users:
        if u.login == user_login:
            return u
    raise HTTPException(status_code=404, detail="User not found")