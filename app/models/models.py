from datetime import datetime
from typing import List

from pydantic import BaseModel


class User(BaseModel):
    """
    A user's details.

    :param id: The id of the user.
    :type id: int
    :param login: The username of the user.
    :type login: str
    :param created_at: The date the user was created.
    :type created_at: datetime
    :param avatar_url: The url of the user's avatar.
    :type: avatar_url: str
    :param bio: The bio of the user.
    :type: bio: str
    """
    id: int
    login: str
    created_at: datetime
    avatar_url: str
    bio: str

class UserSummary(BaseModel):
    """
    A user's summary (id & login).

    :param id: The id of the user.
    :type id: int
    :param login: The username of the user.
    :type login: str
    """
    id: int
    login: str

raw_users = load_filtered_users("data/filtered_users.json")
users: List[User] = []

for user in raw_users:
    if isinstance(user.get("created_at"), str):
        user["created_at"] = user["created_at"].replace("Z", "+00:00")
        user["created_at"] = datetime.fromisoformat(user["created_at"])
    users.append(User(**user))