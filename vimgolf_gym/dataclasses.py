"""
Shared Pydantic dataclasses for vimgolf-gym
"""

import json
from datetime import datetime
from pathlib import Path

from parse import parse
from pydantic import BaseModel, Field
from typing import Optional

class VimGolfCustomChallenge(BaseModel):
    """Represents a custom VimGolf challenge

    Attributes:
        input: The input of the challenge (required)
        output: The output of the challenge (required)
        name: The name of the challenge (optional)
        description: The description of the challenge (optional)
        solution: The VimGolf solution (optional)
    """
    input: str
    output: str
    name: Optional[str]
    description: Optional[str]
    solution: Optional[str]


class VimGolfEnvResult(BaseModel):
    """Represents a single result of VimGolf challenge environment,

    Attributes:
        correct: The challenge is solved or not
        keys: VimGolf solution keys (converted from raw representation)
        score: Keystrokes count (the lower the better)
    """

    correct: bool
    keys: str
    score: int


class VimGolfParsedPublicSolutionHeader(BaseModel):
    """Represents the header of a VimGolf public solution entry.

    Attributes:
        rank: The rank of the solution (provided as a string in the reference).
        score: The score of the solution (provided as a string in the reference).
        user_name: The name of the user who submitted the solution.
        user_id: The ID of the user who submitted the solution.
        date: The date the solution was submitted.
    """

    rank: str
    score: str
    user_name: str
    user_id: str
    date: datetime


class VimGolfPublicSolution(BaseModel):
    """Represents a public solution entry in VimGolf.

    Attributes:
        rank: The rank of the solution (provided as a string in the reference).
        solution: The sequence of Vim keystrokes used in the solution.
        header: Descriptive header containing user and challenge details.
    """

    rank: str
    solution: str
    header: str


class VimGolfChallengeMetadata(BaseModel):
    """
    Represents metadata for a VimGolf challenge.

    Attributes:
        href: URL path to the challenge.
        title: Name/description of the challenge.
        detail: Explanation of the challenge task.
        challenge_hash: Unique identifier hash for the challenge.
    """

    href: str
    title: str
    detail: str
    challenge_hash: str


class VimGolfChallengeDefinition(BaseModel):
    """
    Represents a VimGolf challenge definition including input/output code and client information.

    Attributes:
        input: Starting code and its type (nested model)
        output: Expected solution code and its type (nested model)
        client_version: Version of the client that created the challenge
    """

    class InputOutputModel(BaseModel):
        """
        Nested model for input/output code definitions.

        Attributes:
            data: The actual code content
            type: Programming language identifier (e.g., 'rb' for Ruby)
        """

        data: str
        type: str

    input: InputOutputModel = Field(..., alias="in")
    output: InputOutputModel = Field(..., alias="out")
    client_version: str = Field(..., alias="client")

    class Config:
        validate_by_name = True


def parse_json_file(file_path: Path, model: type[BaseModel]) -> BaseModel:
    """Parse a JSON file into a specified Pydantic model
    
    Args:
        file_path (Path): Path to the JSON file.
        model (type[BaseModel]): Pydantic model to parse the JSON into.
    
    Returns:
        BaseModel: Parsed model instance
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return model.parse_obj(data)


def parse_public_solution_header(input_str: str) -> VimGolfParsedPublicSolutionHeader:
    """Parse the public solution header string
    
    Args:
        input_str (str): Public solution header string.
    
    Returns:
        VimGolfParsedPublicSolutionHeader
    """
    template = "#{rank} {user_name} / @{user_id} - Score: {score} - {month}/{day}/{year} @ {hour}:{minute}"

    # Parse the input string using the template
    parsed_data = parse(template, input_str)

    # Convert parsed fields to integers and process the year
    month = int(parsed_data["month"])
    day = int(parsed_data["day"])
    year = int(parsed_data["year"])
    hour = int(parsed_data["hour"])
    minute = int(parsed_data["minute"])

    # Adjust two-digit year to four digits
    full_year = year + 2000 if year < 70 else year + 1900

    # Create a datetime object and format the timestamp
    dt = datetime(full_year, month, day, hour, minute)
    timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")

    ret = VimGolfParsedPublicSolutionHeader(
        rank=parsed_data["rank"],
        score=parsed_data["score"],
        user_name=parsed_data["user_name"],
        user_id=parsed_data["user_id"],
        date=timestamp,
    )
    return ret
