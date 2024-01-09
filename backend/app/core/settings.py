import toml
from pydantic import model_validator, NonNegativeInt, confloat
from pydantic_settings import BaseSettings

from app.core.enums import OpenAIModels


class Settings(BaseSettings):
    PROJECT_NAME: str = "Backend API"
    PROJECT_DESCRIPTION: str = "Backend API"
    PROJECT_VERSION: str = "0.1.0"

    @model_validator(mode="before")
    def validate_project_settings(cls, values: dict[str, any]) -> any:
        """Validate project settings."""
        with open("pyproject.toml", "r") as file:
            project = toml.load(file)
        poetry = project["tool"]["poetry"]
        values["PROJECT_NAME"] = values.get("PROJECT_NAME", poetry["name"])
        values["PROJECT_DESCRIPTION"] = values.get(
            "PROJECT_DESCRIPTION", poetry["description"]
        )
        values["PROJECT_VERSION"] = values.get("PROJECT_VERSION", poetry["version"])
        return values

    # OPENAI SETTINGS
    OPENAI_API_KEY: str
    MODEL: OpenAIModels = OpenAIModels.GPT3

    TEMPERATURE: confloat(ge=0.0, le=1.0) = 0.7
    MAX_TOKENS: NonNegativeInt = 2000
    FREQUENCY_PENALTY: confloat(ge=0.0, le=1.0) = 0.0
    TOP_P: confloat(ge=0.0, le=1.0) = 1.0


settings = Settings()
