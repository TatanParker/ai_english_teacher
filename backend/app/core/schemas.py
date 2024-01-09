from pydantic import BaseModel, NonNegativeInt, confloat

from app.core.enums import OpenAIModels


class AbstractModel(BaseModel):
    model_config = dict(from_attributes=True)


class LLMParams(AbstractModel):
    openai_api_key: str | None = None
    model: OpenAIModels = OpenAIModels.GPT3
    temperature: confloat(ge=0.0, le=1.0)
    max_tokens: NonNegativeInt
    frequency_penalty: confloat(ge=0.0, le=1.0)
    top_p: confloat(ge=0.0, le=1.0)
