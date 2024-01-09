from enum import StrEnum


class OpenAIModels(StrEnum):
    DAVINCI = "davinci-002"
    TEXT_DAVINCI = "text-davinci-003"
    BABBAGE = "babbage-002"
    ADA = "text-ada-001"
    GPT3 = "gpt-3.5-turbo-instruct"


class TeacherActions(StrEnum):
    GRAMMAR = "grammar"
    STYLE = "style"
    SUMMARIZATION = "summarization"


class StyleTypes(StrEnum):
    FREE = "Free"
    CONCRETE = "Concrete"
    RULES = "Set of rules"
    WEBPAGE = "From a webpage"
    DOCUMENT = "From a document"


class SummarizationTypes(StrEnum):
    DEFAULT = "Default"
    BASIC = "Basic"
    WEBPAGE = "From a webpage"
    DOCUMENT = "From a document"


class SummarizationChainTypes(StrEnum):
    STUFF = "stuff"
    MAP_REDUCE = "map_reduce"
    REFINE = "refine"
