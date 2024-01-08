

grammar_template = """
    Correct just the grammatical errors of “Text:” in standard English and place the result as the answer.
    Please, I want the given text back as similar as possible to the original text, but perfectly written in standard English.:
     Text: {input}
"""


style_template = """
    Change the style of “Text:” according the following rules “Style:” and place the result as the answer:
     Text: {input}
     Style: {context}
"""

summarization_template_default = """
    Please provide a summary of the following text
    
    TEXT:
    {input}
"""

summarization_template_basic = """
    Please provide a summary of the following text.
    Please provide your output in a manner that a 5 year old would understand
    
    TEXT:
    {input}
"""

summarization_template_advanced = """
    You will be given a single piece of a big document, could be a book or a large paper or essay.
    This section will be enclosed in triple backticks (```).
    Your goal is to give a summary of this section so that a reader will have a full understanding of what happened.
    Your response should be at least three paragraphs and fully encompass what was said in the passage.
    
    ```{input}```
    FULL SUMMARY:
"""

summarization_map_template = """
    Write a concise summary of the following:
    "{input}"
    CONCISE SUMMARY:
"""

summarization_combine_template = """
    Write a concise summary of the following text delimited by triple backquotes.
    Return your response in bullet points which covers the key points of the text.
    ```{input}```
    BULLET POINT SUMMARY:
"""

summarization_refine_template = """
    "Here's your first summary: {prev_response}. "
    "Now add to it based on the following context: {input}"
"""