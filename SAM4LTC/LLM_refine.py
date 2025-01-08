# -*- coding: utf-8 -*-
import openai
from openai import OpenAI
import pandas as pd
import time


def ask_gpt_question(sentence, triple):
    human, time, location  = triple
    # openai.api_key = 'sk-ZdrxA'
    openai.api_key = 'your api key'
    response = openai.chat.completions.create(
        model='gpt-3.5-turbo',
        temperature=0,
        messages=[{"role": "system",
                   "content": ""
                   },
                  {"role": "user", "content":  f"""
                        Please rewrite the sentence according to the requirements.
                        Requirements:

                            \item 1. Do not change the original meaning of the sentence
                            \item 2. Use the exactly same words in the list: [{human}, {time} {location}]
                            \item 3. You can add, reduce, or modify some other words or phrases, such as verbs
                            \item 4. The rewritten sentence cannot be the same as the original sentence
                        
                        Sentence that needs to be rewritten:{sentence}
                   """}]
    )
    return response.choices[0].message.content
