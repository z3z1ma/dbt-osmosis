import os

import openai


def create_llm_prompt(sql_content, filename):
    header = """DO NOT ADD A HEADER TO DBT YAML.
THIS CODE WILL APPEND TO AN EXISTING YAML FILE.

Examples of YAML structure:

"""
    prompt = f""" 
You are a helpful SQL Developer and Expert in dbt. 
Your job is to receive a SQL and generate the YAML in dbt format.
You will not respond anything else, just the YAML code formated to be saved into a file.

IMPORTANT RULES:

1. DO NOT PROSE.
2. DO NOT DEVIATE OR INVENT FROM THE CONTEXT. 
3. Always follow dbt convetion!
4. The context will always be ONE FULL SQL.
5. DO NOT WRAP WITH MARKDOWN.
6. The model name will always be the file name.
7. NO NEW LINE BETWEEN COLUMNS!

{header}

  - name: model_name
    description: markdown_string

    columns:
      - name: column_name
        description: markdown_string
      - name: column_name
        description: markdown_string
      - name: column_name
        description: markdown_string
      - name: column_name
        description: markdown_string

INCLUDE TESTS IF YOU KNOW WHAT THE COLUMN NEEDS.

File Name to be used as MODEL NAME: {filename}

Convert the following DBT SQL code to YAML:
"""
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": sql_content},
    ]
    return messages
