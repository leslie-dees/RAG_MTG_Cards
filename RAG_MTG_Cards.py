import sqlite3
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine
from angle_emb import AnglE
import os
import configparser
from openai import OpenAI

config = configparser.ConfigParser()
config.read('config.ini')
OPENAI_API_KEY = config['API_KEYS']['OPENAI_API_KEY']

angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls')

vector_db_name = "raw/full_card_vector_database.db"


def semantic_search(query, vector_db_name=vector_db_name, number_chunks = 5):
    conn = sqlite3.connect(vector_db_name)
    c = conn.cursor()

    query_embedding = angle.encode(query, to_numpy=True).flatten()

    c.execute("SELECT id, name, card_text, vector FROM vectordb")
    rows = c.fetchall()

    similarities = []
    for row in rows :
        id_, name, card_text, vector_bytes = row
        stored_embedding = np.frombuffer(vector_bytes, dtype=np.float32).flatten()
        sim = 1 - cosine(query_embedding, stored_embedding)
        similarities.append((id_, card_text, sim))

    similarities.sort(key=lambda x: x[2], reverse=True)  # Sort by similarity scores in descending order

    top_matches = similarities[:number_chunks]

    conn.close()

    return [(match[1], match[2]) for match in top_matches]

def rag_query(query, RAG=True):
    chunks = semantic_search(query, vector_db_name, 3)
    
    prompt_rag = ""
    if RAG:
        chunks = semantic_search(query, vector_db_name, 3)

        prompt_rag = ""
        for chunk in chunks:
            prompt_rag += chunk[0]+"\n\n"

        prompt = f"""[INST]
        Given the following card data, provide me with the exact text of {query} in the format of :
        \nname: \nmana_cost: \ncmc: \ntype_line: \noracle_text: \npower: \ntoughness: \ncolors: \ncolor_identity: \nkeywords:

        \n{prompt_rag}

        Use only the data in the provided chunks above. [/INST]
        """
    else:
        prompt = f"""[INST]
        Provide me with the exact text of {query} in the format of :
        \nname: \nmana_cost: \ncmc: \ntype_line: \noracle_text: \npower: \ntoughness: \ncolors: \ncolor_identity: \nkeywords: [/INST]
        """

    
    client = OpenAI(api_key = OPENAI_API_KEY)

    response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "user", "content": prompt},
      ]
    )
    return response.choices[0].message.content