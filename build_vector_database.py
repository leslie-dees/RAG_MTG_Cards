import json
import pandas as pd
from angle_emb import AnglE

df = pd.read_csv("raw/filtered_oracle_database.csv")

formatted_rows = []
for index, row in df.iterrows():
    formatted_row = ""
    for column_name, value in row.items():
        formatted_row += f"{column_name}: {value}\n"
    formatted_rows.append(formatted_row.strip())

# Chunk data into 5 card piles with 2 newlines between each card
chunked_data = []
current_chunk = ""
for row in formatted_rows:
    current_chunk += row + "\n\n"
    if len(current_chunk.split('\n\n')) == 6:  # Each chunk contains 5 cards and 1 extra newline character
        chunked_data.append(current_chunk.strip())
        current_chunk = ""

# If there are remaining cards not included in chunks
if current_chunk:
    chunked_data.append(current_chunk.strip())


angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()

import sqlite3
from tqdm import tqdm

# Connect to the database
conn = sqlite3.connect('card_vector_database.db')
c = conn.cursor()

# Create table
c.execute('''CREATE TABLE IF NOT EXISTS vectors
             (id INTEGER PRIMARY KEY, text TEXT, vector BLOB)''')

# Encode and save the encodings along with the corresponding indices
for i, chunked in enumerate(tqdm(chunked_data, desc="Encoding and saving")):
    encodings = angle.encode(chunked, to_numpy=True)
    
    # Insert the encoded chunk and its corresponding text into the database
    c.execute("INSERT INTO vectors (id, text, vector) VALUES (?, ?, ?)",
              (i, chunked, encodings.tobytes()))

# Commit changes and close the connection
conn.commit()
conn.close()