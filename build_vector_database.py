import pandas as pd
from angle_emb import AnglE
import sqlite3
from tqdm import tqdm

df = pd.read_csv("raw/filtered_oracle_database.csv")

formatted_rows = []
card_names = []
for index, row in df.iterrows():
    formatted_row = ""
    for column_name, value in row.items():
        formatted_row += f"{column_name}: {value}\n"
    card_names.append(row['name'])
    formatted_rows.append(formatted_row.strip())

# Chunk data into 5 card piles with 2 newlines between each card
chunked_data = []
chunked_names = []
current_chunk = ""
current_name_chunk = ""
for i, row in enumerate(formatted_rows):
    current_chunk += row + "\n\n"
    current_name_chunk += card_names[i] + "\n"
    if len(current_chunk.split('\n\n')) == 6:  # Each chunk contains 5 cards and 1 extra newline character
        chunked_data.append(current_chunk.strip())
        current_chunk = ""
        chunked_names.append(current_name_chunk.strip())
        current_name_chunk = ""

# If there are remaining cards not included in chunks
if current_chunk:
    chunked_data.append(current_chunk.strip())
    chunked_names.append(current_name_chunk.strip())

# Use embeddings model with high semantic accuracy
angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()

conn = sqlite3.connect('raw/full_card_vector_database.db')
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS vectordb
             (id INTEGER PRIMARY KEY, name TEXT, card_text TEXT, vector BLOB)''')

# Encode and save the encodings along with the corresponding indices, name, and text
for i, (names, chunked) in enumerate(tqdm(zip(chunked_names, chunked_data), desc="Encoding and saving")):
    encodings = angle.encode(names, to_numpy=True)
    
    c.execute("INSERT INTO vectors (id, name, text, vector) VALUES (?, ?, ?, ?)",
              (i, names, chunked, encodings.tobytes()))

conn.commit()
conn.close()