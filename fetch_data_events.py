import psycopg2
import pandas as pd
import os

DB_CONFIG = {
    "dbname": "FAIC album",
    "user": "postgres",
    "password": "08102005",
    "host": "localhost", 
    "port": "5432",
}

conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()
query = "SELECT album.id, album.name,album.description, image.image_base64 FROM album join image on album.id = image.album_id"
cursor.execute(query)
data = cursor.fetchall()
conn.close()

df = pd.DataFrame(data, columns=["id", "name","description","image_base64",])
