import psycopg2
import pandas as pd
import os

DB_CONFIG = {
    "dbname": "Info",
    "user": "postgres",
    "password": "08102005",
    "host": "localhost",  
    "port": "5432",
} 

conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()
query = "SELECT ho_ten,gen, mssv, gioi_tinh, ngay_sinh, ban, chuc_vu, link_fb FROM thanhvienfaic"
cursor.execute(query)
data = cursor.fetchall()
conn.close()

df = pd.DataFrame(data, columns=["ho_ten","gen", "mssv", "gioi_tinh", "ngay_sinh", "ban", "chuc_vu", "link_fb"])
