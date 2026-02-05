import pandas as pd 

df = pd.read_excel("/home/alencaravelar/Desktop/zetta-lab/zetta-lab2/zetta-lab2/data/raw/dados-bh.xls")

df.to_csv("dados_bh.csv", index=False, encoding="utf-8")
