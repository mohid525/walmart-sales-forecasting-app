import pandas as pd

df = pd.read_csv("data/cleaned_engineered_data.csv")
df_small = df.head(10000)
df_small.to_csv("data/cleaned_engineered_data_small.csv", index=False)

print("Done!")
