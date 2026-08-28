import pandas as pd 

df = pd.read_parquet("data/mini/rollout_0000.parquet")

print(df.columns)

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(df['state'].iloc[0][0:6])
def separation_same_car(df):
    K = 4  
    env, ep = df["env"].to_numpy(), df["episode"].to_numpy()
    index = [i for i in range(len(df) - K + 1) if env[i] == env[i + K - 1] and ep[i] == ep[i + K - 1]]
    return index 



print(separation_same_car(df))

        




