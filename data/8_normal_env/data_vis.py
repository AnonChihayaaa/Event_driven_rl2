import pandas as pd

# 指定.pkl文件的路径
file_path = 'data/8_normal_env/train.pkl'

# 使用read_pickle函数读取数据
df = pd.read_pickle(file_path)

# 显示DataFrame的前5行
print(df)