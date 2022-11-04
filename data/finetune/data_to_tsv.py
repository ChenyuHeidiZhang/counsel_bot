import pandas as pd

df = pd.read_csv("../../counsel-chat/data/20200325_counsel_chat.csv", encoding='utf-8')

df_selected = df[['questionText', 'answerText']]
df_selected['questionText'] = df_selected['questionText'].str.replace('\n', ' ')
df_selected['questionText'] = df_selected['questionText'].str.replace('    ', ' ')

df_selected['answerText'] = df_selected['answerText'].str.replace('\n', ' ')
df_selected['answerText'] = df_selected['answerText'].str.replace('    ', ' ')

from sklearn.model_selection import train_test_split

train, test = train_test_split(df_selected, test_size=0.1)
# print(test)
print(train)

train.to_csv('counselchat_train.tsv', sep="\t", header=False, index=False)
test.to_csv('counselchat_test.tsv', sep="\t", header=False, index=False)