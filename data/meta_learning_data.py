import pandas as pd

df = pd.read_csv("../../counsel-chat/data/20200325_counsel_chat.csv", encoding='utf-8')

df_selected = df[['questionText', 'answerText', 'topic']]
df_selected['questionText'] = df_selected['questionText'].str.replace('\n', ' ')
df_selected['questionText'] = df_selected['questionText'].str.replace('    ', ' ')

df_selected['answerText'] = df_selected['answerText'].str.replace('\n', ' ')
df_selected['answerText'] = df_selected['answerText'].str.replace('    ', ' ')


for topic in df_selected['topic'].unique():
    topic_data = df_selected[df_selected['topic'] == topic][['questionText', 'answerText']]
    topic_data.to_csv(f'meta_learn/{topic}.tsv', sep="\t", header=False, index=False)
