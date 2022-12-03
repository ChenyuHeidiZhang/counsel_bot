import os
import numpy as np

path = '../formatted_results'
result_files = [os.path.join(path, f) for f in os.listdir(path)]
np.random.shuffle(result_files)

with open('file_orders.txt', 'w') as f:
    for result_file in result_files:
        f.write(result_file + '\n')

NUM_PER_PERSON = 20
with open('person1.txt', 'w') as out1, open('person2.txt', 'w') as out2, open('person3.txt', 'w') as out3:
    for fileid, result_file in enumerate(result_files):
        out1.write(f"===== BATCH {fileid+1} =====\n")
        out2.write(f"===== BATCH {fileid+1} =====\n")
        out3.write(f"===== BATCH {fileid+1} =====\n")
        with open(result_file, 'r') as f:
            id = 0
            saved = []
            for line in f:
                if line.startswith('====='):
                    if id < NUM_PER_PERSON:
                        outf = out1
                    elif id < NUM_PER_PERSON*2:
                        outf = out2
                    elif id < NUM_PER_PERSON*3:
                        outf = out3
                    outf.write('=== ' + str(id+1) + ' ===\n')
                    outf.write(saved[0] + saved[1])
                    saved = []
                    id +=1
                else:
                    saved.append(line)
                if id >= NUM_PER_PERSON*3: break
