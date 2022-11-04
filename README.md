# counsel_bot

## Curent progress:
#### MAML
Data prepared for meta learning are under `data/meta_learn` \
Currently working on applying MAML on GPT2 (`gpt2/maml.py`)

#### Finetuning T5x finetuning.
Custom task registration is in `t5x/t5x/register_task.py`.
Run finetuning with `bash t5x/finetune.sh` and task-specific configs are in `t5x/counsel-bot_finetune.gin`. \
documentation: https://github.com/google-research/t5x/blob/main/docs/usage/finetune.md

tensorflow environment: \
`docker run --gpus all -it --rm  -v ~/counsel_bot:/home --runtime=nvidia tensorflow/tensorflow:latest-gpu` \
The commands in the Dockerfile are also necessary.

For running locally, install TensorRT: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar \
Symbolic links may be needed: \
`sudo ln -s libnvinfer.so.8 libnvinfer.so.7`
`sudo ln -s libnvinfer_plugin.so.8 libnvinfer_plugin.so.7`

## Reference models:
DialoGPT: https://github.com/microsoft/DialoGPT
* GODEL: https://github.com/microsoft/GODEL (a predecessor of DialoGPT, aiming for knowledge-grounded goal-oriented dialogue; not sure if it will work well on our case)

T5X: https://github.com/google-research/t5x (an improved implementation of T5, a text-to-text model that is easy to tranfer)

