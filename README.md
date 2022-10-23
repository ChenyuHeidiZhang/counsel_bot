# counsel_bot

## Curent progress:
Finally able to set up T5x finetuning on a GPU.
Custom task registration is in `t5x/t5x/register_task.py`.
Run finetuning with `bash t5x/finetune.sh` and task-specific configs are in `t5x/counsel-bot_finetune.gin`. \
documentation: https://github.com/google-research/t5x/blob/main/docs/usage/finetune.md

## Reference models:
DialoGPT: https://github.com/microsoft/DialoGPT
* GODEL: https://github.com/microsoft/GODEL (a predecessor of DialoGPT, aiming for knowledge-grounded goal-oriented dialogue; not sure if it will work well on our case)

T5X: https://github.com/google-research/t5x (an improved implementation of T5, a text-to-text model that is easy to tranfer)

