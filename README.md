# counsel_bot

### Dataset
CounselChat data prepared for meta learning and in-context learning are under `data/meta_learn`.

CounselChat data prepared for finetuning are under `data/finetune`.

### Models

#### MAML
To run MAML on GPT2: 
```
cd gpt2
python3 maml_higher.py
```
Arguments can be found in the .py file.

#### In Context Learning
We implemented ICL for both GPT2 and GPT3.
Simply run `python3 icl.py` under the desired model directory would run the experiment. See the .py file for more details about the commandline arguments.

#### Finetuning
Under `gpt2`, run `python3 finetune.py` with optional arguments to finetune a GPT2 model on our CounselChat dataset. Run `python3 finetune.py --test` to test the finetuned model.

We finetuned a GPT-3 model using "text-curie-001". The finetuned model is called "curie:ft-personal-2022-11-28-23-41-16". Check `gpt3/finetune.py` for how to run inference on it.

##### [IMPORTANT] Notes on GPT-3
There is an API key with limited credits under `gpt3/run_icl.sh`. This can be used for testing. Before running GPT3, export this API key.


#### Finetuning T5x
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

## References:
T5X: https://github.com/google-research/t5x (an improved implementation of T5, a text-to-text model that is easy to tranfer)

Higher library: https://github.com/facebookresearch/higher

(not used:)
DialoGPT: https://github.com/microsoft/DialoGPT
* GODEL: https://github.com/microsoft/GODEL (a predecessor of DialoGPT, aiming for knowledge-grounded goal-oriented dialogue; not sure if it will work well on our case)


