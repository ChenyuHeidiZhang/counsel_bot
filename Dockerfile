FROM python:3.7
RUN --mount=type=bind,target=/home,source=/home/heidi/counsel_bot
RUN python t5x/setup.py install
RUN pip isntall sacrebleu
RUN apt-get update
RUN apt install -y tmux
RUN pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
