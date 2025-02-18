source .venv/bin/activate
ps aux|grep wandb|grep -v grep | awk '{print $2}'|xargs kill -9 &> /dev/null
export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
export PYTHONPATH=${PWD}:${PYTHONPATH}
cd src
export PYTHONPATH=${PWD}:${PYTHONPATH}

cd ../experiments


python3 standardTI.py "$@"
