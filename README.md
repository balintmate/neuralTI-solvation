# Neural TI

This repository contains the implementation of the paper
> [Solvation Free Energies from Neural Thermodynamic Integration](https://arxiv.org/abs/2410.15815) by Bálint Máté, François Fleuret and Tristan Bereau.

## Environment
The ```install.sh``` script will create a virtualenv necessary to run the experiments. This requires python>=3.9 and cuda12.


##  Experiment

The ```run_exp.sh``` activates the virtualenv created by ```install.sh``` and then executes ```experiments/main.py``` using the configs ```experiments/config.yaml``` and ```experiments/LJ3D.yaml```. When executing for the first time, it begins with generating the training data using MCMC. The samples are then dumped to files and loaded in later runs.


## Logging with wandb
All the plots and metrics are logged to the ```experiments/wandb```directory by default. If you create a file at ```experiments/wandb.key``` containing your weights and biases key, then all the logs will be pushed to your wandb account.

## Bib
If you find our paper or this repository useful, consider citing us at

```
@misc{mate2024solvationfreeenergiesneural,
        title={Solvation Free Energies from Neural Thermodynamic Integration}, 
        author={Bálint Máté and François Fleuret and Tristan Bereau},
        year={2024},
        eprint={2410.15815},
        archivePrefix={arXiv},
        primaryClass={cond-mat.stat-mech},
        url={https://arxiv.org/abs/2410.15815}, 
}
```