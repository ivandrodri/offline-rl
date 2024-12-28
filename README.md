# Offline RL Laboratory

Welcome to the Offline RL Laboratory repository! This project serves as a playground for testing new ideas and 
experimenting with offline Reinforcement Learning (RL) and Imitation Learning (IL) in a simple and streamlined way. 
It leverages the [Tianshou](https://github.com/thu-ml/tianshou) RL framework and 
[Minari](https://github.com/Farama-Foundation/Minari), which is increasingly becoming the standard for offline RL 
dataset standardization and manipulation.

> **Note**: This project is a **work in progress (WIP)**. Contributions, suggestions, and feedback are highly encouraged and welcome! 🎉

### Getting Started

**High-Level Overview**: For a conceptual introduction to Offline RL and Minari Datasets, see 
[notebooks/nb_0_IntroOfflineRL.ipynb](notebooks/nb_0_IntroOfflineRL.ipynb) and 
[notebooks/nb_1_offline_RL_dataset_libraries.ipynb](notebooks/nb_1_offline_RL_dataset_libraries.ipynb).

**Technical Details**: For a deeper dive into Offline RL theory, check out [notebooks/nb_3_offline_RL_theory.ipynb](notebooks/nb_3_offline_RL_theory.ipynb).

**Hands-On Examples**: Explore practical examples in the other notebooks located in the notebooks/ folder.

### How to use this repository

There are multiple ways of viewing/executing the content. 

1. If you just want to view the rendered notebooks, 
   open `nb_html/index.html` in your browser.

2. If you want to execute the notebooks, you will either need to
   install the dependencies or use docker.
   For running without docker, create a [poetry](https://python-poetry.org/) environment (with python 3.11),
   e.g., with `poetry shell`.

   Then, install the dependencies and the package with

   ```shell
   python poetry_install.py
   bash build_scripts/install_presentation_requirements.sh
   ```

3. If you want to use docker instead,
   you can build the image locally using:
    
   ```shell
   docker build -t offline-rl:local .
   ```

   You can then start the container e.g., with
    
   ```shell
   docker run -it -p 8888:8888 offline-rl:local jupyter notebook --ip=0.0.0.0
   ```

4. Finally, for creating source code documentation, you can run
    
   ```shell
   bash build_scripts/build_docs.sh
   ```

   and then open `docs/build/html/index.html` in your browser.
   This will also rebuild the jupyter-book based notebook documentation
   that was originally found in the `html` directory.

5. Pre-commit Hooks Setup

This project uses [pre-commit](https://pre-commit.com/) to automatically run checks and formatters on the code before 
each commit. These hooks help ensure code quality and consistency. Make sure you have `pre-commit` installed.

```bash
pip install pre-commit
```

Finally:

```bash
pre-commit install
pre-commit run --all-files
```