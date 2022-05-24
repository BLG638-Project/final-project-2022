# final-project-2022
Final Project of Istanbul Technical University BLG-638E (Deep Reinforcement Learning) Lecture

## How to set up

### Simstar

- Install Simstar using the link below
  `LINK HERE`

#### Windows

- Just click on Simstar.exe and Simstar is ready.

#### Linux

- Install Vulkan Libraries
  ```
  sudo apt-get install vulkan-utils
  ```

- Run below commands to make Simstar executable
  - To run Simstar without rendering, use -RenderOffScreen flag
  - To change Simstar's port, use -api_port=XXXX flag
    ```
    cd Simstar
    chmod 777 -R \*
    ./Simstar.sh
    ```

### Python

- Install the requirements
  ```
  pip install -r requirements.txt
  ```
- Install Simstar client from inside PythonAPI using
  ```
  python setup.py install --user
  ```

## Pytorch Version

The final evaluation will be using pytorch version 1.11 and CUDA version 10.2.

## Installation Test

There are multiple stages that needs to be checked.

### 1. Test Simstar Executable

Open the simstar executable, allow for networking if asked.

![opening_screen](PythonAPI/img/opening.png)

### 2. Test Environment Setup

```
cd training_example
python train.py
```