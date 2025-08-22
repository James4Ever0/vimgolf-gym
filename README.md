
<!-- TODO: create a gym-like environment called "cybergod-gym" which we can remote into other machines and act upon them -->

<!-- TODO: create human labeling environment for vimgolf-gym and cybergod-gym as web application -->

<!-- TODO: create a dedicated cybergod_vimgolf_gym docker image, separate from cybergod_worker_terminal and so on -->

# vimgolf-gym

OpenAI gym like environment and benchmark for Vimgolf.

## Installation

```bash
pip install vimgolf-gym
```

## Demo

## Usage

```python
import vimgolf_gym

env = vimgolf_gym.make("vimgolf-test")
env.act("hello world\n")
img = env.screenshot() # output a PIL image
env.render() # preview screenshot
env.reset()

if env.success:
   env.get_last_success_result
```
