
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

![vimgolf-test-success](https://github.com/user-attachments/assets/011c21d7-5b4b-4836-ac14-e4b8126c3ab4)

![vimgolf-local-4d1a1c36567bac34a9000002-fail](https://github.com/user-attachments/assets/c6f4c2ba-1506-42c1-8d47-28816d338e94)


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
