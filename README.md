
<!-- TODO: create a gym-like environment called "cybergod-gym" which we can remote into other machines and act upon them -->

<!-- TODO: create human labeling environment for vimgolf-gym and cybergod-gym as web application -->

<!-- TODO: create a dedicated cybergod_vimgolf_gym docker image, separate from cybergod_worker_terminal and so on -->

# vimgolf-gym

OpenAI gym like environment and benchmark for Vimgolf.

## Demo

![vimgolf-test-success](https://github.com/user-attachments/assets/011c21d7-5b4b-4836-ac14-e4b8126c3ab4)

<details>

<summary>Code:</summary>

```python

```

</details>


![vimgolf-local-4d1a1c36567bac34a9000002-fail](https://github.com/user-attachments/assets/c6f4c2ba-1506-42c1-8d47-28816d338e94)


<details>

<summary>Code:</summary>

```python

```

</details>


## Installation

```bash
pip install vimgolf-gym
```

If you do have vim installed locally, you can use this docker image:

```bash
# build the image
bash build_docker_image.sh
docker tag cybergod_vimgolf_gym agile4im/cybergod_vimgolf_gym

# or pull the image
docker pull agile4im/cybergod_vimgolf_gym
```

## Usage

```python
import vimgolf_gym

# if you have vim installed locally
env = vimgolf_gym.make("vimgolf-test")

# or run the executor with docker
env = vimgolf_gym.make("vimgolf-test", use_docker=True)

# take an action
env.act("hello world\n")

# take a screenshot and output a PIL image
img = env.screenshot()

# preview screenshot
env.render()

# reset the environment
env.reset()

# check if the environment has at least one success result
if env.success:
   env.get_last_success_result()
```
