
<!-- TODO: create a gym-like environment called "cybergod-gym" which we can remote into other machines and act upon them -->

<!-- TODO: create human labeling environment for vimgolf-gym and cybergod-gym as web application -->

<!-- TODO: create a dedicated cybergod_vimgolf_gym docker image, separate from cybergod_worker_terminal and so on -->

<!-- TODO: use worst human submission to calculate relative score -->
<!-- TODO: calculate overall score for multiple challenges as benchmark result -->

<!-- formula: estimated_worst_solution_score = human_worst_solution_score ? (string_length_of_output + 10) -->
<!-- relative_inverse_score = estimated_worst_solution_score / agent_score (the higher the better)  -->

<div>
<p align="center"><h1 align="center">vimgolf-gym</h1></p>
<p align="center">OpenAI gym like, customizable environment and benchmark for Vimgolf.</p>
<p align="center">
<a href="https://github.com/james4ever0/vimgolf-gym/blob/main/LICENSE"><img alt="License: UNLICENSE"
 src="https://img.shields.io/badge/license-UNLICENSE-green.svg?style=flat"></a>
<a href="https://pypi.org/project/vimgolf-gym/"><img alt="PyPI" src="https://img.shields.io/pypi/v/vimgolf-gym"></a>
<a href="https://james4ever0.github.io/vimgolf-gym/"><img src="https://img.shields.io/badge/API-Docs-blueviolet" alt="API documentation"></a>
<a href="https://pepy.tech/projects/vimgolf-gym"><img src="https://static.pepy.tech/badge/vimgolf-gym" alt="PyPI Downloads"></a>
<a href="https://github.com/james4ever0/vimgolf-gym"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://huggingface.co/datasets/James4Ever0/vimgolf_challenges_and_solutions"><img alt="Vimgolf public challenge dataset" src="https://img.shields.io/badge/🤗-HuggingFace-blue"></a>
<a href="https://hub.docker.com/r/agile4im/cybergod_vimgolf_gym"><img alt="Docker image: agile4im/cybergod_vimgolf_gym" src="https://img.shields.io/badge/dockerhub-gray?logo=docker"></a>
<a href="https://www.benchflow.ai/benchmarks/James4ever0/Vimgolf"><img alt="Benchflow" src="https://img.shields.io/badge/Benchflow-blueviolet"></a>
<a href="https://vimgolf.netlify.com"><img alt="Netlify Status" src="https://api.netlify.com/api/v1/badges/ac58a477-2f8b-42cd-b3b5-5e3ebeb8fa78/deploy-status"></a>
</p>
</div>

## Demo

### A simple script for solving the "vimgolf-test" challenge

![vimgolf-test-success](https://github.com/user-attachments/assets/011c21d7-5b4b-4836-ac14-e4b8126c3ab4)

Console output:

```
Success: True
Results:
[VimGolfEnvResult(correct=True, keys='ihello world<NL>hello world<Esc>:wq<NL>', score=29)]
```

<details>

<summary>Reproduction code</summary>

```python
import vimgolf_gym
import time
import PIL.Image

def test_demo():
    """
    Run a demo of vimgolf-gym, interacting with the environment by
    typing "hello world" into the buffer and then saving and quitting vim.
    Takes screenshots of the process and saves them to a .gif file.
    """
    env = vimgolf_gym.make("vimgolf-test")
    images: list[PIL.Image.Image] = []
    images.append(env.screenshot())
    env.act("i")
    images.append(env.screenshot())
    env.act("hello world\n")
    images.append(env.screenshot())
    env.act("hello world")
    images.append(env.screenshot())
    env.act("\x1b:wq")
    images.append(env.screenshot())
    env.act("\n")
    images.append(env.screenshot())
    time.sleep(1)
    images.append(env.screenshot())
    print("Success:", env.success)
    print("Results:")
    try:
         import rich
         rich.print(env.results)
    except ImportError:
         print(env.results)
    env.close()
    write_images_to_gif(images=images, output_gif_path="vimgolf-test-success.gif")


def write_images_to_gif(
    images: list[PIL.Image.Image], output_gif_path: str, interval=1000
):
    durations = [interval] * len(images)

    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=1,
    )

if __name__ == "__main__":
    test_demo()
```

</details>

### A trial on the "vimgolf-local-4d1a1c36567bac34a9000002" challenge

![vimgolf-local-4d1a1c36567bac34a9000002-fail](https://github.com/user-attachments/assets/c6f4c2ba-1506-42c1-8d47-28816d338e94)

Console output:

```
Success: False
Results:
[VimGolfEnvResult(correct=False, keys=':wq<NL>', score=4)]
```

<details>

<summary>Reproduction code</summary>

```python
import vimgolf_gym
import time
import PIL.Image

def write_images_to_gif(
    images: list[PIL.Image.Image], output_gif_path: str, interval=1000
):
    durations = [interval] * len(images)

    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=1,
    )

def test_local():
    """
    Test a local challenge with the given challenge id.

    It checks the data of the challenge in the local dataset, and then runs the
    challenge in the local environment and takes screenshots of the process.
    """
    challenge_id = "4d1a1c36567bac34a9000002"
    assert challenge_id in vimgolf_gym.list_local_challenge_ids()
    assert (
        vimgolf_gym.get_local_challenge_definition(challenge_id).client_version
        == "0.5.0"
    )
    assert (
        vimgolf_gym.get_local_challenge_metadata(challenge_id).challenge_hash
        == challenge_id
    )
    assert vimgolf_gym.get_local_challenge_worst_solution(challenge_id).rank == "74"
    assert (
        vimgolf_gym.get_local_challenge_worst_solution_header(challenge_id).score
        == "206"
    )
    env = vimgolf_gym.make("vimgolf-local-%s" % challenge_id)
    images: list[PIL.Image.Image] = []
    images.append(env.screenshot())
    env.act(":wq")
    images.append(env.screenshot())
    env.act("\n")
    images.append(env.screenshot())
    time.sleep(1)
    images.append(env.screenshot())
    print("Success:", env.success)
    print("Results:")
    try:
         import rich
         rich.print(env.results)
    except ImportError:
         print(env.results)
    env.close()
    write_images_to_gif(
        images=images, output_gif_path="vimgolf-local-%s-fail.gif" % challenge_id
    )

if __name__ == "__main__":
   test_local()
```

</details>


## Installation

```bash
# install from pypi
pip install vimgolf-gym

# or install the latest version from github
pip install git+https://github.com/james4ever0/vimgolf-gym.git
```

If you do not have Vim installed locally, or want an extra layer of isolation, you can use this docker image:

```bash
# build the image
bash build_docker_image.sh
docker tag cybergod_vimgolf_gym agile4im/cybergod_vimgolf_gym

# or pull the image
docker pull agile4im/cybergod_vimgolf_gym
```

## Usage

Before everything else, remember to manually sanitize the string before passing into the VimGolf challenge environment, to prevent misjudgement.

```python
import vimgolf_gym

unsanitized_string = "hello world\r\nhello again"

# dos2unix and add newline to end if missing
# output: "hello world\nhello again\n"
sanitized_string = vimgolf_gym.format_vimgolf_string(unsanitized_string)
```

Basic interactions:

```python
import vimgolf_gym
import vimgolf_gym.dataclasses

# a basic "hello world" challenge
env_name = "vimgolf-test"

# a local challenge, format is "vimgolf-local-<challenge_id>"
env_name = "vimgolf-local-4d1a1c36567bac34a9000002"

# an online challenge, format is "vimgolf-online-<challenge_id>"
env_name = "vimgolf-online-4d1a1c36567bac34a9000002"

# if you have vim installed locally
env = vimgolf_gym.make(env_name)

# or run the executor with docker
env = vimgolf_gym.make(env_name, use_docker=True)

# if you want to customize the challenge
env = vimgolf_gym.make("vimgolf-custom", custom_challenge = vimgolf_gym.dataclasses.VimGolfCustomChallenge(input="", output="hello world\n"))

# if you want to read the buffer of the editor (and avoid cheating)
env = vimgolf_gym.make(env_name, log_buffer=True)

# retrieve the editor buffer to track progress
buffer = env.buffer

# reset the env
env.reset()

# close the env
env.close()

# verify a solution by its keys, in vimgolf style
success = env.verify_keys("ihello world<NL>hello world<Esc>:wq<NL>")

# calculate relative inverse score directly
relative_inverse_score = env.calculate_relative_inverse_score(score=100)

# or if you have a known worst score
relative_inverse_score = env.calculate_relative_inverse_score(score=100, worst_score=200)

# if you want to close the environment automatically
with vimgolf_gym.make(env_name) as env:
    # take an action, in raw string
    env.act("hello world\n")

    # take a screenshot and output a PIL image
    img = env.screenshot()

    # preview screenshot
    env.render()

    # reset the environment
    env.reset()

    # check if the environment has at least one success result
    if env.success:
        # VimGolfEnvResult: (correct: bool, keys: str, score: int)
        result: vimgolf_gym.dataclasses.VimGolfEnvResult = env.get_last_success_result()
```

An example custom challenge in yaml:

```yaml
input: |
    The second line
    The first line
output: |
    The first line
    The second line
name: Swap lines
description: Swap the first and second lines of the input
solution: null
```

You can load the challenge with:

```python
import yaml
import vimgolf_gym.dataclasses

input_file = "<path to your challenge file>"
with open(input_file, "r") as f:
    yaml_string = f.read()
yaml_obj = yaml.safe_load(yaml_string)
custom_challenge = vimgolf_gym.dataclasses.VimGolfCustomChallenge.parse_obj(yaml_obj)
```

The local challenges are stored in `~/.cache/cybergod-vimgolf-challenges/`.

If you want to learn more about the local challenges, use the following code:

```python
import vimgolf_gym
import vimgolf_gym.dataclasses

challenge_id = "4d1a1c36567bac34a9000002"

# list all local challenge ids
local_challenge_ids: list[str] = vimgolf_gym.list_local_challenge_ids()

# get the challenge definition
# VimGolfChallengeDefinition: (input: InputOutputModel, output: InputOutputModel, client_version: str)
# InputOutputModel: (data: str, type: str)
challenge: vimgolf_gym.dataclasses.VimGolfChallengeDefinition = get_local_challenge_definition(challenge_id)

# get the challenge metadata
# VimGolfChallengeMetadata: (href: str, title: str, detail: str, challenge_hash: str)
metadata: vimgolf_gym.dataclasses.VimGolfChallengeMetadata = vimgolf_gym.get_local_challenge_metadata(challenge_id)

# get the worst solution
# VimGolfPublicSolution: (rank: str, solution: str, header: str)
solution: vimgolf_gym.dataclasses.VimGolfPublicSolution = vimgolf_gym.get_local_challenge_worst_solution(challenge_id)

# get the worst solution header
# VimGolfParsedPublicSolutionHeader: (rank: str, score: str, user_name: str, user_id: str, data: datetime)
header: vimgolf_gym.dataclasses.VimGolfParsedPublicSolutionHeader = vimgolf_gym.get_local_challenge_worst_solution_header(challenge_id)
```

If you want to obtain online challenge ids, you have a few options:

1. Visit the [Vimgolf website](https://vimgolf.com) and look for the challenge ids.
2. Use `vimgolf` command
   - Install: `pip3 install vimgolf`
   - Run: `vimgolf list`

## Benchmark

We provide a simple benchmark script `vimgolf-benchmark.py` that can be used to benchmark the performance of a model in batch. The script produces a JSONL file which can be used for evaluation.

Before running the script, you need to install dependencies:

```bash
pip3 install litellm==1.76.2 vimgolf_gym==0.1.1
```

To use Ollama for inference, you can run the following command:

```bash

export OLLAMA_API_BASE=<your_ollama_api_base>
MODEL="ollama/<model_id>"

JSONL_OUTPUT="<benchmark_output>.jsonl"
DATASET_NAME="<dataset_name>"
DATASET_DIR=$(realpath "<dataset_dir>")
DATASET_FORMAT="<dataset_format>"
RUNNER="single_shot"

python3 vimgolf-benchmark.py \
--output-jsonl $JSONL_OUTPUT \
--model $MODEL \
--dataset-name $DATASET_NAME \
--dataset-dir $DATASET_DIR \
--dataset-format $DATASET_FORMAT \
--runner $RUNNER
```

If you want to evaluate the results, you need to provide `--solution-format vimgolf-benchmark` to the evaluator script.

## Evaluation

We provide a simple evaluation script `vimgolf-evaluate.py` that can be used to evaluate the performance in batch. The script takes a JSONL file and the format name of the file as input and returns the overall score of the batch.

Before running the script, you need to install dependencies:

```bash
pip3 install vimgolf-gym==0.1.1
```

To evaluate the output from [terminal-bench adaptor](https://github.com/James4Ever0/terminal-bench/tree/vimgolf-comparison) for a parity experiment, you can use the following command:

```bash
python3 -u vimgolf-evaluate.py \
--solution-format terminal-bench-adaptor \
--jsonl-file "<solution-result>.jsonl" \
--validator vimgolf-validator \
--solution-not-longer-than-output \
--result-savepath "<eval-result>.json" \
2>&1 | tee -a "<evaluate-result>.log"
```

## License

The Unlicense
