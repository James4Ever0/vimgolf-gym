# standalone script for creating a benchmark with litellm, using public vimgolf datasets
# currently only uses text terminal buffer dump
# may add extra dependencies to setup.py later, as [benchmark]
# shall use a fixed version of litellm

# TODO: type keys char by char, in the observation process, the agent can summarize, change future keys, according to the feedback

import litellm
import vimgolf_gym
import vimgolf_gym.dataclasses
import argparse
import time
from pathlib import Path
import json
import os


class LLM:
    def __init__(self, model: str):
        self.model = model

    def completion(self, messages: list[dict[str, str]]):
        # messages: [{"content": ..., "role": ...}]
        response = litellm.completion(self.model, messages=messages, stream=False)
        return response


def run_single_shot(
    llm: LLM,
    custom_challenge: vimgolf_gym.dataclasses.VimGolfCustomChallenge,
):
    details = custom_challenge.description
    input = custom_challenge.input
    output = custom_challenge.output
    prompt = f"""
Vimgolf is a game where you try to transform text using the fewest number of keystrokes in Vim.

Your task is to solve the following Vimgolf challenge with details:
  
Details:
  
{details}

The input file wrapped in triple backticks:
  
```
{input}
```

The output file wrapped in triple backticks:
  
```
{output}
```

Your keystokes must be less than the length of output file. Do not naively copy and paste the output file. You must use Vim commands to transform the input file into the output file.

Here are some example solutions, for format demostration:

iHello World<Esc>:wq<NL>

:%s/abcdef/defabc/g<NL>:wq<NL>

Your last line of response will be treated as solution. Do not wrap the solution around any marker (like triple backticks), just write it in plain style.
"""
    response = llm.completion([{"role": "user", "content": prompt}])
    # return a string
    response_content: str = response.chunks[0].content.strip()
    if response_content:
        # retrieve last line
        lines = response_content.splitlines()
        lines = [it.strip() for it in lines if it.strip()]
        solution = lines[-1]
        return solution
    else:
        return ""


def run_multi_turn(
    llm: LLM, custom_challenge: vimgolf_gym.dataclasses.VimGolfCustomChallenge
):
    raise NotImplementedError("Multi turn benchmark not implemented yet")
    verification_failed_keys = []

    with vimgolf_gym.make("vimgolf-custom", custom_challenge=custom_challenge) as env:
        action = ...  # shall be utf8 string with ansi control sequences
        env.act(action)
        env.log_file  # str, vimgolf log file path, can be copied
        env.results
        env.success_results
        env.success  # bool
        env.executor.terminal.vt_screen.display  # string terminal dump
        keys = ...
        env.verify_keys(keys)


def build_vimgolf_public_task(task_path: Path):

    challenge_definition_path = task_path / "challenge.json"
    challenge_metadata_path = task_path / "metadata.json"

    challenge_definition = json.loads(challenge_definition_path.read_text())
    challenge_metadata = json.loads(challenge_metadata_path.read_text())

    input_content = challenge_definition["in"]["data"]
    output_content = challenge_definition["out"]["data"]
    title = challenge_metadata["title"]
    detail = challenge_metadata["detail"]

    custom_challenge = vimgolf_gym.dataclasses.VimGolfCustomChallenge(
        input=input_content,
        output=output_content,
        name=title,
        description=detail,
        solution=None,
    )
    return custom_challenge


def build_vimgolf_custom_task(task_path: Path):
    challenge_definition_path = task_path / "challenge.json"
    challenge_definition = json.loads(challenge_definition_path.read_text())
    custom_challenge = vimgolf_gym.dataclasses.VimGolfCustomChallenge.parse_obj(
        challenge_definition
    )
    return custom_challenge


class BenchmarkRunner:
    def __init__(
        self,
        llm: LLM,
        dataset_dir: Path,
        dataset_format: str,
        log_basedir: Path,
        task_timeout: int,
        runner: str,
    ):
        """
        Constructor for BenchmarkRunner.

        Args:
            llm (litellm.LiteLLM): The LiteLLM model to use.
            dataset_dir (pathlib.Path): The directory of the vimgolf dataset.
            log_basedir (pathlib.Path): The base directory of the log directory.
            task_timeout (int): The timeout in seconds to run a task.
            runner (str): The runner to use.
        """
        self.task_timeout = task_timeout
        self.llm = llm
        self.dataset_dir = dataset_dir
        self.dataset_format = dataset_format
        self.log_basedir = log_basedir
        self.timestamp = time.time()
        self.runner = runner
        self.cli_args = f"dataset_dir={dataset_dir}, log_basedir={log_basedir}"

    def run_task(self, task_id: str):
        # create a new directory in the log directory, named with timestamp and cli args
        log_dir: Path = self.log_basedir.resolve() / task_id
        log_dir.mkdir(parents=True, exist_ok=True)

        task_path = self.dataset_dir / task_id

        if self.dataset_format == "vimgolf_public":
            custom_challenge = build_vimgolf_public_task(task_path)
        elif self.dataset_format == "vimgolf_custom":
            custom_challenge = build_vimgolf_custom_task(task_path)
        else:
            raise ValueError(f"Unknown dataset format: {self.dataset_format}")
        start_time = time.time()
        end_time = start_time + self.task_timeout

        llm = self.llm

        # TODO: set a hard timeout for the task, running the task in a separate thread
        if self.runner == "single_shot":
            solution = run_single_shot(llm=llm, custom_challenge=custom_challenge)
        elif self.runner == "multi_turn":
            solution = run_multi_turn(llm=llm, custom_challenge=custom_challenge)
        else:
            raise ValueError(f"Unknown runner: {self.runner}")
        elapsed_time = time.time() - start_time
        ret = dict(
            task_id=task_id,
            start_time=start_time,
            elapsed_time=elapsed_time,
            solution=solution,
            input_content=custom_challenge.input,
            output_content=custom_challenge.output,
        )
        return ret

    def run_all(self, milestone: int = 0):
        task_id_list = os.listdir(self.dataset_dir)
        task_id_list.sort()
        for index, task_id in enumerate(task_id_list):
            if index < milestone:
                print("Skipping task %s before milestone %s" % (task_id, milestone))
                continue
            print("Running task %s" % task_id)
            task_result = self.run_task(task_id)
            yield task_result


def main():
    # parse args: dataset path, model name, log path
    # store logs in a subdirectory of the log path, named with timestamp, cli args
    # create a new directory in the subdirectory with task_id
    # logs are: terminal screenshots, success flag, start time, end time, model name, key events and timestamps, llm conversation history, game state history, best keys, best score
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-name", type=str, required=True, help="name of vimgolf dataset"
    )
    parser.add_argument(
        "--dataset-format",
        required=True,
        help="format of thr dataset, can be vimgolf_public, vimgolf_custom",
    )
    parser.add_argument(
        "--dataset-dir", type=Path, required=True, help="path to vimgolf dataset"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="name of litellm model"
    )
    parser.add_argument(
        "--log-dir", type=Path, default=Path("./logs"), help="path to log directory"
    )
    parser.add_argument(
        "--task-timeout", type=int, default=840, help="timeout for each task in seconds"
    )
    parser.add_argument(
        "--runner", type=str, default="single_shot", help="runner to use"
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        help="path to output jsonl file for saving solutions",
        required=True,
    )
    parser.add_argument(
        "--milestone", type=int, default=0, help="milestone for skipping tasks"
    )

    args = parser.parse_args()

    # create a new directory in the log directory, named with timestamp and cli args
    log_dir: Path = (
        args.log_dir.resolve()
        + "/"
        + time.strftime("%Y-%m-%d-%H-%M-%S")
        + "-"
        + args.model
        + "-"
        + args.dataset_name
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    llm = LLM(args.model)

    runner = BenchmarkRunner(
        llm=llm,
        dataset_dir=args.dataset_dir.resolve(),
        dataset_format=args.dataset_format,
        log_basedir=log_dir,
        task_timeout=args.task_timeout,
        runner=args.runner,
    )
    results = runner.run_all(milestone=args.milestone)
    with open(args.output_jsonl, "a+") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print("Output saved to", args.output_jsonl)


if __name__ == "__main__":
    main()
