# standalone script for creating a benchmark with litellm, using public vimgolf datasets
# currently only uses text terminal buffer dump
# may add extra dependencies to setup.py later, as [benchmark]
# shall use a fixed version of litellm

# TODO: type keys char by char, in the observation process, the agent can summarize, change future keys, according to the feedback

# TODO: write stdout, stderr to a file using tee, at top of the run log directory
# TODO: write run task result to corresponding log folder
import asyncio
import litellm
import vimgolf_gym
import vimgolf_gym.dataclasses
import argparse
import time
from pathlib import Path
import json
import os
import sys
import copy
import atexit
import typing


async def run_challenge(
    runner: str,
    llm: "LLM",
    custom_challenge: vimgolf_gym.dataclasses.VimGolfCustomChallenge,
):
    if runner == "single_shot":
        return await run_single_shot(llm=llm, custom_challenge=custom_challenge)
    elif runner == "multi_turn":
        return await run_multi_turn(llm=llm, custom_challenge=custom_challenge)
    else:
        raise ValueError(f"Unknown runner: {runner}")


# TODO: ability to change the filename on the fly
class TeeLogger:
    def __init__(self, filename, stream: typing.TextIO):
        self.file = open(filename, "a")
        atexit.register(self.file.close)
        self.stream = stream

    def write(self, message):
        self.stream.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()


def redirect_stdout_stderr(log_file: str):
    # Redirect both stdout and stderr
    assert sys.stdout
    assert sys.stderr
    sys.stdout = TeeLogger(log_file, sys.stdout)
    sys.stderr = TeeLogger(log_file, sys.stderr)


class LLM:
    def __init__(self, model: str):
        self.model = model
        self.history = []

    def dump_history(self, clear: bool):
        ret = copy.deepcopy(self.history)
        if clear:
            self.history = []
        return ret

    async def acompletion(self, messages: list[dict[str, str]]):
        # messages: [{"content": ..., "role": ...}]
        self.history.append(dict(type="messages", data=messages))
        print("LLM repsonse:")
        response = await litellm.acompletion(self.model, messages=messages, stream=True)
        full_response = []
        thinking = False

        async for chunk in response:  # type: ignore
            delta = chunk.choices[0]["delta"]  # type: ignore
            if delta.get("reasoning_content") is not None:
                chunk_content = delta["reasoning_content"]
                if not thinking:
                    print("\nThinking...")
                    full_response.append("<thinking>\n")
                    thinking = True
            elif delta.get("content") is not None:
                chunk_content = delta["content"]
                if thinking:
                    print("\nDone thinking.")
                    full_response.append("</thinking>\n")
                    thinking = False
            print(chunk_content, sep="", end="", flush=True)
            if chunk_content:
                full_response.append(chunk_content)
        print("\nLLM response complete.")
        ret = "".join(full_response)
        self.history.append(dict(type="response", data=ret))
        return ret


async def run_single_shot(
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

Here are some example solutions, for format demostration (all solutions shall be in one line):

iHello World<Esc>:wq<NL>

:%s/abcdef/defabc/g<NL>:wq<NL>

Your last line of response will be treated as solution. Do not wrap the solution around any marker (like triple backticks), just write it in plain style. Do not write it in multiline style. Do not write any comment or explanation. Do not write any other text. Just write the solution. If your solution contains multiple steps, you will concatenate these steps into one line, optionally using <NL> as separator, depending on the situation.

Example response:

I think the following solution is optimal:

iHello World<Esc>:s/World/Earth/g<NL>:wq<NL>

Please write your solution according to the rules and the example response:
"""
    print("Prompt:")
    print(prompt)
    response_content = await llm.acompletion([{"role": "system", "content": prompt}])
    if response_content:
        # retrieve last line
        solution = get_last_non_empty_line(response_content)
        return solution
    else:
        return ""

def get_last_non_empty_line(content:str):
    lines = content.splitlines()
    lines = [it.strip() for it in lines if it.strip()]
    if lines:
        return lines[-1]
    else:
        return ""

async def run_multi_turn(
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
        dataset_name: str,
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
            dataset_name (str): The dataset name.
            log_basedir (pathlib.Path): The base directory of the log directory.
            task_timeout (int): The timeout in seconds to run a task.
            runner (str): The runner to use.
        """
        self.task_timeout = task_timeout
        self.llm = llm
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.dataset_format = dataset_format
        self.log_basedir = log_basedir
        self.timestamp = time.time()
        self.runner = runner
        self.cli_args = f"dataset_name={dataset_name}, task_timeout={task_timeout}, dataset_format={dataset_format}, timestamp={self.timestamp}, model={llm.model}"
        metadata_path = self.log_basedir.resolve() / "metadata.json"
        metadata = dict(
            cli_args=dict(
                dataset_name=dataset_name,
                task_timeout=task_timeout,
                dataset_format=dataset_format,
                timestamp=self.timestamp,
                model=llm.model,
                runner=runner,
            ),
            task_count=len(os.listdir(self.dataset_dir)),
            task_id_list=os.listdir(self.dataset_dir),
        )
        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=4,
            )
        )
        self.run_log_path = self.log_basedir.resolve() / "runner.log"
        self.run_log_path.write_text(self.cli_args + "\n")
        redirect_stdout_stderr(str(self.run_log_path))

    async def run_task(self, task_id: str):
        # create a new directory in the log directory, named with timestamp and cli args
        start_time = time.time()
        trial_timestamp = time.strftime(
            r"%Y-%m-%d-%H-%M-%S", time.localtime(start_time)
        )
        trial_name = f"{task_id}-{self.dataset_name}-{trial_timestamp}"
        log_dir: Path = self.log_basedir.resolve() / task_id / trial_name
        log_dir.mkdir(parents=True, exist_ok=True)

        result_path = log_dir / "result.json"
        llm_history_path = log_dir / "llm_history.json"

        task_path = self.dataset_dir / task_id

        if self.dataset_format == "vimgolf_public":
            custom_challenge = build_vimgolf_public_task(task_path)
        elif self.dataset_format == "vimgolf_custom":
            custom_challenge = build_vimgolf_custom_task(task_path)
        else:
            raise ValueError(f"Unknown dataset format: {self.dataset_format}")
        llm = self.llm
        solution = ""
        status = "unknown"

        # TODO: set a hard timeout for the task, running the task in a separate thread
        task = asyncio.create_task(
            run_challenge(
                runner=self.runner, llm=llm, custom_challenge=custom_challenge
            )
        )
        try:
            solution = await asyncio.wait_for(task, timeout=self.task_timeout)
            status = "success"
        except asyncio.TimeoutError:
            print(f"Task {task_id} timed out after {self.task_timeout} seconds")
            solution = ""
            status = "timeout"
            was_cancelled = task.cancel()
            print("Task cancelled:", was_cancelled)

        end_time = time.time()
        elapsed_time = end_time - start_time
        ret = dict(
            task_id=task_id,
            dataset_name=self.dataset_name,
            status=status,
            trial_name=trial_name,
            start_time=start_time,
            end_time=end_time,
            elapsed_time=elapsed_time,
            solution=solution,
            input_content=custom_challenge.input,
            output_content=custom_challenge.output,
        )
        result_path.write_text(json.dumps(ret, indent=4))
        llm_history = llm.dump_history(clear=True)
        llm_history_path.write_text(json.dumps(llm_history, indent=4))
        return ret

    async def run_all(self, milestone: int = 0):
        task_id_list = os.listdir(self.dataset_dir)
        task_id_list.sort()
        for index, task_id in enumerate(task_id_list):
            if index < milestone:
                print("Skipping task %s before milestone %s" % (task_id, milestone))
                continue
            print("Running task %s" % task_id)
            # TODO: log prompt, log task result, log llm response
            task_result = await self.run_task(task_id)
            print("Task %s complete" % task_id)
            print("Task result:", task_result)
            yield task_result


async def main():
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
    parser.add_argument("--max-tasks", type=int, default=0, help="max tasks to run")

    args = parser.parse_args()

    # create a new directory in the log directory, named with timestamp and dataset name
    log_dir: Path = args.log_dir.resolve() / (
        time.strftime(r"%Y-%m-%d-%H-%M-%S", time.localtime()) + "-" + args.dataset_name
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    llm = LLM(args.model)

    runner = BenchmarkRunner(
        llm=llm,
        dataset_dir=args.dataset_dir.resolve(),
        dataset_name=args.dataset_name,
        dataset_format=args.dataset_format,
        log_basedir=log_dir,
        task_timeout=args.task_timeout,
        runner=args.runner,
    )
    results = runner.run_all(milestone=args.milestone)
    max_tasks = args.max_tasks if args.max_tasks > 0 else float("inf")
    processed_tasks = 0

    with open(args.output_jsonl, "a+") as f:
        async for result in results:
            f.write(json.dumps(result) + "\n")
            f.flush()
            processed_tasks += 1
            if processed_tasks >= max_tasks:
                break
    print("Output saved to", args.output_jsonl)


if __name__ == "__main__":
    asyncio.run(main())
