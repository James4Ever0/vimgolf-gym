import vimgolf_gym
import vimgolf_gym.dataclasses
import vimgolf_gym.lib
import json
import argparse
import pydantic
import hashlib
import shlex
import subprocess
import shutil
import os


class TerminalBenchAdaptorSolution(pydantic.BaseModel):
    task_id: str
    trial_name: str
    challenge_hash: str
    solution: str


class VimGolfBenchmarkSolution(pydantic.BaseModel):
    task_id: str
    start_time: float
    elapsed_time: float
    solution: str
    input_content: str
    output_content: str


def prepare_input(
    solution_format: str, solution
) -> vimgolf_gym.dataclasses.VimGolfCustomChallenge:
    if solution_format == "terminal-bench-adaptor":
        solution = TerminalBenchAdaptorSolution.parse_obj(solution)
        challenge_definition = vimgolf_gym.lib.get_local_challenge_definition(
            solution.challenge_hash
        )
        challenge_input = challenge_definition.input.data
        challenge_output = challenge_definition.output.data
        solution.challenge_hash
        return vimgolf_gym.dataclasses.VimGolfCustomChallenge(
            input=challenge_input, output=challenge_output, solution=solution.solution
        )
    elif solution_format == "vimgolf-benchmark":
        solution = VimGolfBenchmarkSolution.parse_obj(solution)
        return vimgolf_gym.dataclasses.VimGolfCustomChallenge(
            input=solution.input_content,
            output=solution.output_content,
            solution=solution.solution,
        )
    else:
        raise ValueError(f"Unknown solution format: {solution_format}")


def run_vimgolf_local(custom_challenge: vimgolf_gym.dataclasses.VimGolfCustomChallenge):
    validated = False
    with vimgolf_gym.make(
        "vimgolf-custom",
        custom_challenge=custom_challenge,
    ) as env:
        if custom_challenge.solution:
            validated = env.verify_keys(custom_challenge.solution)
    return validated


def run_vimgolf_docker(
    custom_challenge: vimgolf_gym.dataclasses.VimGolfCustomChallenge,
):
    validated = False
    with vimgolf_gym.make(
        "vimgolf-custom", custom_challenge=custom_challenge, use_docker=True
    ) as env:
        if custom_challenge.solution:
            validated = env.verify_keys(custom_challenge.solution)
    return validated


def sha256_checksum(content: str) -> str:
    sha256 = hashlib.sha256()
    sha256.update(content.encode("utf-8"))
    return sha256.hexdigest()


def assert_docker_privilege():
    assert shutil.which("docker"), "Docker not found in PATH"
    # assert user is in docker group or has permission to run docker without sudo
    assert (
        os.geteuid() == 0
        or subprocess.run(["groups"], capture_output=True, text=True).stdout.find(
            "docker"
        )
        != -1
    ), "User does not have permission to run Docker commands"


def run_vimgolf_validator(
    custom_challenge: vimgolf_gym.dataclasses.VimGolfCustomChallenge,
):
    validated = False
    input_content = custom_challenge.input
    output_content = custom_challenge.output
    solution_keys = custom_challenge.solution

    cmd = shlex.split(
        "docker run --rm --network=none agile4im/vimgolf-verifier:v0.0.2 python /app/vimgolf-verifier.py single_shot"
    ) + ["--input_content", input_content, "--solution_keys", solution_keys]
    try:
        output = subprocess.check_output(cmd, timeout=15.0) # type: ignore
        output = json.loads(output)
        checksum_server = output["checksum"]
        checksum_output = sha256_checksum(output_content)
        validated = checksum_server == checksum_output
    except subprocess.CalledProcessError:
        pass
    except subprocess.TimeoutExpired:
        pass
    except json.JSONDecodeError:
        pass
    except KeyError:
        pass
    return validated


class Evaluator:
    def __init__(self, solution_format: str, jsonl_file: str, validator: str):
        self.solution_format = solution_format
        self.jsonl_file = jsonl_file
        self.validator = validator

        if self.validator == "vimgolf-validator":
            assert_docker_privilege()

    def evaluate(self) -> list[bool]:
        results = []
        working_solutions = []
        challenges = []
        with open(self.jsonl_file, "r") as f:
            for line in f:
                solution = json.loads(line)
                # use solution_format to parse the solution
                custom_challenge = prepare_input(self.solution_format, solution)
                challenges.append(custom_challenge)
        print("Evaluating", len(challenges), "solutions")
        for index, custom_challenge in enumerate(challenges):
            print("Processing item (%s/%s)" % (index + 1, len(challenges)))
            print("Checking solution:", repr(custom_challenge.solution))
            # evaluate the solution
            validated = False

            if custom_challenge.solution:
                if self.validator == "vimgolf-local":
                    validated = run_vimgolf_local(custom_challenge)
                elif self.validator == "vimgolf-docker":
                    validated = run_vimgolf_docker(custom_challenge)
                elif self.validator == "vimgolf-validator":
                    validated = run_vimgolf_validator(custom_challenge)
                else:
                    raise ValueError("Unknown validator:", self.validator)
            else:
                pass
            # retrieve the evaluation result
            results.append(validated)
            if validated:
                working_solutions.append(custom_challenge.solution)
            print(
                "Result for item (%s/%s) is %s"
                % (index + 1, len(challenges), validated)
            )
        print("Evaluation finished")
        print("Working solutions (%s):" % len(working_solutions))
        for it in working_solutions:
            print(repr(it))
        return results


def calculate_stats(results: list[bool]):
    return {
        "total": len(results),
        "validated": sum(results),
        "invalidated": len(results) - sum(results),
        "pass_rate": (sum(results) / len(results)) * 100,
    }


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate a batch of VimGolf solutions."
    )
    # evaluation format for parsing: terminal-bench-adaptor
    parser.add_argument(
        "--solution-format", type=str, help="Formating style of the JSONL file.",
        required=True
    )
    # jsonl filepath
    parser.add_argument(
        "--jsonl-file",
        type=str,
        help="Path to the JSONL file containing the VimGolf solutions.",
        required=True
    )
    # validator type
    parser.add_argument(
        "--validator",
        type=str,
        help="Which validator to use for scoring solutions: vimgolf-local, vimgolf-docker, vimgolf-validator",
        required=True
    )
    args = parser.parse_args()

    solution_format = args.solution_format
    jsonl_filepath = args.jsonl_file
    validator = args.validator

    evaluator = Evaluator(
        solution_format=solution_format,
        jsonl_file=jsonl_filepath,
        validator=validator,
    )
    results = evaluator.evaluate()
    stats = calculate_stats(results)
    print("Statistics:")
    print(json.dumps(stats, indent=4))


if __name__ == "__main__":
    main()
