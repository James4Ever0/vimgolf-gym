import vimgolf_gym
import vimgolf_gym.dataclasses
import vimgolf_gym.lib
import vimgolf.vimgolf
import json
import argparse
import pydantic
import hashlib
import shlex
import subprocess
import shutil
import tempfile
import os
import pathlib


class TerminalBenchAdaptorSolution(pydantic.BaseModel):
    task_id: str
    trial_name: str
    challenge_hash: str
    solution: str


class VimGolfBenchmarkSolution(pydantic.BaseModel):
    task_id: str
    dataset_name: str
    trial_name: str
    start_time: float
    elapsed_time: float
    solution: str
    input_content: str
    output_content: str


def extract_first_non_empty_line(content: str) -> str:
    content = vimgolf.vimgolf.format_(content)
    lines = content.split("\n")
    for line in lines:
        if line.strip():
            return line
    return ""


def prepare_input(
    solution_format: str, solution
) -> vimgolf_gym.dataclasses.VimGolfCustomChallenge:
    if solution_format == "terminal-bench-adaptor":
        solution = TerminalBenchAdaptorSolution.parse_obj(solution)
        solution.solution = extract_first_non_empty_line(solution.solution)
        challenge_definition = vimgolf_gym.lib.get_local_challenge_definition(
            solution.challenge_hash
        )
        challenge_input = challenge_definition.input.data
        challenge_output = challenge_definition.output.data
        solution.challenge_hash
        return vimgolf_gym.dataclasses.VimGolfCustomChallenge(
            input=challenge_input,
            output=challenge_output,
            solution=solution.solution,
            name=solution.trial_name,
        )
    elif solution_format == "vimgolf-benchmark":
        solution = VimGolfBenchmarkSolution.parse_obj(solution)
        solution.solution = extract_first_non_empty_line(solution.solution)
        return vimgolf_gym.dataclasses.VimGolfCustomChallenge(
            input=solution.input_content,
            output=solution.output_content,
            solution=solution.solution,
            name=solution.trial_name,
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
    tainted = False
    input_content = vimgolf.vimgolf.format_(custom_challenge.input)
    output_content = vimgolf.vimgolf.format_(custom_challenge.output)
    solution_keys = custom_challenge.solution
    if not solution_keys:
        print("Empty solution keys")
        return validated

    with tempfile.TemporaryDirectory() as tmpdir:
        input_file_relpath = "input.txt"
        solution_file_relpath = "solution.txt"
        cmd = shlex.split(
            f"docker run --rm -v {tmpdir}:/verifier-input --entrypoint python --network=none agile4im/vimgolf-verifier:v0.0.3 /app/vimgolf-verifier.py single_shot"
        ) + [
            "--input_content",
            "/verifier-input/" + input_file_relpath,
            "--solution_keys",
            "/verifier-input/" + solution_file_relpath,
            "--load_from_path",
            "--remove_load_paths",
        ]
        docker_input_file_path = pathlib.Path(tmpdir) / input_file_relpath
        docker_input_file_path.write_text(input_content)
        docker_solution_file_path = pathlib.Path(tmpdir) / solution_file_relpath
        docker_solution_file_path.write_text(solution_keys)
        try:
            output = subprocess.check_output(cmd, timeout=15.0)  # type: ignore
            output = json.loads(output)
            checksum_server = output["checksum"]
            checksum_output = sha256_checksum(output_content)
            validated = checksum_server == checksum_output
            try:
                # confirm deletion
                assert not docker_input_file_path.is_file()
                assert not docker_solution_file_path.is_file()
            except AssertionError:
                tainted = True
        except subprocess.CalledProcessError:
            pass
        except subprocess.TimeoutExpired:
            pass
        except json.JSONDecodeError:
            pass
        except KeyError:
            pass
    if tainted:
        print("Warning: validator container is tainted")
    return validated


class Evaluator:
    def __init__(
        self,
        solution_format: str,
        jsonl_file: str,
        validator: str,
        solution_not_longer_than_output: bool,
    ):
        self.solution_format = solution_format
        self.jsonl_file = jsonl_file
        self.validator = validator
        self.solution_not_longer_than_output = solution_not_longer_than_output

        if self.validator == "vimgolf-validator":
            assert_docker_privilege()

    def _evaluate_single(
        self, custom_challenge: vimgolf_gym.dataclasses.VimGolfCustomChallenge
    ) -> bool:
        assert custom_challenge.name, "Challenge name is required"
        print("Challenge ID:", custom_challenge.name)
        print("Checking solution:", repr(custom_challenge.solution))
        # evaluate the solution
        validated = False
        if self.solution_not_longer_than_output:
            key_length = len(
                vimgolf.vimgolf.tokenize_keycode_reprs(custom_challenge.solution)
            )
            if key_length > len(custom_challenge.output):
                print(
                    "Invalidate solution: solution is longer than output (%s > %s)"
                    % (key_length, len(custom_challenge.output))
                )
                return False
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
        return validated

    def evaluate(self):
        results = []
        working_solutions = []
        validated_ids = []
        invalidated_ids = []
        challenges: list[vimgolf_gym.dataclasses.VimGolfCustomChallenge] = []
        with open(self.jsonl_file, "r") as f:
            for line in f:
                solution = json.loads(line)
                # use solution_format to parse the solution
                custom_challenge = prepare_input(self.solution_format, solution)
                challenges.append(custom_challenge)
        print("Evaluating", len(challenges), "solutions")
        for index, custom_challenge in enumerate(challenges):
            print("Processing item (%s/%s)" % (index + 1, len(challenges)))
            # retrieve the evaluation result
            validated = self._evaluate_single(custom_challenge)
            results.append(validated)
            if validated:
                working_solutions.append(custom_challenge.solution)
                validated_ids.append(custom_challenge.name)
            else:
                invalidated_ids.append(custom_challenge.name)
            print(
                "Result for item (%s/%s) is %s"
                % (index + 1, len(challenges), validated)
            )
        print("Evaluation finished")
        print("Working solutions (%s):" % len(working_solutions))
        for it in working_solutions:
            print(repr(it))
        return dict(
            results=results,
            validated_ids=validated_ids,
            invalidated_ids=invalidated_ids,
            working_solutions=working_solutions,
        )


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
        "--solution-format",
        type=str,
        help="Formating style of the JSONL file.",
        required=True,
    )
    # jsonl filepath
    parser.add_argument(
        "--jsonl-file",
        type=str,
        help="Path to the JSONL file containing the VimGolf solutions.",
        required=True,
    )
    # validator type
    parser.add_argument(
        "--validator",
        type=str,
        help="Which validator to use for scoring solutions: vimgolf-local, vimgolf-docker, vimgolf-validator",
        required=True,
    )
    parser.add_argument(
        "--solution-not-longer-than-output",
        action="store_true",
        help="Filter solutions that are longer than the output. This is done by comparing the the key count in solution to the length of the output",
    )
    parser.add_argument(
        "--result-savepath",
        type=str,
        required=True,
        help="Path to the result file as JSON.",
    )

    args = parser.parse_args()

    solution_format = args.solution_format
    jsonl_filepath = args.jsonl_file
    validator = args.validator
    result_savepath = args.result_savepath

    evaluator = Evaluator(
        solution_format=solution_format,
        jsonl_file=jsonl_filepath,
        validator=validator,
        solution_not_longer_than_output=args.solution_not_longer_than_output,
    )
    eval_result = evaluator.evaluate()
    print("Eval result:")
    print(eval_result)
    results = eval_result["results"]
    stats = calculate_stats(results)
    print("Statistics:")
    print(json.dumps(stats, indent=4))
    # save everything to json
    data = dict(eval_result=eval_result, stats=stats)
    with open(result_savepath, "w") as f:
        f.write(json.dumps(data, indent=4))
    print("Result saved to:", result_savepath)


if __name__ == "__main__":
    main()
