import vimgolf_gym
import vimgolf_gym.dataclasses
import vimgolf_gym.lib
import json
import argparse
import pydantic

# need evaluation format for parsing: terminal-bench-adaptor
# need jsonl filepath


class TerminalBenchAdaptorSolution(pydantic.BaseModel):
    task_id: str
    trial_name: str
    challenge_hash: str
    solution: str


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
    else:
        raise ValueError(f"Unknown solution format: {solution_format}")


class Evaluator:
    def __init__(self, solution_format: str, jsonl_file: str, use_docker: bool):
        self.solution_format = solution_format
        self.jsonl_file = jsonl_file
        self.use_docker = use_docker

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
                with vimgolf_gym.make(
                    "vimgolf-custom",
                    custom_challenge=custom_challenge,
                    use_docker=self.use_docker,
                ) as env:
                    validated = env.verify_keys(custom_challenge.solution)
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
    parser.add_argument(
        "--solution-format", type=str, help="Formating style of the JSONL file."
    )
    parser.add_argument(
        "--jsonl-file",
        type=str,
        help="Path to the JSONL file containing the VimGolf solutions.",
    )
    parser.add_argument(
        "--use-docker",
        action="store_true",
        help="Use Docker to run the VimGolf challenges for isolation.",
    )
    args = parser.parse_args()

    solution_format = args.solution_format
    jsonl_filepath = args.jsonl_file
    use_docker = args.use_docker

    evaluator = Evaluator(
        solution_format=solution_format,
        jsonl_file=jsonl_filepath,
        use_docker=use_docker,
    )
    results = evaluator.evaluate()
    stats = calculate_stats(results)
    print("Statistics:")
    print(json.dumps(stats, indent=4))


if __name__ == "__main__":
    main()
