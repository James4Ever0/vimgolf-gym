"""
Library of vimgolf-gym
"""

# TODO: implement a openai-gym style interface
# reference link: https://github.com/Farama-Foundation/Gymnasium

# TODO: implement a gradual scoring system by comparing the buffer with the target output, extracting the vim edit buffer in the middle of execution

# Vim API reference:
# https://github.com/LachlanGray/vim-agent
# https://github.com/nsbradford/VimGPT

# TODO: dockerize the executor part, push the image, offer an option or environment variable to use dockerized executor

import atexit
import os
import pathlib
import shutil
import sys
import tempfile
import typing
import time
import zipfile

import PIL.Image
import requests
import uuid

import vimgolf.vimgolf as vimgolf
import vimgolf_gym.dataclasses as dataclasses
import vimgolf_gym.log_parser as log_parser
import vimgolf_gym.terminal_executor as terminal_executor
from vimgolf_gym._vimrc import (
    _prepare_cybergod_vimrc_with_buffer_file,
    _CYBERGOD_VIMGOLF_VIMRC_FILEPATH,
)
import subprocess

_HOMEDIR = os.path.expanduser("~")

_CYBERGOD_VIMGOLF_DATASET_BASEDIR = os.path.join(
    _HOMEDIR, ".cache", "cybergod-vimgolf-dataset"
)

_CYBERGOD_VIMGOLF_GYM_DATASET_DOWNLOADED = os.path.join(
    _CYBERGOD_VIMGOLF_DATASET_BASEDIR, "DATASET_DOWNLOADED"
)  # a flag to indicate whether the dataset has been downloaded

os.makedirs(_CYBERGOD_VIMGOLF_DATASET_BASEDIR, exist_ok=True)

__all__ = [
    "make",
    "list_local_challenge_ids",
    "get_local_challenge_definition",
    "get_local_challenge_metadata",
    "get_local_challenge_worst_solution",
    "get_local_challenge_worst_solution_header",
    "VimGolfEnv",
]


class DatasetInitError(Exception):
    pass


def assert_challenge_id_length(challenge_id: str) -> bool:
    """Assert the challenge_id length to be 24"""
    assert len(challenge_id) == 24


def make(
    env_name: str,
    custom_challenge: typing.Optional[dataclasses.VimGolfCustomChallenge] = None,
    use_docker: bool = False,
    log_buffer: bool = False,
) -> "VimGolfEnv":
    """
    Create a VimGolf environment.

    The env_name can be one of the following:
    - `vimgolf-test`: A simple test environment.
    - `vimgolf-local-<challenge_id>`: A local environment for a specific challenge.
    - `vimgolf-online-<challenge_id>`: An online environment for a specific challenge.
    - `vimgolf-custom`: A custom environment. The `custom_challenge` parameter will be used.

    Args:
        env_name (str): The name of the environment to create.
        use_docker (bool, optional): Whether to use a dockerized executor. Defaults to False.
        log_buffer (bool, optional): Whether to log the editor buffer or not. Defaults to False.
        custom_challenge (Optional[VimGolfCustomChallenge], optional): The custom challenge to use. Defaults to None.

    Raises:
        NotImplementedError: If the environment name is not recognized.

    Returns:
        VimGolfEnv: The created environment
    """
    if use_docker:
        os.environ["VIMGOLF_GYM_USE_DOCKER"] = "1"
    else:
        os.environ["VIMGOLF_GYM_USE_DOCKER"] = "0"

    if log_buffer:
        os.environ["VIMGOLF_GYM_LOG_BUFFER"] = "1"
    else:
        os.environ["VIMGOLF_GYM_LOG_BUFFER"] = "0"

    if env_name == "vimgolf-test":
        env = make_test()
    elif env_name == "vimgolf-custom":
        assert (
            custom_challenge
        ), "custom_challenge must be provided for vimgolf-custom environment"
        env = make_env_with_text(
            input=custom_challenge.input, output=custom_challenge.output
        )
    elif env_name.startswith("vimgolf-local-"):
        challenge_id = env_name[len("vimgolf-local-") :]
        assert_challenge_id_length(challenge_id)
        env = make_offline_with_cybergod_dataset(challenge_id)
    elif env_name.startswith("vimgolf-online-"):
        challenge_id = env_name[len("vimgolf-online-") :]
        assert_challenge_id_length(challenge_id)
        env = make_online(challenge_id)
    else:
        raise NotImplementedError
    return env


def make_test() -> "VimGolfEnv":
    """
    Create an environment for a simple test challenge.

    The test challenge is about typing "hello world" into the buffer and then saving and quitting vim.
    The expected solution is "hello world\nhello world\n".
    """
    input_text = ""
    output_text = "hello world\nhello world\n"
    return make_env_with_text(input_text, output_text)


def make_env_with_text(input_text: str, output_text: str) -> "VimGolfEnv":
    """
    Create a VimGolfEnv with given input and output text.

    Creates a temporary directory and two files, one for the input and one for the output.
    Then it calls make_offline with the paths of the two files.

    Args:
        input_text (str): The starting code.
        output_text (str): The expected solution code.

    Returns:
        VimGolfEnv: The environment object.
    """
    tempdir = tempfile.TemporaryDirectory()
    atexit.register(tempdir.cleanup)
    input_file = os.path.join(tempdir.name, "input.txt")
    output_file = os.path.join(tempdir.name, "output.txt")
    with open(input_file, "w") as f:
        f.write(input_text)
    with open(output_file, "w") as f:
        f.write(output_text)
    return make_offline(input_file, output_file)


def make_offline(input_file: str, output_file: str) -> "VimGolfEnv":
    """
    Create an environment from a VimGolf challenge given as a local file.

    Args:
        input_file (str): Path to the file containing the starting code.
        output_file (str): Path to the file containing the expected solution code.

    Returns:
        VimGolfEnv: An environment for the given challenge.
    """
    use_docker = os.environ.get("VIMGOLF_GYM_USE_DOCKER", None) == "1"
    log_buffer = os.environ.get("VIMGOLF_GYM_LOG_BUFFER", None) == "1"
    return VimGolfEnv(
        input_file=input_file,
        output_file=output_file,
        use_docker=use_docker,
        log_buffer=log_buffer,
    )


def make_online(challenge_id: str) -> "VimGolfEnv":
    """
    Create an environment from a VimGolf challenge online.

    Given a challenge_id, obtain the challenge definition from the VimGolf website
    and create an environment out of it.

    Args:
        challenge_id (str): Unique identifier for the challenge.

    Returns:
        VimGolfEnv: An environment for the given challenge.
    """
    challenge_url = vimgolf.get_challenge_url(challenge_id)
    challenge_data = requests.get(challenge_url).content
    challenge = dataclasses.VimGolfChallengeDefinition.parse_raw(challenge_data)
    return make_env_with_challenge(challenge)


def make_env_with_challenge(
    challenge: dataclasses.VimGolfChallengeDefinition,
) -> "VimGolfEnv":
    """
    Create an environment from a VimGolfChallengeDefinition object.

    This function simply passes the input and output text to make_env_with_text,
    which will create a VimGolfEnv with the given text.

    Args:
        challenge: VimGolfChallengeDefinition object

    Returns:
        VimGolfEnv: An environment for the given challenge
    """
    return make_env_with_text(
        input_text=challenge.input.data, output_text=challenge.output.data
    )


def init_cybergod_vimgolf_dataset() -> None:
    """
    Initialize the local dataset by downloading it if it does not exist yet.

    After this function is called, the local dataset should be downloaded and
    ready to use.

    This function is called by `list_local_challenge_ids` and `make_offline_with_cybergod_dataset`.
    """
    if not os.path.exists(_CYBERGOD_VIMGOLF_GYM_DATASET_DOWNLOADED):
        download_cybergod_vimgolf_dataset()


def list_local_challenge_ids() -> list[str]:
    """
    List all challenge ids in the local dataset.

    This function will download the local dataset if it does not exist yet.

    Returns:
        list[str]: a list of all challenge ids in the local dataset.
    """

    init_cybergod_vimgolf_dataset()
    challenges_dir = os.path.join(
        _CYBERGOD_VIMGOLF_DATASET_BASEDIR,
        "challenges",
    )
    challenge_ids = os.listdir(challenges_dir)
    return challenge_ids


def make_offline_with_cybergod_dataset(challenge_id: str) -> "VimGolfEnv":
    """
    Load a VimGolf challenge from the local dataset and make an environment
    out of it.

    Given a challenge_id, find the corresponding challenge definition in the
    dataset, parse it into a VimGolfChallengeDefinition object, and create a
    VimGolfEnv out of it.

    Args:
        challenge_id (str): Unique identifier for the challenge.

    Returns:
        VimGolfEnv: An environment for the given challenge.
    """
    init_cybergod_vimgolf_dataset()
    challenge = get_local_challenge_definition(challenge_id)
    return make_env_with_challenge(challenge)


def get_local_challenge_metadata(challenge_id: str):
    """
    Load a VimGolf challenge's metadata from the local dataset.

    Given a challenge_id, find the corresponding JSON file in the dataset
    and parse it into a VimGolfChallengeMetadata object.

    Args:
        challenge_id (str): Unique identifier for the challenge.

    Returns:
        VimGolfChallengeMetadata: Parsed challenge metadata.

    Raises:
        AssertionError: If the metadata file does not exist.
    """
    metadata_file = os.path.join(
        _CYBERGOD_VIMGOLF_DATASET_BASEDIR, "challenges", challenge_id, "metadata.json"
    )
    assert os.path.exists(metadata_file), (
        "Metadata file '%s' does not exist" % metadata_file
    )
    with open(metadata_file, "r") as f:
        metadata = dataclasses.VimGolfChallengeMetadata.parse_raw(f.read())
    return metadata


def get_local_challenge_worst_solution(challenge_id: str):
    """
    Load the worst solution for a challenge from the local dataset.

    Given a challenge_id, find the corresponding JSON file in the dataset
    and parse it into a VimGolfPublicSolution object.

    Args:
        challenge_id (str): Unique identifier for the challenge.

    Returns:
        VimGolfPublicSolution: Parsed worst solution.

    Raises:
        AssertionError: If the worst solution file does not exist.
    """
    metadata_file = os.path.join(
        _CYBERGOD_VIMGOLF_DATASET_BASEDIR,
        "challenges",
        challenge_id,
        "worst_solution.json",
    )
    assert os.path.exists(metadata_file), (
        "Worst solution file '%s' does not exist" % metadata_file
    )
    with open(metadata_file, "r") as f:
        Worst_solution = dataclasses.VimGolfPublicSolution.parse_raw(f.read())
    return Worst_solution


def get_local_challenge_worst_solution_header(challenge_id: str):
    """
    Parse the worst solution's header string from the local dataset.

    Given a challenge_id, find the corresponding worst solution file in the dataset
    and parse its header string into a VimGolfParsedPublicSolutionHeader object.

    Args:
        challenge_id (str): Unique identifier for the challenge.

    Returns:
        VimGolfParsedPublicSolutionHeader: Parsed header string.

    Raises:
        AssertionError: If the worst solution file does not exist.
    """
    solution = get_local_challenge_worst_solution(challenge_id)
    header = solution.header
    ret = dataclasses.parse_public_solution_header(header)
    return ret


def get_local_challenge_definition(challenge_id: str):
    """
    Load a VimGolf challenge definition from the local dataset.

    Given a challenge_id, find the corresponding JSON file in the dataset
    and parse it into a VimGolfChallengeDefinition object.

    Args:
        challenge_id (str): Unique identifier for the challenge.

    Returns:
        VimGolfChallengeDefinition: Parsed challenge definition.

    Raises:
        AssertionError: If the challenge file does not exist.
    """
    challenge_file = os.path.join(
        _CYBERGOD_VIMGOLF_DATASET_BASEDIR, "challenges", challenge_id, "challenge.json"
    )
    assert os.path.exists(challenge_file), (
        "Challenge file '%s' does not exist" % challenge_file
    )
    with open(challenge_file, "r") as f:
        challenge = dataclasses.VimGolfChallengeDefinition.parse_raw(f.read())
    return challenge


def download_cybergod_vimgolf_dataset():
    """
    Download the CyberGod VimGolf dataset from various sources.

    This function is called when the dataset is not initialized yet. It downloads the dataset
    from various sources (Kaggle, Hugging Face, GitHub Releases, GitHub Mirror) and extracts it
    to the dataset directory. After the download is finished, it touches the flag file
    _CYBERGOD_VIMGOLF_GYM_DATASET_DOWNLOADED to indicate that the dataset is initialized.

    If the download fails, it raises an exception and cleans up the dataset directory.

    :raises DatasetInitError: If the dataset download fails.
    """
    print(
        "Initializing CyberGod VimGolf dataset at:", _CYBERGOD_VIMGOLF_DATASET_BASEDIR
    )
    try:
        # TODO: add huggingface, hf-mirror.com, github releases and github mirror links
        download_urls = [
            "https://www.kaggle.com/api/v1/datasets/download/jessysisca/vimgolf-challenges-and-solutions",
            "https://hf-mirror.com/datasets/James4Ever0/vimgolf_challenges_and_solutions/resolve/main/challenges.zip?download=true",
            "https://huggingface.co/datasets/James4Ever0/vimgolf_challenges_and_solutions/resolve/main/challenges.zip?download=true",
            "https://github.com/James4Ever0/vimgolf-gym/releases/download/dataset-release/challenges.zip",
            "https://bgithub.xyz/James4Ever0/vimgolf-gym/releases/download/dataset-release/challenges.zip",
        ]
        with tempfile.TemporaryDirectory() as tempdir:
            zip_file_path = os.path.join(
                tempdir, "vimgolf-challenges-and-solutions.zip"
            )
            with open(zip_file_path, "wb") as f:
                content = None
                for url in download_urls:
                    try:
                        print("Downloading:", url)
                        content = requests.get(
                            url, allow_redirects=True, timeout=10
                        ).content
                        break
                    except requests.Timeout:
                        print("Timeout, trying next URL")
                if content:
                    f.write(content)
                else:
                    raise DatasetInitError("Failed to download the dataset")
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                # extract to CYBERGOD_VIMGOLF_GYM_DATASET_DIR
                zip_ref.extractall(_CYBERGOD_VIMGOLF_DATASET_BASEDIR)
        # after all, touch the flag _CYBERGOD_VIMGOLF_GYM_DATASET_DOWNLOADED
        pathlib.Path(_CYBERGOD_VIMGOLF_GYM_DATASET_DOWNLOADED).touch()
    finally:
        if not os.path.exists(_CYBERGOD_VIMGOLF_GYM_DATASET_DOWNLOADED):
            # cleanup the dataset basedir, if the dataset is not downloaded successfully
            shutil.rmtree(_CYBERGOD_VIMGOLF_DATASET_BASEDIR)


class VimGolfEnv:
    def __init__(
        self,
        input_file: str,
        output_file: str,
        width: int = 80,
        height: int = 24,
        use_docker: bool = False,
        log_buffer: bool = False,
    ):
        """Initialize the environment with the given input and output files.

        Args:
            input_file (str): the input file path
            output_file (str): the output file path
            width (int): the width of the terminal
            height (int): the height of the terminal
            use_docker (bool): whether use dockerized executor or local (requiring vim installed)
            log_buffer (bool): whether to log the editor buffer or not
        """

        self.use_docker = use_docker
        """whether use dockerized executor or local (requiring vim installed)"""

        self.input_file = input_file
        """the input file path"""

        self.output_file = output_file
        """the output file path"""

        self.log_buffer = log_buffer
        """whether to log the editor buffer or not"""

        assert os.path.isfile(
            self.input_file
        ), f"Input file {self.input_file} does not exist."
        assert os.path.isfile(
            self.output_file
        ), f"Output file {self.output_file} does not exist."

        # TODO: run a modified version of vimgolf local python script writing progress to a jsonl file, which embeds in this script, for easy state inspection and data collection (we can create a temporary directory for cleanup)
        self.log_directory = tempfile.TemporaryDirectory()
        """the log directory, where tempfiles stored"""

        self.log_file = os.path.join(self.log_directory.name, "vimgolf.log")
        """the log file path, used to retrieve progress info of the vimgolf process"""

        self.buffer_file = os.path.join(
            self.log_directory.name, "vimgolf_editor_buffer"
        )
        """the editor buffer, used to track granual progress"""

        self._container_name = str(uuid.uuid4())

        if self.use_docker:
            mountpoint = "/vimgolf_gym_workdir"
            docker_output_file = os.path.join(mountpoint, "out")
            docker_input_file = os.path.join(mountpoint, "in")
            docker_buffer_file = os.path.join(mountpoint, "vimgolf_editor_buffer")
            docker_log_file = os.path.join(mountpoint, "vimgolf.log")
            shutil.copy(self.input_file, os.path.join(self.log_directory.name, "in"))
            shutil.copy(self.output_file, os.path.join(self.log_directory.name, "out"))
            extra_docker_run_params = []
            if self.log_buffer:
                _prepare_cybergod_vimrc_with_buffer_file(docker_buffer_file)
                extra_docker_run_params += [
                    "-v",
                    "%s:%s:ro"
                    % (
                        _CYBERGOD_VIMGOLF_VIMRC_FILEPATH,
                        "/usr/local/lib/python3.10/dist-packages/vimgolf/vimgolf.vimrc",
                    ),
                ]
            self.command = [
                "docker",
                "run",
                "--rm",
                "-it",
                "--name",
                self._container_name,
                "-v",
                "%s:%s" % (self.log_directory.name, mountpoint),
                *extra_docker_run_params,
                "--entrypoint",
                "python3",
                "agile4im/cybergod_vimgolf_gym",
                "-m",
                "vimgolf_gym.vimgolf",
                "--input_file",
                docker_input_file,
                "--output_file",
                docker_output_file,
                "--log_file",
                docker_log_file,
            ]
        else:
            extra_flags = []
            if self.log_buffer:
                extra_flags += ["--buffer_file", self.buffer_file]
            self.command = [
                sys.executable,
                "-m",
                "vimgolf_gym.vimgolf",
                "--input_file",
                self.input_file,
                "--output_file",
                self.output_file,
                "--log_file",
                self.log_file,
                *extra_flags,
            ]

        self._closing = False

        self.command: list[str]
        """the command passed to underlying terminal executor"""

        self.width = width
        """terminal width"""
        self.height = height
        """terminal height"""
        self.create_executor_and_log_watcher()

        self.executor: terminal_executor.TerminalExecutor
        """terminal executor for running vimgolf process"""
        self.log_watcher: log_parser.VimGolfLogWatcher
        """log watcher for tracking vimgolf log output"""
        atexit.register(self.log_directory.cleanup)
        atexit.register(self._kill_docker_container)

    @property
    def buffer(self) -> typing.Optional[bytes]:
        """The editor buffer"""
        if not self.log_buffer:
            return None
        else:
            if os.path.isfile(self.buffer_file):
                with open(self.buffer_file, "rb") as f:
                    return f.read()

    def act(self, action: str):
        """Take an action

        Args:
            action (str): the action to take
        """
        self.executor.input(action)

    @property
    def success(self):
        """Check if the vimgolf challenge has been solved successfully"""
        return self.log_watcher.success

    def get_best_success_result(self):
        """
        Return the best success result in the log watcher.

        Returns:
            Optional[VimGolfEnvResult]: The best success result.
        """
        return self.log_watcher.get_best_success_result()

    def get_last_success_result(self):
        """
        Return the last success result in the log watcher.

        Returns:
            Optional[VimGolfEnvResult]: The last success result
        """
        return self.log_watcher.get_last_success_result()

    @property
    def results(self):
        """The results of the vimgolf challenge environment

        Returns:
            list[VimGolfEnvResult]: The results of the vimgolf challenge environment
        """
        return self.log_watcher.results

    def __enter__(self):
        return self

    def __exit__(self, exc, value, tb):
        self.close()

    @property
    def success_results(self):
        """The success results of the vimgolf challenge environment

        Returns:
            list[VimGolfEnvResult]: The success results of the vimgolf challenge environment
        """
        return self.log_watcher.success_results

    def create_executor_and_log_watcher(self):
        """Create the executor and log watcher"""
        self.executor = terminal_executor.TerminalExecutor(
            command=self.command, width=self.width, height=self.height
        )
        if not hasattr(self, "log_watcher"):
            self.log_watcher = log_parser.VimGolfLogWatcher(self.log_file)
        # shall we wait the executor be ready
        # we wait for the 'play(...)' indicator to appear in the log file.
        while True:
            if os.path.exists(self.log_file):
                with open(self.log_file, "r") as f:
                    log_content = f.read()
                    if "play(...)" in log_content:
                        break
            time.sleep(0.5)

    def reset(self):
        """Reset the environment"""
        self.close()
        self._closing = False
        self.create_executor_and_log_watcher()

    def render(self):
        """Render the environment"""
        screenshot = self.screenshot()
        # display the screenshot
        screenshot.show()

    def screenshot(self):
        """Take a screenshot of the environment

        Returns:
            PIL.Image.Image: The screenshot
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            png_tmpfile_path = os.path.join(tmpdir, "screenshot.png")
            self.executor.screenshot(png_tmpfile_path)
            image = PIL.Image.open(png_tmpfile_path)
            return image

    def _kill_docker_container(self):
        if self.use_docker:
            subprocess.run(
                ["docker", "kill", self._container_name], capture_output=True
            )

    def close(self):
        """Close the environment"""
        if not self._closing:
            self._closing = True
            self.executor.close()
            self._kill_docker_container()
            del self.executor
            if os.path.exists(self.log_file):
                os.remove(self.log_file)
            setattr(self, "executor", None)
