# TODO: implement a openai-gym style interface
# reference link: https://github.com/Farama-Foundation/Gymnasium

# TODO: implement a gradual scoring system by comparing the buffer with the target output, extracting the vim edit buffer in the middle of execution

import vimgolf_gym.terminal_executor as terminal_executor
import vimgolf_gym.log_parser as log_parser
import os
import vimgolf_gym.dataclasses as dataclasses
import tempfile
import PIL.Image
import atexit
import sys
import vimgolf.vimgolf as vimgolf
import urllib.request
import pathlib
import zipfile
import shutil

HOMEDIR = os.path.expanduser("~")

CYBERGOD_VIMGOLF_DATASET_BASEDIR = os.path.join(
    HOMEDIR, ".cache", "cybergod-vimgolf-dataset"
)

CYBERGOD_VIMGOLF_GYM_DATASET_DOWNLOADED = os.path.join(
    CYBERGOD_VIMGOLF_DATASET_BASEDIR, "DATASET_DOWNLOADED"
)  # a flag to indicate whether the dataset has been downloaded

os.makedirs(CYBERGOD_VIMGOLF_DATASET_BASEDIR, exist_ok=True)


def make(env_name: str):
    if env_name == "vimgolf-test":
        env = make_test()
    else:
        raise NotImplementedError
    return env


def make_test():
    input_text = ""
    output_text = "hello world\nhello world\n"
    return make_env_with_text(input_text, output_text)


def make_env_with_text(input_text: str, output_text: str):
    tempdir = tempfile.TemporaryDirectory()
    atexit.register(tempdir.cleanup)
    input_file = os.path.join(tempdir.name, "input.txt")
    output_file = os.path.join(tempdir.name, "output.txt")
    with open(input_file, "w") as f:
        f.write(input_text)
    with open(output_file, "w") as f:
        f.write(output_text)
    env = VimGolfEnv(input_file, output_file)
    return env


def make_offline(input_file: str, output_file: str):
    return VimGolfEnv(input_file, output_file)


def make_online(challenge_id: str):
    challenge_url = vimgolf.get_challenge_url(challenge_id)
    challenge_data = urllib.request.urlopen(challenge_url).read()
    challenge = dataclasses.VimGolfChallengeDefinition.parse_raw(challenge_data)
    return make_env_with_text(input_text=challenge.input, output_text=challenge.output)


def make_offline_with_cybergod_dataset(challenge_id: str):
    if not os.path.exists(CYBERGOD_VIMGOLF_GYM_DATASET_DOWNLOADED):
        download_cybergod_vimgolf_dataset()
    challenge_file = os.path.join(
        CYBERGOD_VIMGOLF_DATASET_BASEDIR, challenge_id, "challenge.json"
    )
    assert os.path.exists(challenge_file)
    with open(challenge_file, "r") as f:
        challenge = dataclasses.VimGolfChallengeDefinition.parse_raw(f.read())
    return make_env_with_text(input_text=challenge.input, output_text=challenge.output)


def download_cybergod_vimgolf_dataset():
    try:
        zip_download_urls = "https://www.kaggle.com/api/v1/datasets/download/jessysisca/vimgolf-challenges-and-solutions"

        with tempfile.TemporaryDirectory() as tempdir:
            zip_file_path = os.path.join(
                tempdir, "vimgolf-challenges-and-solutions.zip"
            )
            urllib.request.urlretrieve(zip_download_urls, zip_file_path)
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                # extract to CYBERGOD_VIMGOLF_GYM_DATASET_DIR
                zip_ref.extractall(CYBERGOD_VIMGOLF_DATASET_BASEDIR)
        # after all, touch the flag CYBERGOD_VIMGOLF_GYM_DATASET_DOWNLOADED
        pathlib.Path(CYBERGOD_VIMGOLF_GYM_DATASET_DOWNLOADED).touch()
    finally:
        if not os.path.exists(CYBERGOD_VIMGOLF_GYM_DATASET_DOWNLOADED):
            # cleanup the dataset basedir, if the dataset is not downloaded successfully
            shutil.rmtree(CYBERGOD_VIMGOLF_DATASET_BASEDIR)


class VimGolfEnv:
    def __init__(
        self, input_file: str, output_file: str, width: int = 80, height: int = 24
    ):
        """Initialize the environment with the given input and output files.

        :param input_file: the input file path
        :param output_file: the output file path
        :param width: the width of the terminal
        :param height: the height of the terminal
        """
        self.input_file = input_file
        self.output_file = output_file
        # TODO: run a modified version of vimgolf local python script writing progress to a jsonl file, which embeds in this script, for easy state inspection and data collection (we can create a temporary directory for cleanup)
        self.log_directory = tempfile.TemporaryDirectory()

        self.log_file = os.path.join(self.log_directory, "vimgolf.log")

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
        ]

        self.width = width
        self.height = height
        self.create_executor_and_log_watcher()

        atexit.register(self.log_directory.cleanup)

    def act(self, action: str):
        """Take an action

        :param action: the action to take
        """
        self.executor.input(action)

    def success(self):
        """Check if the vimgolf challenge has been solved successfully"""
        return self.log_watcher.parser.success

    def get_best_success_result(self):
        return self.log_watcher.parser.get_best_success_result()

    def get_last_success_result(self):
        return self.log_watcher.parser.get_last_success_result()

    def dump_results(self):
        """Dump the results of the vimgolf challenge"""
        return self.log_watcher.parser.results

    def dump_success_results(self):
        """Dump the success results of the vimgolf challenge"""
        return self.log_watcher.parser.success_results

    def create_executor_and_log_watcher(self):
        """Create the executor and log watcher"""
        self.executor = terminal_executor.TerminalExecutor(
            command=self.command, width=self.width, height=self.height
        )
        self.log_watcher = log_parser.VimGolfLogWatcher(self.log_file)

    def reset(self):
        """Reset the environment"""
        self.close()
        self.create_executor_and_log_watcher()

    def render(self):
        """Render the environment"""
        screenshot = self.screenshot()
        # display the screenshot
        screenshot.show()

    def screenshot(self):
        """Take a screenshot of the environment

        :return: the screenshot"""
        with tempfile.TemporaryDirectory() as tmpdir:
            png_tmpfile_path = os.path.join(tmpdir, "screenshot.png")
            self.executor.screenshot(png_tmpfile_path)
            image = PIL.Image.open(png_tmpfile_path)
            return image

    def close(self):
        """Close the environment"""
        self.executor.close()
        del self.executor
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        setattr(self, "executor", None)
