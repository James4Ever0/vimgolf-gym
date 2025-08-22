# TODO: implement a openai-gym style interface
# reference link: https://github.com/Farama-Foundation/Gymnasium

# TODO: implement a gradual scoring system by comparing the buffer with the target output, extracting the vim edit buffer in the middle of execution

import vimgolf.vimgolf as vimgolf
import vimgolf_gym.terminal_executor as terminal_executor
import os
import vimgolf_gym.dataclasses as dataclasses
import tempfile
import PIL.Image
import atexit

def make(env_name: str):
    env = VimGolfEnv()
    return env


def make_offline(): ...


def make_online(challenge_id: str):
    vimgolf.get_challenge_url(challenge_id)


def make_offline_with_cybergod_dataset(challenge_id: str):
    if not os.path.exists("cybergod_vimgolf_dataset.zip"):
        download_cybergod_vimgolf_dataset()


def download_cybergod_vimgolf_dataset():
    # identical file download urls
    zip_download_urls = [
        "https://www.kaggle.com/api/v1/datasets/download/jessysisca/vimgolf-challenges-and-solutions"
    ]

    huggingface_repo_urls = [
        "https://huggingface.co/datasets/James4Ever0/vimgolf_challenges_and_solutions"  # need to clone the repo
    ]
    # get one that is downloadable
    ...


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
        self.command = ["vimgolf", "local", self.input_file, self.output_file]
        self.width = width
        self.height = height
        self.create_executor()
        # atexit.register()

    def act(self, action: str):
        """Take an action

        :param action: the action to take"""
        self.executor.input(action)

    def create_executor(self):
        """Create the executor"""
        self.executor = terminal_executor.TerminalExecutor(
            command=self.command, width=self.width, height=self.height
        )

    def reset(self):
        """Reset the environment"""
        self.close()
        self.create_executor()

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
        setattr(self, "executor", None)
