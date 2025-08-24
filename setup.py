from setuptools import setup, find_packages, Command
from shutil import rmtree
import os
import sys

here = os.path.abspath(os.path.dirname(__file__))
VERSION="0.0.5"

class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
            rmtree(os.path.join(here, "build"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        sys.exit()


# Read the contents of README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="vimgolf-gym",
    version=VERSION,  # Specify your initial version
    author="James Brown",  # Replace with your name
    author_email="randomvoidmail@foxmail.com",  # Replace with your email
    description="A gym environment for VimGolf challenges",  # Provide a short description
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/james4ever0/vimgolf-gym",  # Replace with your project URL
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Choose an appropriate license
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",  # Specify required Python version
    install_requires=requirements,
    cmdclass={
        "upload": UploadCommand,
    },
)