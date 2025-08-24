"""
Terminal executor

Code source: [terminal_asciicast_record_executor.py](https://github.com/James4Ever0/agi_computer_control/blob/master/web_gui_terminal_recorder/executor_and_replayer/terminal_asciicast_record_executor.py) (modified)
"""

# pending pypi project name: termexec

import threading
import time
import os
import signal

import agg_python_bindings
import ptyprocess
from pydantic import BaseModel
from typing import Protocol, cast


class Cursor(BaseModel):
    """Cursor position and visibility.

    Attributes:
        x (int): The x-coordinate (column) of the cursor.
        y (int): The y-coordinate (row/line) of the cursor.
        hidden (bool): True if the cursor is hidden, False otherwise.
    """

    x: int
    y: int
    hidden: bool


def decode_bytes_to_utf8_safe(_bytes: bytes):
    """
    Decode with UTF-8, but replace errors with a replacement character (�).
    """
    ret = _bytes.decode("utf-8", errors="replace")
    return ret


# screen init params: width, height
# screen traits: write_bytes, display, screenshot


class _TerminalScreenProtocol(Protocol):
    def write_bytes(self, _bytes: bytes): ...
    @property
    def display(self) -> str: ...
    def screenshot(self, png_output_path: str): ...
    def close(self): ...
    @property
    def cursor(self) -> Cursor: ...


class AvtScreen:
    def __init__(self, width: int, height: int):
        """
        Initialize an AvtScreen object.

        Args:
            width (int): The width of the virtual terminal emulator.
            height (int): The height of the virtual terminal emulator.
        """
        self.vt = agg_python_bindings.TerminalEmulator(width, height)
        """terminal emulator provided by avt"""

    def write_bytes(self, _bytes: bytes):
        """
        Write the given bytes to the virtual terminal emulator.

        The bytes are decoded with UTF-8 and any decoding errors are replaced with a
        replacement character (�).

        Args:
            _bytes (bytes): The bytes to be written to the virtual terminal emulator.
        """
        decoded_bytes = decode_bytes_to_utf8_safe(_bytes)
        self.vt.feed_str(decoded_bytes)

    @property
    def cursor(self):
        """
        Get the current position of the cursor as a `Cursor` object.

        Returns:
            Cursor: A `Cursor` object with `x` and `y` properties for the current column and row of the cursor, respectively, and a `hidden` property which is `True` if the cursor is hidden and `False` otherwise.
        """
        col, row, visible = self.vt.get_cursor()
        ret = Cursor(x=col, y=row, hidden=not visible)
        return ret

    @property
    def display(self):
        """
        Get the current display of the terminal emulator as a string.

        Returns:
            str: A string representation of the current display of the terminal emulator.
        """
        ret = "\n".join(self.vt.text_raw())
        return ret

    def screenshot(self, png_output_path: str):
        """
        Saves the current state of the terminal emulator to a PNG image file.

        The image has the same width and height as the terminal emulator.

        Args:
            png_output_path: The path to write the PNG image to.
        """
        self.vt.screenshot(png_output_path)

    def close(self):
        """
        Releases all resources used by the terminal emulator.

        This is necessary to avoid crashes when creating multiple instances of this class.
        """
        del self.vt


class TerminalProcess:
    def __init__(self, command: list[str], width: int, height: int, backend="avt"):
        """
        Initializes the terminal emulator with a command to execute.

        Args:
            command (list[str]): List of command strings to execute in the terminal
            width (int): Width of the terminal emulator
            height (int): Height of the terminal emulator
            backend (str, optional): Backend to use for terminal emulator. Defaults to "avt".
        """
        rows, cols = height, width
        self.pty_process: ptyprocess.PtyProcess = cast(ptyprocess.PtyProcess,ptyprocess.PtyProcess.spawn(command, dimensions=(rows, cols)))
        """a process executing command in a pseudo terminal"""
        
        if backend == "avt":
            self.vt_screen = AvtScreen(width=width, height=height)
            """virtual terminal screen"""
        else:
            raise ValueError(
                "Unknown terminal emulator backend '%s' (known ones: avt, pyte)"
                % backend
            )
        
        self.vt_screen = cast(_TerminalScreenProtocol, self.vt_screen)

        self.__pty_process_reading_thread = threading.Thread(
            target=self.__read_and_update_screen, daemon=True
        )
        self.__start_ptyprocess_reading_thread()

    def __start_ptyprocess_reading_thread(self):
        """Starts a thread to read output from the terminal process and update the Pyte screen"""
        self.__pty_process_reading_thread.start()

    def write(self, data: bytes):
        """Writes input data to the terminal process"""
        self.pty_process.write(data)

    def close(self):
        """Closes the terminal process and the reading thread"""
        os.kill(self.pty_process.pid, signal.SIGTERM)
        time.sleep(0.5)
        if self.pty_process.isalive:
            os.kill(self.pty_process.pid, signal.SIGKILL)
        self.vt_screen.close()
        self.__pty_process_reading_thread.join(timeout=0.5)

    def __read_and_update_screen(self, poll_interval=0.01):
        """Reads available output from terminal and updates Pyte screen

        Args:
            poll_interval (float, optional): Interval in seconds to poll for available output. Defaults to 0.01.
        """
        while True:
            try:
                # ptyprocess.read is blocking. only pexpect has read_nonblocking
                process_output_bytes = self.pty_process.read(1024)
                # write bytes to pyte screen
                self.vt_screen.write_bytes(process_output_bytes)
            except KeyboardInterrupt:  # user interrupted
                break
            except SystemExit:  # python process exit
                break
            except SystemError:  # python error
                break
            except EOFError:  # terminal died
                break
            except:
                # Timeout means no data available, EOF means process ended
                pass
            finally:
                time.sleep(poll_interval)


class TerminalExecutor:
    def __init__(self, command: list[str], width: int, height: int):
        """
        Initializes executor with a command to run in terminal emulator, using avt as backend.

        Args:
            command (list[str]): List of command strings to execute
            width (int): Width of the terminal emulator
            height (int): Height of the terminal emulator
        """
        self.terminal = TerminalProcess(command=command, width=width, height=height)

    def input(self, text: str):
        """
        Sends input text to the terminal process

        Args:
            text (str): The input text to send
        """
        self.terminal.write(text.encode())
        # Allow time for processing output
        time.sleep(0.1)

    @property
    def display(self) -> str:
        """
        Get the current display of the terminal emulator as a string.
        """

        return self.terminal.vt_screen.display

    def screenshot(self, png_save_path: str):
        """
        Saves the current display of the terminal emulator as a .png file

        Args:
            png_save_path (str): The path to save the screenshot to
        """
        self.terminal.vt_screen.screenshot(png_save_path)

    def close(self):
        """
        Closes the terminal emulator process and the associated reading thread.
        """
        self.terminal.close()


def test_harmless_command_locally_with_bash():
    """
    Tests the TerminalExecutor class with a harmless command by running a docker alpine container
    and executing a series of input events, then taking a screenshot and dumping the terminal
    display to a file.
    """
    SLEEP_INTERVAL = 0.5
    command = ["docker", "run", "--rm", "-it", "alpine"]
    input_events = ['echo "Hello World!"', "\n"]
    executor = TerminalExecutor(command=command, width=80, height=24)
    time.sleep(1)
    for event in input_events:
        executor.input(event)
        time.sleep(SLEEP_INTERVAL)
    # check for screenshot, text dump
    text_dump = executor.display
    print("Dumping terminal display to ./terminal_executor_text_dump.txt")
    with open("./terminal_executor_text_dump.txt", "w+") as f:
        f.write(text_dump)
    print("Taking terminal screenshot at ./terminal_executor_screenshot.png")
    executor.screenshot("./terminal_executor_screenshot.png")
    print("Done")


def test():
    """
    Runs a test for the TerminalExecutor class by running a harmless command
    locally with bash and taking a screenshot and dumping the terminal display
    to a file.

    This test is useful for checking that the TerminalExecutor class works in
    a real environment.
    """
    test_harmless_command_locally_with_bash()


if __name__ == "__main__":
    test()
