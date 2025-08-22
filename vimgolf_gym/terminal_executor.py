# code from: https://github.com/James4Ever0/agi_computer_control/blob/master/web_gui_terminal_recorder/executor_and_replayer/terminal_asciicast_record_executor.py (modified)

# remove pyte dependency

import ptyprocess
import time
import threading
import agg_python_bindings
import pydantic

class Cursor(pydantic.BaseModel):
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


class AvtScreen:
    def __init__(self, width: int, height: int):
        self.vt = agg_python_bindings.TerminalEmulator(width, height)

    def write_bytes(self, _bytes: bytes):
        decoded_bytes = decode_bytes_to_utf8_safe(_bytes)
        self.vt.feed_str(decoded_bytes)

    @property
    def cursor(self):
        col, row, visible = self.vt.get_cursor()
        ret = Cursor(x=col, y=row, hidden=not visible)
        return ret

    @property
    def display(self):
        ret = "\n".join(self.vt.text_raw())
        return ret

    def screenshot(self, png_output_path: str):
        self.vt.screenshot(png_output_path)
    
    def close(self):
        del self.vt


class TerminalProcess:
    def __init__(self, command: list[str], width: int, height: int, backend="avt"):
        """
        Initializes the terminal emulator with a command to execute.
        Args:
            command: List of command strings to execute in the terminal
        """
        rows, cols = height, width
        self.pty_process = ptyprocess.PtyProcess.spawn(command, dimensions=(rows, cols))
        if backend == "avt":
            self.vt_screen = AvtScreen(width=width, height=height)
        else:
            raise ValueError(
                "Unknown terminal emulator backend '%s' (known ones: avt, pyte)"
                % backend
            )
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
        self.pty_process.close()
        self.vt_screen.close()
        self.__pty_process_reading_thread.join()

    def __read_and_update_screen(self, poll_interval=0.01):
        """Reads available output from terminal and updates Pyte screen"""
        while True:
            try:
                # ptyprocess.read is blocking. only pexpect has read_nonblocking
                process_output_bytes = self.pty_process.read(1024)
                # write bytes to pyte screen
                self.vt_screen.write_bytes(process_output_bytes)
            except KeyboardInterrupt: # user interrupted
                break
            except SystemExit: # python process exit
                break
            except SystemError: # python error
                break
            except EOFError: # terminal died
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
            command: List of command strings to execute
        """
        self.terminal = TerminalProcess(
            command=command, width=width, height=height
        )

    def input(self, text: str):
        """
        Sends input text to the terminal process
        """
        self.terminal.write(text.encode())
        # Allow time for processing output
        time.sleep(0.1)

    @property
    def display(self) -> str:
        return self.terminal.vt_screen.display

    def screenshot(self, png_save_path: str):
        self.terminal.vt_screen.screenshot(png_save_path)
    
    def close(self):
        self.terminal.close()


def test_harmless_command_locally_with_bash():
    SLEEP_INTERVAL = 0.5
    command = ["docker", "run" , "--rm", "-it", "alpine"]
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
    test_harmless_command_locally_with_bash()

if __name__ == "__main__":
    test()