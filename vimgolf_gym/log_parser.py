import json
import copy
import os
from abc import ABC, abstractmethod
from typing import Type

# we keep track of the log file attributes
# specifically, the file size.
# if the file size changes, we will reparse the log.
# in more advanced usage, we could seek to given point and parse from there, avoiding reparsing from the beginning.


class AbstractLogParser(ABC):
    def __init__(self): ...
    @abstractmethod
    def feed_line(self, line: str): ...


class LogWatcher:
    def __init__(self, log_file: str, parser_class: Type[AbstractLogParser]):
        self.log_file = log_file
        self.parser_class = parser_class
        self.parser = parser_class()
        self.last_filesize = 0
        self.last_position = 0  # Track the last read position

    def update(self, style="advanced"):
        # Use advanced update by default
        if style == "advanced":
            self.advanced_update()
        elif style == "simple":
            self.simple_update()
        elif style == "naive":
            self.naive_update()
        else:
            raise ValueError(
                "Unrecognized update option: %s (should be in advanced, simple, naive)"
                % style
            )

    def simple_update(self):
        current_size = (
            os.path.getsize(self.log_file) if os.path.exists(self.log_file) else 0
        )
        if current_size != self.last_filesize:
            self.naive_update()
            self.last_filesize = current_size

    def naive_update(self):
        self.parser = self.parser_class()
        if os.path.exists(self.log_file):
            with open(self.log_file, "r") as f:
                for line in f.readlines():
                    self.parser.feed_line(line)
        self.last_filesize = (
            os.path.getsize(self.log_file) if os.path.exists(self.log_file) else 0
        )
        self.last_position = self.last_filesize

    def advanced_update(self):
        if not os.path.exists(self.log_file):
            return

        current_size = os.path.getsize(self.log_file)

        # If file size decreased (file was truncated), do a full reset
        if current_size < self.last_filesize:
            self.naive_update()
            return

        # If file size increased, read only the new content
        if current_size > self.last_filesize:
            with open(self.log_file, "r") as f:
                f.seek(self.last_position)
                while True:
                    line = f.readline()
                    if not line:  # End of file
                        break
                    self.parser.feed_line(line)

            # Update tracking variables
            self.last_filesize = current_size
            self.last_position = f.tell()


class VimGolfLogWatcher(LogWatcher):
    def __init__(self, log_file: str):
        super().__init__(log_file=log_file, parser_class=VimGolfLogParser)
        self.parser: VimGolfLogParser



class VimGolfLogParser(AbstractLogParser):
    def __init__(self):
        self.results = []

    def feed_line(self, line: str):
        try:
            data = json.loads(line.strip())
            if type(data) == dict:
                if data.get("event_type", None) == "vimgolf_result":
                    event_data = data.get("event_data", None)
                    if type(event_data) == dict:
                        self.results.append(event_data)
        except json.JSONDecodeError:
            ...

    @property
    def success_results(self):
        return [it for it in self.results if it["correct"]]

    @property
    def success(self):
        return len(self.success_results) != 0

    def get_last_success_result(self):
        success_results = self.success_results
        if success_results:
            return success_results[-1]

    def get_best_success_result(self):
        """Return the result with the lowest score"""
        success_results = copy.deepcopy(self.success_results)
        if success_results:
            success_results.sort(key=lambda x: x["score"])
            return success_results[0]

