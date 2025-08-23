"""
Log parser for VimGolf executor
"""

import copy
import json
import os
from abc import ABC, abstractmethod
from typing import Type

from vimgolf_gym.dataclasses import VimGolfEnvResult


class AbstractLogParser(ABC):
    def __init__(self): ...
    @abstractmethod
    def feed_line(self, line: str): ...


# We keep track of the log file attributes.
# Specifically, the file size.
# If the file size changes, we will reparse the log.
# In more advanced usage, we could seek to given point and parse from there, avoiding reparsing from the beginning.
class LogWatcher:
    def __init__(self, log_file: str, parser_class: Type[AbstractLogParser]):
        """
        Initialize the log watcher with a log file and a parser class.

        Args:
            log_file (str): The path to the log file.
            parser_class (Type[AbstractLogParser]): A class that implements the AbstractLogParser interface.

        Attributes:
            log_file (str): The path to the log file.
            parser_class (Type[AbstractLogParser]): The class of the parser.
            parser (AbstractLogParser): An instance of the parser class.
            last_filesize (int): The size of the log file when it was last checked.
            last_position (int): The last read position in the log file.
        """
        self.log_file = log_file
        self.parser_class = parser_class
        self.parser = parser_class()
        self.last_filesize = 0
        self.last_position = 0  # Track the last read position

    def update(self, style="advanced"):
        """
        Update the log watcher using one of three strategies.

        The ``advanced`` strategy is the default. It checks the file size
        and only reads new content if the file size has changed. If the file
        size is smaller than the last recorded size, it resets the parser and
        reads the entire file.

        The ``simple`` strategy reads the entire log file every time it is
        called. This is simple, but slow for large log files.

        The ``naive`` strategy reads the entire log file every time it is
        called, but it does not reset the parser. This is faster than the
        ``simple`` strategy, but it does not handle the case where the log
        file is truncated.

        Args:
            style (str, optional): The update strategy to use. Must be one of 
                ``advanced``, ``simple``, or ``naive``. Defaults to ``advanced``.
        """
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
        """
        Read the entire log file and reset the parser if the file size has changed.

        The ``simple`` strategy reads the entire log file every time it is
        called. This is simple, but slow for large log files.
        """
        current_size = (
            os.path.getsize(self.log_file) if os.path.exists(self.log_file) else 0
        )
        if current_size != self.last_filesize:
            self.naive_update()
            self.last_filesize = current_size

    def naive_update(self):
        """
        Reset the parser and read the entire log file.

        This is slow for large log files, but it is simple and handles
        the case where the log file is truncated.

        """
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
        """
        Read only the new content from the log file.

        The ``advanced`` strategy reads only the new content from the log file,
        starting from the last recorded position. This is fast for large log
        files, but it requires keeping track of the last recorded position.

        If the file size has decreased, it resets the parser and reads the
        entire file.
        """
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
    def __init__(self, log_file: str, update_style="advanced"):
        """
        Initialize a VimGolfLogWatcher with a log file and an update style.

        The VimGolfLogWatcher is a LogWatcher that is specialized for
        reading VimGolf logs.

        Args:
            log_file (str): The path to the log file.
            update_style (str, optional): The update strategy to use. Must be one of 
                ``advanced``, ``simple``, or ``naive``. Defaults to ``advanced``.

        Returns:
            [return_type]: [description of return value]
        """
        super().__init__(log_file=log_file, parser_class=VimGolfLogParser)
        self.parser: VimGolfLogParser
        self.update_style = update_style

    def default_update(self):
        """
        Call the update method with the default update style.

        This method is a convenience wrapper around the update method,
        calling it with the default update style specified in the
        constructor.

        See the update method for more details.
        """
        self.update(style=self.update_style)

    @property
    def success(self):
        """
        Check if the vimgolf challenge has been solved successfully.

        This property checks if the vimgolf challenge has been solved
        successfully. It updates the log watcher if necessary before
        checking the success status.

        Returns:
            bool: True if the challenge has been solved successfully, False otherwise.
        """
        self.default_update()
        return self.parser.success

    def get_best_success_result(self):
        """
        Return the best success result in the log watcher.

        This method returns the best success result in the log watcher,
        updating the log watcher if necessary before returning the result.

        Returns:
            VimGolfEnvResult: The best success result in the log watcher, or None if 
            there is no success result.
        """
        self.default_update()
        return self.parser.get_best_success_result()

    def get_last_success_result(self):
        """
        Return the last success result in the log watcher.

        This method returns the last success result in the log watcher,
        updating the log watcher if necessary before returning the result.

        Returns:
            VimGolfEnvResult: The last success result in the log watcher, or None if 
            there is no success result.
        """
        self.default_update()
        return self.parser.get_last_success_result()

    @property
    def results(self):
        """
        The results of the vimgolf challenge environment.

        This property returns the results of the vimgolf challenge environment,
        updating the log watcher if necessary before returning the results.

        Returns:
            list[VimGolfEnvResult]: The results of the vimgolf challenge environment, or an empty list if
            there are no results.
        """
        self.default_update()
        return self.parser.results

    @property
    def success_results(self):
        """
        The successful results of the vimgolf challenge environment.

        This property returns the successful results of the vimgolf challenge
        environment, updating the log watcher if necessary before returning the
        results.

        Returns:
            list[VimGolfEnvResult]: The successful results of the vimgolf challenge environment, or an
            empty list if there are no successful results.
        """
        self.default_update()
        return self.parser.success_results


class VimGolfLogParser(AbstractLogParser):
    def __init__(self):
        """
        Initialize the VimGolfLogParser.

        The VimGolfLogParser is initialized with an empty list of results.
        """
        self.results: list[VimGolfEnvResult] = []

    def feed_line(self, line: str):
        """
        Feed a line to the parser.

        The line should be a JSON-formatted string. The parser will attempt to
        parse the line as a JSON object, and if it is a dictionary, it will
        check if the dictionary has an "event_type" key with value
        "vimgolf_result". If so, it will attempt to parse the value of the
        "event_data" key as a VimGolfEnvResult object and append it to the
        results list.

        If the line is not a valid JSON object, or if the JSON object does not
        have the correct structure, the line will be ignored.

        Args:
            line (str): The line of text to feed to the parser.
        """
        try:
            data = json.loads(line.strip())
            if type(data) == dict:
                if data.get("event_type", None) == "vimgolf_result":
                    event_data = data.get("event_data", None)
                    if type(event_data) == dict:
                        parsed_result = VimGolfEnvResult.parse_obj(event_data)
                        self.results.append(parsed_result)
        except json.JSONDecodeError:
            ...

    @property
    def success_results(self):
        """
        The successful results of the vimgolf challenge environment.

        This property returns the successful results of the vimgolf challenge
        environment, which are the results in the results list where the
        correct attribute is True.

        Returns:
            list[VimGolfEnvResult]: The successful results of the vimgolf challenge environment, or an
            empty list if there are no successful results.
        """
        return [it for it in self.results if it.correct]

    @property
    def success(self):
        """
        Check if the vimgolf challenge has been solved successfully.

        This property checks if the vimgolf challenge has been solved
        successfully. It returns True if there are any successful results in
        the results list, and False otherwise.

        Returns:
            bool: True if the challenge has been solved successfully, False otherwise.
        """
        return len(self.success_results) != 0

    def get_last_success_result(self):
        """
        Return the last success result in the log watcher.

        This method returns the last success result in the log watcher,
        which is the last result in the success_results list.

        Returns:
            VimGolfEnvResult: The last success result in the log watcher, or None if there is no
            success result.
        """
        success_results = self.success_results
        if success_results:
            return success_results[-1]

    def get_best_success_result(self):
        """Return the result with the lowest score"""
        success_results = copy.deepcopy(self.success_results)
        if success_results:
            success_results.sort(key=lambda x: x.score)
            return success_results[0]
