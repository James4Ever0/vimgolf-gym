"""
Vimgolf API wrapper
"""

import argparse
import json
import logging
import typing
from vimgolf_gym._vimrc import (
    _CYBERGOD_VIMGOLF_VIMRC_FILEPATH,
    _prepare_cybergod_vimrc_with_buffer_file,
    _VIMGOLF_VIMRC_FILEPATH,
)

from vimgolf.vimgolf import (
    IGNORED_KEYSTROKES,
    Challenge,
    Result,
    Status,
    VimRunner,
    filecmp,
    format_,
    get_challenge_url,
    get_keycode_repr,
    input_loop,
    logger,
    os,
    parse_keycodes,
    sys,
    tempfile,
    tokenize_keycode_reprs,
    upload_result,
    working_directory,
    write,
)


def parse_commandline_arguments():
    """
    Parse the command line arguments using argparse.

    Args:
        None

    Returns:
        A Namespace object with the parsed arguments.

    The parsed arguments are:
        - --input_file: str, the path to the input file
        - --output_file: str, the path to the output file
        - --log_file: str, the path to the log file
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--log_file", type=str, required=True)
    parser.add_argument("--buffer_file", type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    """
    Parse command line arguments and execute the vimgolf local challenge.

    It sets up the logging to a file and then calls the local function to execute
    the challenge.

    The local function is called with the input file, output file and an empty
    string as the init_keys argument.

    """
    args = parse_commandline_arguments()
    log_file = args.log_file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    if args.buffer_file:
        local(
            infile=args.input_file,
            outfile=args.output_file,
            buffer_file=args.buffer_file,
        )
    else:
        local(infile=args.input_file, outfile=args.output_file)


def local(
    infile: str, outfile: str, buffer_file: typing.Optional[str] = None, init_keys=""
):
    """
    Execute a local VimGolf challenge.

    It reads the input file, output file, and runs the challenge. The challenge
    is defined by the input and output text and the file extensions.

    Args:
        infile: str, the path to the input file
        outfile: str, the path to the output file
        init_keys: str, the initial keys to type into Vim
        buffer_file: typing.Optional[str]: Where to write the vim editor buffer. Defaults to None.

    Returns:
        Status: The status of the challenge
    """
    logger.info("local(%s, %s)", infile, outfile)
    with open(infile, "r") as f:
        in_text = format_(f.read())
    with open(outfile, "r") as f:
        out_text = format_(f.read())
    _, in_extension = os.path.splitext(infile)
    _, out_extension = os.path.splitext(outfile)
    challenge = Challenge(
        in_text=in_text,
        out_text=out_text,
        in_extension=in_extension,
        out_extension=out_extension,
        id=None,
        compliant=None,
        api_key=None,
        init_keys=init_keys,
    )
    status = play(challenge, buffer_file=buffer_file)
    return status


def play(challenge, results=None, buffer_file: typing.Optional[str] = None):
    """
    Execute a VimGolf challenge.

    Args:
        challenge: Challenge, the challenge to play
        results: list of Result, the results of previous plays
        buffer_file: typing.Optional[str]: Where to write the vim editor buffer. Defaults to None.

    Returns:
        Status: The status of the challenge
    """
    if results is None:
        results = []
    logger.info("play(...)")
    with tempfile.TemporaryDirectory() as workspace, working_directory(workspace):
        infile = "in"
        if challenge.in_extension:
            infile += challenge.in_extension
        outfile = "out"
        if challenge.out_extension:
            outfile += challenge.out_extension
        logfile = "log"
        with open(outfile, "w") as f:
            f.write(challenge.out_text)

        try:
            # If there were init keys specified, we need to convert them to a
            # form suitable for feedkeys(). We can't use Vim's -s option since
            # it takes escape codes, not key codes. See Vim #4041 and TODO.txt
            # ("Bug: script written with "-W scriptout" contains Key codes,
            # while the script read with "-s scriptin" expects escape codes").
            # The conversion is conducted here so that we can fail fast on
            # error (prior to playing) and to avoid repeated computation.
            keycode_reprs = tokenize_keycode_reprs(challenge.init_keys)
            init_feedkeys = []
            for item in keycode_reprs:
                if item == "\\":
                    item = "\\\\"  # Replace '\' with '\\'
                elif item == '"':
                    item = '\\"'  # Replace '"' with '\"'
                elif item.startswith("<") and item.endswith(">"):
                    item = "\\" + item  # Escape special keys ("<left>" -> "\<left>")
                init_feedkeys.append(item)
            init_feedkeys = "".join(init_feedkeys)
        except Exception:
            logger.exception("invalid init keys")
            write("Invalid keys: {}".format(challenge.init_keys), color="red")
            return Status.FAILURE

        write("Launching vimgolf session", color="yellow")
        while True:
            with open(infile, "w") as f:
                f.write(challenge.in_text)
            with open(outfile, 'w') as f:
                f.write(challenge.out_text)
            if buffer_file:
                _prepare_cybergod_vimrc_with_buffer_file(buffer_file)
                vimrc = _CYBERGOD_VIMGOLF_VIMRC_FILEPATH
            else:
                vimrc = _VIMGOLF_VIMRC_FILEPATH
            play_args = [
                "-Z",  # restricted mode, utilities not allowed
                "-n",  # no swap file, memory only editing
                "--noplugin",  # no plugins
                "-i",
                "NONE",  # don't load .viminfo (e.g., has saved macros, etc.)
                "+0",  # start on line 0
                "-u",
                vimrc,  # vimgolf .vimrc
                "-U",
                "NONE",  # don't load .gvimrc
                "-W",
                logfile,  # keylog file (overwrites existing)
                '+call feedkeys("{}", "t")'.format(init_feedkeys),  # initial keys
                infile,
            ]
            if VimRunner.run(play_args) != Status.SUCCESS:
                return Status.FAILURE

            correct = filecmp.cmp(infile, outfile, shallow=False)
            logger.info("correct: %s", str(correct).lower())
            with open(logfile, "rb") as _f:
                # raw keypress representation saved by vim's -w
                raw_keys = _f.read()

            # list of parsed keycode byte strings
            keycodes = parse_keycodes(raw_keys)
            keycodes = [
                keycode for keycode in keycodes if keycode not in IGNORED_KEYSTROKES
            ]

            # list of human-readable key strings
            keycode_reprs = [get_keycode_repr(keycode) for keycode in keycodes]
            logger.info("keys: %s", "".join(keycode_reprs))

            score = len(keycodes)
            logger.info("score: %d", score)

            result = Result(correct=correct, keys="".join(keycode_reprs), score=score)
            logger.info(
                json.dumps(
                    dict(
                        event_type="vimgolf_result",
                        event_data=dict(
                            correct=correct, keys="".join(keycode_reprs), score=score
                        ),
                    )
                )
            )
            results.append(result)

            write("Here are your keystrokes:", color="green")
            for keycode_repr in keycode_reprs:
                color = "magenta" if len(keycode_repr) > 1 else None
                write(keycode_repr, color=color, end=None)
            write("")

            if correct:
                write("Success! Your output matches.", color="green")
                write("Your score:", color="green")
            else:
                write(
                    "Uh oh, looks like your entry does not match the desired output.",
                    color="red",
                )
                write("Your score for this failed attempt:", color="red")
            write(score)

            upload_eligible = challenge.id and challenge.compliant and challenge.api_key

            while True:
                # Generate the menu items inside the loop since it can change across iterations
                # (e.g., upload option can be removed)
                with open(infile, 'w') as f:
                    f.write(challenge.in_text)
                with open(outfile, 'w') as f:
                    f.write(challenge.out_text)
                menu = []
                if not correct:
                    menu.append(("d", "Show diff"))
                if upload_eligible and correct:
                    menu.append(("w", "Upload result"))
                menu.append(("r", "Retry the current challenge"))
                menu.append(("q", "Quit vimgolf"))
                valid_codes = [x[0] for x in menu]
                for option in menu:
                    write("[{}] {}".format(*option), color="yellow")
                selection = input_loop("Choice> ")
                if selection not in valid_codes:
                    write(
                        "Invalid selection: {}".format(selection),
                        stream=sys.stderr,
                        color="red",
                    )
                elif selection == "d":
                    # diffsplit is used instead of 'vim -d' to avoid the "2 files to edit" message.
                    diff_args = [
                        "-n",
                        outfile,
                        "-c",
                        "vertical diffsplit {}".format(infile),
                    ]
                    if VimRunner.run(diff_args) != Status.SUCCESS:
                        return Status.FAILURE
                elif selection == "w":
                    upload_status = upload_result(
                        challenge.id, challenge.api_key, raw_keys
                    )
                    if upload_status == Status.SUCCESS:
                        write("Uploaded entry!", color="green")
                        leaderboard_url = get_challenge_url(challenge.id)
                        write(
                            "View the leaderboard: {}".format(leaderboard_url),
                            color="green",
                        )
                        upload_eligible = False
                    else:
                        write(
                            "The entry upload has failed",
                            stream=sys.stderr,
                            color="red",
                        )
                        message = "Please check your API key on vimgolf.com"
                        write(message, stream=sys.stderr, color="red")
                else:
                    break
            if selection == "q":
                break
            write("Retrying vimgolf challenge", color="yellow")

        write("Thanks for playing!", color="green")
        return Status.SUCCESS


if __name__ == "__main__":
    main()
