"""

VimGolfGym: A gym environment for VimGolf challenges.
"""

import os
from vimgolf_gym.lib import (
    get_local_challenge_definition,
    get_local_challenge_metadata,
    get_local_challenge_worst_solution,
    get_local_challenge_worst_solution_header,
    list_local_challenge_ids,
    make,
)

if os.environ.get("PDOC_PROCESS"):
    from vimgolf_gym import dataclasses
    from vimgolf_gym import lib
    from vimgolf_gym import log_parser
    from vimgolf_gym import terminal_executor
    from vimgolf_gym import vimgolf

    __all__ = [
        "make",
        "list_local_challenge_ids",
        "get_local_challenge_definition",
        "get_local_challenge_metadata",
        "get_local_challenge_worst_solution",
        "get_local_challenge_worst_solution_header",
        "dataclasses",
        "lib",
        "log_parser",
        "terminal_executor",
        "vimgolf",
    ]
else:

    __all__ = [
        "make",
        "list_local_challenge_ids",
        "get_local_challenge_definition",
        "get_local_challenge_metadata",
        "get_local_challenge_worst_solution",
        "get_local_challenge_worst_solution_header",
    ]
