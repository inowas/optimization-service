from config import OPTIMIZATION_DATA, JSON_ENDING
import json
from pathlib import Path
from typing import Union


def create_input_and_output_filepath(task_id,
                                     extensions):
    # Create a filepath to that id
    return tuple(f"{OPTIMIZATION_DATA}{task_id}{extension}{JSON_ENDING}"
                 for extension in extensions)


def load_json(filepath: Union[Path, str]) -> dict:
    with open(filepath, "r") as f:
        # Read json from it into data
        return json.load(f)


def write_json(obj: dict,
               filepath: Union[Path, str]) -> None:
    with open(filepath, "w") as f:
        json.dump(obj, f)