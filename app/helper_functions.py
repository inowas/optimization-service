from config import OPTIMIZATION_DATA, JSON_ENDING
import json
from pathlib import Path
from typing import Union
from copy import deepcopy
from sqlalchemy.ext.declarative import declarative_base
from typing import List


def create_input_and_output_filepath(folder: Union[str, Path],
                                     task_id: str,
                                     file_types: List[str]):
    if not Path(folder).is_dir():
        raise NotADirectoryError(f"Error: the folder {folder} does not exist.")

    if not Path(folder, task_id).is_dir():
        Path(folder, task_id).mkdir()

    # Create a filepath to that id
    return tuple(f"{str(Path(folder, task_id, file_type))}{JSON_ENDING}"
                 for file_type in file_types)


def load_json(filepath: Union[Path, str]) -> dict:
    with open(filepath, "r") as f:
        # Read json from it into data
        return json.load(f)


def write_json(obj: dict,
               filepath: Union[Path, str]) -> None:
    with open(filepath, "w") as f:
        json.dump(obj, f)


def get_table_for_optimization_id(table_class,
                                  optimization_id):
    base = declarative_base()

    # IndividualTaskTable.__tablename__ = f"{table_class.__name__.lower()}_{optimization_id}"

    class IndividualTaskTable(base, table_class):
        __tablename__ = f"{table_class.__name__.lower()}_{optimization_id}"

        def __init__(self, **args):
            super().__init__(**args)

    return IndividualTaskTable
