from config import OPTIMIZATION_DATA, JSON_ENDING
import json
from pathlib import Path
from typing import Union
from copy import deepcopy
from sqlalchemy.ext.declarative import declarative_base


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


def get_table_for_optimization_id(table_class,
                                  optimization_id):
    Base = declarative_base()

    # IndividualTaskTable.__tablename__ = f"{table_class.__name__.lower()}_{optimization_id}"

    class IndividualTaskTable(Base, table_class):
        __tablename__ = f"{table_class.__name__.lower()}_{optimization_id}"

        def __init__(self, **args):
            super().__init__(**args)

    return IndividualTaskTable
