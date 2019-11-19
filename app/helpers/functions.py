from typing import Union, List
from uuid import uuid4
import json
from jsonschema import RefResolver
from pathlib import Path, PurePosixPath
from urllib.request import urlopen
from sqlalchemy.ext.declarative import declarative_base

from app.helpers.config import HTTPS_STRING


def create_input_and_output_filepath(folder: Union[str, Path],
                                     task_id: Union[str, uuid4],
                                     file_name: List[str] = None,
                                     file_type: str = ""):
    if not Path(folder, task_id).is_dir():
        raise NotADirectoryError(f"Error: the folder {folder} does not exist.")

    # Create a filepath to that id
    if not file_name:
        return str(Path(folder, task_id))

    return tuple(f"{str(Path(folder, task_id, file))}{file_type}"
                 for file in file_name)


def load_json(filepath: Union[Path, str]) -> dict:
    with open(filepath, "r") as f:
        # Read json from it into data
        return json.load(f)


def write_json(obj: dict,
               filepath: Union[Path, str]) -> None:
    with open(filepath, "w") as f:
        json.dump(obj, f)


def get_table_for_optimization_id(table_class,
                                  optimization_id: str):
    base = declarative_base()

    # IndividualTaskTable.__tablename__ = f"{table_class.__name__.lower()}_{optimization_id}"

    class IndividualTaskTable(base, table_class):
        __tablename__ = f"{table_class.__name__.lower()}_{optimization_id}"

        def __init__(self, **args):
            super().__init__(**args)

    return IndividualTaskTable


def get_schema_and_refresolver(schema_path):
    assert isinstance(schema_path, PurePosixPath), \
        f"Error: schema_path requires a PurePosixPath, not {type(schema_path)}."

    with urlopen(f"{HTTPS_STRING}{schema_path}") as f:
        schema_data = json.load(f)

    refresolver = RefResolver(f"{HTTPS_STRING}{schema_path}",
                              referrer=schema_data)

    return schema_data, refresolver
