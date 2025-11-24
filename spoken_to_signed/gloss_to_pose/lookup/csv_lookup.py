import csv
import os
from typing import Dict

from .lookup import PoseLookup


class CSVPoseLookup(PoseLookup):
    def __init__(self, source_path: str, backup: PoseLookup = None, pose_path_builder=None, priority_mode: str = "shortest"):
        if not source_path:
            raise ValueError("Lexicon path must not be empty")

        source_path = os.path.expanduser(source_path)
        base_directory = None
        csv_path = source_path

        if os.path.isdir(source_path):
            candidate = os.path.join(source_path, 'index.csv')
            if not os.path.exists(candidate):
                raise ValueError(f"Could not find index.csv inside directory '{source_path}'")
            csv_path = candidate
            base_directory = source_path
        elif not os.path.isfile(source_path):
            raise ValueError(f"Lexicon file '{source_path}' does not exist")

        with open(csv_path, mode='r', encoding='utf-8') as f:
            rows = [dict(row) for row in csv.DictReader(f)]

        if pose_path_builder is not None:
            rows = [pose_path_builder(dict(row)) for row in rows]
            base_directory = None

        super().__init__(rows=rows, directory=base_directory, backup=backup, priority_mode=priority_mode)
