import json
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import pyarrow as pa
import pyarrow.json as paj

import datasets


@dataclass
class JsonConfig(datasets.BuilderConfig):
    """BuilderConfig for JSON."""

    features: Optional[datasets.Features] = None
    use_threads: bool = True
    block_size: Optional[int] = None
    newlines_in_values: Optional[bool] = None

    @property
    def pa_read_options(self):
        return paj.ReadOptions(use_threads=self.use_threads, block_size=self.block_size)

    @property
    def pa_parse_options(self):
        return paj.ParseOptions(newlines_in_values=self.newlines_in_values)

    @property
    def schema(self):
        return pa.schema(self.features.type) if self.features is not None else None


class Json(datasets.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = JsonConfig

    def _info(self):
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        """We handle string, list and dicts in datafiles"""
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        data_files = dl_manager.download_and_extract(self.config.data_files)
        if isinstance(data_files, (str, list, tuple)):
            files = data_files
            if isinstance(files, str):
                files = [files]
            return [datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"files": files})]
        splits = []
        for split_name, files in data_files.items():
            if isinstance(files, str):
                files = [files]
            splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files}))
        return splits

    def _generate_tables(self, files):
        for i, file in enumerate(files):
            with open(file, encoding="utf-8") as f:
                dataset = json.load(f)
            # We accept two format: a list of dicts or a dict of lists
            if isinstance(dataset, (list, tuple)):
                pa_table = paj.read_json(
                    BytesIO("\n".join(json.dumps(row) for row in dataset).encode("utf-8")),
                    read_options=self.config.pa_read_options,
                    parse_options=self.config.pa_parse_options,
                )
            else:
                pa_table = pa.Table.from_pydict(mapping=dataset)

            if self.config.schema:
                # Cast allows str <-> int/float, while parse_option explicit_schema does NOT
                pa_table = pa_table.cast(self.config.schema)
            yield i, pa_table