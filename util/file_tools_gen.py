"""
MIT License

Copyright (c) 2025 taro nakano

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import json
import csv
# import yaml
import logging
import re
import xml.etree.ElementTree as XmlEt
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Iterator, Iterable, Union, NamedTuple, Optional
from collections import OrderedDict


# --- 1. Unified Exception ---
class FileProcessingError(Exception):
    """Base exception for all file operation and parsing errors in this library."""
    pass


# --- 2. Domain Models ---
class Size(NamedTuple):
    width: int
    height: int
    depth: int


class Bndbox(NamedTuple):
    xmin: float
    ymin: float
    xmax: float
    ymax: float


class VocXmlDataset(NamedTuple):
    size: Size
    objects: List[tuple]  # List of (label_name, Bndbox)


# --- 3. Logging Setup ---
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# --- 4. Format Handlers ---

class FileHandler(ABC):
    """Abstract base class for file operations with common validation logic."""

    def __init__(self, encoding: str = 'utf-8'):
        self.encoding = encoding

    @abstractmethod
    def read(self, file_path: Union[str, Path]) -> Any:
        pass

    @abstractmethod
    def write(self, data: Any, file_path: Union[str, Path]) -> None:
        pass

    def _safe_open(self, path: Path, mode: str, newline: str = None):
        """Opens a file and wraps OS-level exceptions into FileProcessingError."""
        try:
            return path.open(mode, encoding=self.encoding, newline=newline)
        except Exception as e:
            raise FileProcessingError(f"IO Error at '{path.as_posix()}': {e}") from e

    def _validate_suffix(self, file_path: Path, valid_suffixes: List[str]) -> None:
        """Validates file extensions (case-insensitive)."""
        allowed = [s.lower() if s.startswith('.') else f'.{s.lower()}' for s in valid_suffixes]
        if file_path.suffix.lower() not in allowed:
            raise FileProcessingError(
                f"Invalid extension: '{file_path.suffix}'. Expected one of: {valid_suffixes}"
            )


class JsonHandler(FileHandler):
    def read(self, file_name: Union[str, Path]) -> Any:
        path = Path(file_name)
        with self._safe_open(path, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                raise FileProcessingError(f"JSON Decode Error in {path.name}: {e}") from e

    def write(self, data: Any, file_name: Union[str, Path], indent: int = 2) -> None:
        path = Path(file_name)
        with self._safe_open(path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)


class JsonlHandler(FileHandler):
    """Handles JSON Lines format with optional fault tolerance."""

    def read(self, file_name: Union[str, Path], ignore_errors: bool = False) -> Iterator[Dict]:
        path = Path(file_name)
        with self._safe_open(path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    yield json.loads(clean_line)
                except json.JSONDecodeError as e:
                    if ignore_errors:
                        logger.warning(f"Skipping malformed JSON at {path.name}:{line_num}: {e}")
                        continue
                    raise FileProcessingError(f"JSONL Decode Error at line {line_num}: {e}") from e

    def write(self, data: Iterable[Dict], file_name: Union[str, Path]) -> None:
        path = Path(file_name)
        with self._safe_open(path, 'w') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class CsvHandler(FileHandler):
    def read(self, file_path: Union[str, Path]) -> Iterator[List[str]]:
        path = Path(file_path)
        with self._safe_open(path, 'r', newline='') as f:
            yield from csv.reader(f)

    def write(self, data: Iterable[Iterable[Any]], file_path: Union[str, Path]) -> None:
        path = Path(file_path)
        with self._safe_open(path, 'w', newline='') as f:
            csv.writer(f).writerows(data)


# class YamlHandler(FileHandler):
#     def read(self, file_path: Union[str, Path]) -> Any:
#         path = Path(file_path)
#         with self._safe_open(path, 'r') as f:
#             return yaml.safe_load(f)
#
#     def write(self, data: Any, file_path: Union[str, Path]) -> None:
#         path = Path(file_path)
#         with self._safe_open(path, 'w') as f:
#             yaml.safe_dump(data, f, allow_unicode=True, default_flow_style=False)


class TxtHandler(FileHandler):
    def read(self, file_path: Union[str, Path]) -> Iterator[str]:
        path = Path(file_path)
        with self._safe_open(path, 'r') as f:
            for line in f:
                yield line.rstrip('\n\r')

    def write(self, data: Iterable[str], file_path: Union[str, Path]) -> None:
        path = Path(file_path)
        with self._safe_open(path, 'w') as f:
            for line in data:
                f.write(f"{line}\n")


class XmlHandler(FileHandler):
    def read(self, file_path: Union[str, Path]) -> XmlEt.Element:
        path = Path(file_path)
        self._validate_suffix(path, ['.xml'])
        try:
            with self._safe_open(path, 'r') as f:
                return XmlEt.parse(f).getroot()
        except XmlEt.ParseError as e:
            raise FileProcessingError(f"XML Parse Error in {path.name}: {e}") from e

    def write(self, data: XmlEt.Element, file_path: Union[str, Path], pretty_print: bool = True) -> None:
        path = Path(file_path)
        if pretty_print:
            XmlEt.indent(data, space="  ", level=0)
        try:
            XmlEt.ElementTree(data).write(path, encoding=self.encoding, xml_declaration=True)
        except Exception as e:
            raise FileProcessingError(f"Failed to write XML to {path.name}: {e}") from e


# --- 5. Domain Analyzers ---

class VocXmlAnalyzer:
    """Analyzer for Pascal VOC XML format with strict schema validation."""

    def __init__(self, handler: Optional[XmlHandler] = None):
        self._handler = handler or XmlHandler()

    def _safe_cast_value(self, element: Optional[XmlEt.Element], tag: str, cast_type: type, default: Any = 0) -> Any:
        """Helper to safely extract and cast XML text content, handling empty tags and malformed strings."""
        if element is None:
            return default
        text = element.findtext(tag)
        if text is None or text.strip() == "":
            return default
        try:
            return cast_type(text.strip())
        except (ValueError, TypeError):
            return default

    def read(self, file_path: Union[str, Path]) -> VocXmlDataset:
        root = self._handler.read(file_path)
        try:
            xml_size = root.find('size')
            if xml_size is None:
                raise FileProcessingError("Required tag <size> is missing.")

            size_obj = Size(
                width=self._safe_cast_value(xml_size, 'width', int),
                height=self._safe_cast_value(xml_size, 'height', int),
                depth=self._safe_cast_value(xml_size, 'depth', int)
            )

            objects = []
            for obj in root.findall('object'):
                name = obj.findtext('name', 'unknown')
                bnd = obj.find('bndbox')
                if bnd is not None:
                    bbox = Bndbox(
                        xmin=self._safe_cast_value(bnd, 'xmin', float),
                        ymin=self._safe_cast_value(bnd, 'ymin', float),
                        xmax=self._safe_cast_value(bnd, 'xmax', float),
                        ymax=self._safe_cast_value(bnd, 'ymax', float)
                    )
                    objects.append((name, bbox))
            return VocXmlDataset(size=size_obj, objects=objects)
        except Exception as e:
            if isinstance(e, FileProcessingError): raise
            raise FileProcessingError(f"Schema validation failed for '{Path(file_path).name}': {e}") from e


class PbtxtAnalyzer:
    """
    Robust parser for Protobuf Text Format (label_map.pbtxt).
    Uses regex to handle varied spacing, quotes, and block structures.
    """
    # Regex to capture key-value pairs like id: 1 or name: "class_a"
    _KV_PATTERN = re.compile(r'(\w+)\s*:\s*["\']?([^"\']+)["\']?')

    def __init__(self, handler: Optional[TxtHandler] = None):
        self._handler = handler or TxtHandler()

    def read(self, file_path: Union[str, Path]) -> OrderedDict:
        path = Path(file_path)
        items = OrderedDict()
        current_id: Optional[int] = None
        current_name: Optional[str] = None

        # Parse line by line to maintain low memory footprint
        for line in self._handler.read(path):
            line = line.strip()
            # Ignore comments and block braces
            if line.startswith('#') or line in ('item {', '}'):
                continue

            match = self._KV_PATTERN.search(line)
            if match:
                key, value = match.groups()
                if key == 'id':
                    try:
                        current_id = int(value)
                    except ValueError:
                        continue
                elif key == 'name':
                    current_name = value

            # Once both id and name are found, commit to dictionary
            if current_id is not None and current_name is not None:
                items[current_name] = current_id
                current_id, current_name = None, None

        return items


if __name__ == "__main__":
    """
    python -m util.file_tools
    """

    from logging import DEBUG, INFO, basicConfig

    basicConfig(level=INFO)

    ''' csv read and write '''
    csv_h = CsvHandler()
    csv_file = "./data/sample_write.csv"

    data_list = [
        ["hoge", 3],
        ["fuga", 4],
    ]

    csv_h.write(data_list, csv_file)
    csv_data = csv_h.read(csv_file)

    print(list(csv_data))

    """ test for jsonl """
    jl_h = JsonlHandler()
    jl_file = "./data/sample_write.jsonl"

    dict_list = [
        {
            "id": "task_10",
            "instruction": "朝食",
        },
        {
            "id": "task_11",
            "instruction": "ビル",
        },
        {
            "id": "task_12",
            "instruction": "人物",
        }
    ]

    jl_h.write(dict_list, file_name=jl_file)

    jl_data = jl_h.read(jl_file)
    print(list(jl_data))

    """ test for json """
    j_h = JsonHandler()
    j_file = "./data/sample_write.json"

    dict_data = {"data": dict_list}

    j_h.write(dict_data, file_name=j_file)

    j_data = j_h.read(j_file)
    print(j_data)


