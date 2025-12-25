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
import yaml
import logging
import re
import xml.etree.ElementTree as XmlEt
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Iterator, Iterable, Union, NamedTuple, Optional
from collections import OrderedDict
from contextlib import contextmanager

# --- Logging Setup ---
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# --- Unified Exception ---
class FileProcessingError(Exception):
    """Base exception for all file operation and parsing errors in this library."""
    pass


# --- Format Handlers ---

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

    @contextmanager
    def _atomic_write_path(self, file_path: Path):
        """Context manager that writes to a temporary file and replaces only if successful."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            yield temp_path  # For the purpose of temporary transfer within the with statement using yield.
            temp_path.replace(path)
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise FileProcessingError(f"Atomic write failed for {path}: {e}") from e

    @staticmethod
    def _validate_suffix(file_path: Path, valid_suffixes: List[str]) -> None:
        """
        Validates file extensions (case-insensitive).
        This is a static method as it does not depend on instance state.
        """
        allowed = [s.lower() if s.startswith('.') else f'.{s.lower()}' for s in valid_suffixes]
        if file_path.suffix.lower() not in allowed:
            raise FileProcessingError(
                f"Invalid extension: '{file_path.suffix}'. Expected one of: {valid_suffixes}"
            )


class JsonHandler(FileHandler):
    def read(self, file_path: Union[str, Path]) -> Any:
        path = Path(file_path)
        with self._safe_open(path, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                raise FileProcessingError(f"JSON Decode Error in {path.name}: {e}")

    def write(self, data: Any, file_path: Union[str, Path], indent: int = 2) -> None:
        path = Path(file_path)
        with self._atomic_write_path(path) as tmp:
            with tmp.open('w', encoding=self.encoding) as f:
                json.dump(data, f, ensure_ascii=False, indent=indent)


class JsonlHandler(FileHandler):
    def read(self, file_path: Union[str, Path]) -> List[Dict]:
        path = Path(file_path)
        results = []
        with self._safe_open(path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                clean_line = line.strip()
                if not clean_line: continue
                try:
                    results.append(json.loads(clean_line))
                except json.JSONDecodeError as e:
                    raise FileProcessingError(f"JSONL Error at {path.name}:{line_num}: {e}")
        return results

    def write(self, data: Iterable[Dict], file_path: Union[str, Path]) -> None:
        path = Path(file_path)
        with self._atomic_write_path(path) as tmp:
            with tmp.open('w', encoding=self.encoding) as f:
                for entry in data:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class CsvHandler(FileHandler):
    def read(self, file_path: Union[str, Path]) -> List[List[str]]:
        path = Path(file_path)
        with self._safe_open(path, 'r', newline='') as f:
            return list(csv.reader(f))

    def write(self, data: Iterable[Iterable[Any]], file_path: Union[str, Path]) -> None:
        path = Path(file_path)
        with self._atomic_write_path(path) as tmp:
            with tmp.open('w', encoding=self.encoding, newline='') as f:
                csv.writer(f).writerows(data)


class YamlHandler(FileHandler):
    def __init__(self, encoding: str = 'utf-8', allow_unicode: bool = True, sort_keys: bool = False):
        super().__init__(encoding)
        self.allow_unicode = allow_unicode
        self.sort_keys = sort_keys

    def read(self, file_path: Union[str, Path]) -> Any:
        path = Path(file_path)
        with self._safe_open(path, 'r') as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise FileProcessingError(f"YAML Load Error in {path.name}: {e}")

    def write(self, data: Any, file_path: Union[str, Path]) -> None:
        path = Path(file_path)
        with self._atomic_write_path(path) as tmp:
            with tmp.open('w', encoding=self.encoding) as f:
                yaml.safe_dump(data, f, allow_unicode=self.allow_unicode,
                               sort_keys=self.sort_keys, default_flow_style=False)


class TxtHandler(FileHandler):
    def read(self, file_path: Union[str, Path]) -> str:
        path = Path(file_path)
        with self._safe_open(path, 'r') as f:
            return f.read()

    def write(self, data: str, file_path: Union[str, Path]) -> None:
        path = Path(file_path)
        with self._atomic_write_path(path) as tmp:
            with tmp.open('w', encoding=self.encoding) as f:
                f.write(data)


class XmlHandler(FileHandler):
    def read(self, file_path: Union[str, Path]) -> XmlEt.Element:
        path = Path(file_path)
        self._validate_suffix(path, ['.xml'])
        try:
            with self._safe_open(path, 'r') as f:
                return XmlEt.parse(f).getroot()
        except XmlEt.ParseError as e:
            raise FileProcessingError(f"XML Parse Error in {path.name}: {e}")

    def write(self, data: XmlEt.Element, file_path: Union[str, Path], pretty_print: bool = True) -> None:
        path = Path(file_path)
        if pretty_print:
            XmlEt.indent(data, space="  ", level=0)
        with self._atomic_write_path(path) as tmp:
            XmlEt.ElementTree(data).write(tmp, encoding=self.encoding, xml_declaration=True)


# --- Domain Analyzers ---

class PromptConfigAnalyzer:
    """
    Analyzer for prompt version management and loading.
    Resolves specific prompt configuration files by referencing a central 'registry.yaml'.

    Example structure of './registry.yaml':
    ----------------------------------
    versions:
      - type: "math_phys_v0"
        version: "0.0"
        file: "versions/math_phys_v0.0.yaml"
        status: "deprecated"
        description: "Initial version for problem generation."

      - type: "math_phys_v1"
        version: "1.0"
        file: "versions/math_phys_v1.0.yaml"
        status: "active"
        description: "Optimized version with improved reasoning."

    Example structure of './versions/math_phys_v1.0.yaml':
    ----------------------------------
    metadata:
      type: "math_phys_v1"
      version: "1.0"
      description: "math_phys_problem_gen_v1.py:"

    prompts:
      system: |
        You are an expert at devising complex mathematical ...

      user_template: |
        Some example problem-solution pairs are given to facilitate your ...
    """

    def __init__(
            self,
            prompts_dir: Union[str, Path],
            registry_file_name: str = "registry.yaml",
            yaml_handler: Optional[YamlHandler] = None,
    ):
        """
        Initializes the analyzer with a prompt directory and an optional YAML handler.

        Args:
            prompts_dir: Path to the directory containing 'registry.yaml' and prompt files.
            yaml_handler: Optional instance of YamlHandler for file operations.
        """
        self.prompts_dir = Path(prompts_dir)
        self._yaml = yaml_handler or YamlHandler()
        self.registry_path = self.prompts_dir / registry_file_name

    def _load_registry(self) -> Dict[str, Any]:
        """
        Internal method to load the prompt registry file.

        Raises:
            FileProcessingError: If the registry file does not exist.
        """
        if not self.registry_path.exists():
            raise FileProcessingError(f"Registry file not found at: {self.registry_path}")
        return self._yaml.read(self.registry_path)

    def load_config(self, version: str, prompt_type: str = "generate") -> Dict[str, Any]:
        """
        Loads prompt configurations for the specified version and type.

        Args:
            version: The version string of the prompt (e.g., '1.0').
            prompt_type: The category/type of the prompt (default: 'generate').

        Returns:
            Dict[str, Any]: The contents of the resolved prompt configuration file.

        Raises:
            FileProcessingError: If the version/type is not found or the config file is missing.
        """
        if not version:
            raise FileProcessingError("Prompt version must be specified.")

        registry = self._load_registry()

        # Search for entry matching both version and type
        versions = registry.get('versions', [])
        version_info = next(
            (v for v in versions
             if str(v.get('version')) == str(version) and v.get('type', 'generate') == prompt_type),
            None
        )

        if not version_info:
            available = [(v.get('version'), v.get('type', 'generate')) for v in versions]
            raise FileProcessingError(
                f"Version '{version}' with type '{prompt_type}' not found in registry. "
                f"Available versions: {available}"
            )

        # Resolve the specific configuration file path
        config_path = self.prompts_dir / version_info['file']
        if not config_path.exists():
            raise FileProcessingError(f"Prompt config file not found: {config_path}")

        config = self._yaml.read(config_path)

        # Log loading details using the shared logger
        logger.info(f"Successfully loaded prompt version: {version} (Type: {prompt_type})")
        logger.info(f"  Status: {version_info.get('status')}")
        logger.info(f"  Description: {version_info.get('description')}")

        return config


if __name__ == "__main__":
    """
    python -m util.file_tools_gen
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

    jl_h.write(dict_list, file_path=jl_file)

    jl_data = jl_h.read(jl_file)
    print(list(jl_data))

    """ test for json """
    j_h = JsonHandler()
    j_file = "./data/sample_write.json"

    dict_data = {"data": dict_list}

    j_h.write(dict_data, file_path=j_file)

    j_data = j_h.read(j_file)
    print(j_data)

    """ test for yaml prompt analyzer """
    prompts_dir = Path("./client/concrete/prompts")
    analyzer = PromptConfigAnalyzer(prompts_dir)
    try:
        config = analyzer.load_config(version='1.0', prompt_type="math_phys_v1")
        system_prompt = config['prompts']['system']
        user_template = config['prompts']['user_template']
        print(f"Loaded: {system_prompt[:30]}...")
    except FileProcessingError as e:
        print(f"Error: {e}")
