import os
from contextlib import closing
from pathlib import Path
from typing import Dict, Optional, Union

from sqlfluff.cli.commands import get_config, get_linter_and_formatter
from sqlfluff.cli.outputstream import FileOutput


def lint_command(
    sql: Union[Path, str],
    extra_config_path: Optional[str] = None,
    ignore_local_config: bool = False,
) -> Dict:
    """Lint specified file or SQL string.

    This is essentially a streamlined version of the SQLFluff command-line lint
    function, sqlfluff.cli.commands.lint().
    """
    # TODO: Should get_config() be a one-time thing in /register?
    config = get_config(extra_config_path, ignore_local_config, require_dialect=False, nocolor=True)
    with closing(FileOutput(config, os.devnull)) as output_stream:
        lnt, formatter = get_linter_and_formatter(config, output_stream)

        if isinstance(sql, str):
            # Lint SQL passed in as a string
            result = lnt.lint_string_wrapped(sql)
        else:
            # Lint a SQL file
            result = lnt.lint_paths(
                tuple([str(sql)]),
                ignore_files=False,
            )
    # TODO: Exit code may be useful somehow?
    # return result.stats()["exit code"]
    records = result.as_records()
    assert len(records) == 1
    return records[0]
