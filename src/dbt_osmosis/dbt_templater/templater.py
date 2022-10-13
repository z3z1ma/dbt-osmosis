"""Defines the dbt_osmosis templater."""
import logging
import os.path
import uuid
from pathlib import Path
from typing import Optional

from dbt.clients import jinja
from dbt.exceptions import CompilationException as DbtCompilationException
from dbt.version import get_installed_version
from jinja2_simple_tags import StandaloneTag
from sqlfluff.core.config import FluffConfig
from sqlfluff.core.errors import SQLTemplaterError
from sqlfluff.core.templaters.base import TemplatedFile, large_file_check
from sqlfluff.core.templaters.jinja import JinjaTemplater

from dbt_osmosis.core.osmosis import DbtProjectContainer

templater_logger = logging.getLogger(__name__)


class OsmosisDbtTemplater(JinjaTemplater):
    """dbt templater for dbt-osmosis, based on sqlfluff-templater-dbt."""

    # Same templater name as sqlfluff-templater-dbt. It is functionally
    # equivalent to that templater, but optimized for dbt-osmosis. The two
    # templaters cannot be installed in the same virtualenv.
    name = "dbt"

    def __init__(self, **kwargs):
        self.dbt_project_container: DbtProjectContainer = kwargs.pop("dbt_project_container")
        super().__init__(**kwargs)

    def config_pairs(self):  # pragma: no cover
        """Returns info about the given templater for output by the cli."""
        return [("templater", self.name), ("dbt", get_installed_version().to_version_string())]

    @large_file_check
    def process(
        self,
        *,
        in_str: str,
        fname: Optional[str] = None,
        config: Optional[FluffConfig] = None,
        **kwargs,
    ):
        """Compile a dbt model and return the compiled SQL."""
        try:
            return self._unsafe_process(os.path.abspath(fname) if fname else None, in_str, config)
        except DbtCompilationException as e:
            if e.node:
                return None, [
                    SQLTemplaterError(
                        f"dbt compilation error on file '{e.node.original_file_path}', " f"{e.msg}"
                    )
                ]
            else:
                raise  # pragma: no cover
        # If a SQLFluff error is raised, just pass it through
        except SQLTemplaterError as e:  # pragma: no cover
            return None, [e]

    def _unsafe_process(self, fname: Optional[str], in_str: str, config: FluffConfig = None):
        # Get project
        osmosis_dbt_project = self.dbt_project_container.get_project_by_root_dir(
            # from .sqlfluff templater project_dir
            config.get_section((self.templater_selector, self.name, "project_dir"))
        )

        # Use path if valid, prioritize it as the in_str
        fpath = Path(fname)
        if fpath.exists() and not in_str:
            in_str = fpath.read_text()

        # Generate node
        retry = 3
        while retry > 0:
            # In massive parallel requests, referring to higher loads than we will ever see in all likelihood,
            # this cheap retry increases reliability to ~100% in test cases of up to 50 clients across 1K reqs
            # without requiring the overhead of a synchronization primitive such as a mutex
            temp_node_id = str(uuid.uuid4())
            try:
                mock_node = osmosis_dbt_project.get_server_node(in_str, temp_node_id)
                resp = osmosis_dbt_project.compile_node(mock_node)
            except:
                retry -= 1
            else:
                break
            finally:
                osmosis_dbt_project._clear_node(temp_node_id)

        # Generate context
        ctx = osmosis_dbt_project.generate_runtime_model_context(resp.node)
        env = jinja.get_environment(resp.node)
        env.add_extension(SnapshotExtension)
        compiled_sql = resp.compiled_sql
        make_template = lambda _in_str: env.from_string(_in_str, globals=ctx)

        # Need compiled
        if not compiled_sql:  # pragma: no cover
            raise SQLTemplaterError(
                "dbt templater compilation failed silently, check your "
                "configuration by running `dbt compile` directly."
            )

        # Whitespace
        if not in_str.rstrip().endswith("-%}"):
            n_trailing_newlines = len(in_str) - len(in_str.rstrip("\n"))
        else:
            # Source file ends with right whitespace stripping, so there's
            # no need to preserve/restore trailing newlines.
            n_trailing_newlines = 0

        # LOG
        templater_logger.debug(
            "    Trailing newline count in source dbt model: %r",
            n_trailing_newlines,
        )
        templater_logger.debug("    Raw SQL before compile: %r", in_str)
        templater_logger.debug("    Node raw SQL: %r", in_str)
        templater_logger.debug("    Node compiled SQL: %r", compiled_sql)

        # SLICE
        raw_sliced, sliced_file, templated_sql = self.slice_file(
            raw_str=in_str,
            templated_str=compiled_sql + "\n" * n_trailing_newlines,
            config=config,
            make_template=make_template,
            append_to_templated="\n" if n_trailing_newlines else "",
        )

        return (
            TemplatedFile(
                source_str=in_str,
                templated_str=templated_sql,
                fname=fname,
                sliced_file=sliced_file,
                raw_sliced=raw_sliced,
            ),
            # No violations returned in this way.
            [],
        )


class SnapshotExtension(StandaloneTag):
    """Dummy "snapshot" tags so raw dbt templates will parse.

    For more context, see sqlfluff-templater-dbt.
    """

    tags = {"snapshot", "endsnapshot"}

    def render(self, format_string=None):
        """Dummy method that renders the tag."""
        return ""
