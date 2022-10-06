"""Defines the dbt_osmosis templater."""

import os.path
import logging
import threading
from typing import Optional

from dbt.version import get_installed_version
from dbt.exceptions import (
    CompilationException as DbtCompilationException,
)
from dbt.contracts.graph.compiled import CompiledModelNode
from jinja2 import Environment
from jinja2_simple_tags import StandaloneTag
from sqlfluff.core.errors import SQLTemplaterError, SQLFluffSkipFile
from sqlfluff.core.templaters.base import TemplatedFile, large_file_check
from sqlfluff.core.templaters.jinja import JinjaTemplater

from dbt_osmosis.core.osmosis import DbtAdapterCompilationResult, DbtProject

templater_logger = logging.getLogger(__name__)

DBT_VERSION = get_installed_version()
DBT_VERSION_STRING = DBT_VERSION.to_version_string()
DBT_VERSION_TUPLE = (int(DBT_VERSION.major), int(DBT_VERSION.minor))

if DBT_VERSION_TUPLE >= (1, 3):
    COMPILED_SQL_ATTRIBUTE = "compiled_code"
    RAW_SQL_ATTRIBUTE = "raw_code"
else:
    COMPILED_SQL_ATTRIBUTE = "compiled_sql"
    RAW_SQL_ATTRIBUTE = "raw_sql"

local = threading.local()


def dbt_project_container():
    """Returns the dbt project container from server_v2."""
    # Import here to avoid circular import.
    from dbt_osmosis.core.server_v2 import app

    return app.state.dbt_project_container


class OsmosisDbtTemplater(JinjaTemplater):
    """dbt templater for dbt-osmosis, based on sqlfluff-templater-dbt."""

    # Same templater name as sqlfluff-templater-dbt. It is functionally
    # equivalent to that templater, but optimized for dbt-osmosis. The two
    # templaters cannot be installed in the same virtualenv.
    name = "dbt"

    def config_pairs(self):  # pragma: no cover
        """Returns info about the given templater for output by the cli."""
        return [("templater", self.name), ("dbt", DBT_VERSION_STRING)]

    @large_file_check
    def process(self, *, fname: str, config, **kwargs):
        """Compile a dbt model and return the compiled SQL."""
        try:
            return self._unsafe_process(os.path.abspath(fname), config)
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

    def _find_node(self, fname: str, project: DbtProject):
        expected_node_path = os.path.relpath(fname, start=os.path.abspath(project.args.project_dir))
        node = project.get_node_by_path(expected_node_path)
        if node:
            return node
        skip_reason = self._find_skip_reason(project, expected_node_path)
        if skip_reason:
            raise SQLFluffSkipFile(f"Skipped file {fname} because it is {skip_reason}")
        raise SQLFluffSkipFile(f"File {fname} was not found in dbt project")  # pragma: no cover

    @staticmethod
    def _find_skip_reason(project, expected_node_path) -> Optional[str]:
        """Return string reason if model okay to skip, otherwise None."""
        # Scan macros.
        for macro in project.dbt.macros.values():
            if macro.original_file_path == expected_node_path:
                return "a macro"

        # Scan disabled nodes.
        for nodes in project.dbt.disabled.values():
            for node in nodes:
                if node.original_file_path == expected_node_path:
                    return "disabled"
        return None

    def from_string(*args, **kwargs):
        """Replaces (via monkeypatch) the jinja2.Environment function."""
        globals = kwargs.get("globals")
        if globals and hasattr(local, "original_file_path"):
            model = globals.get("model")
            if model:
                # Is it processing the node we're interested in?
                if model.get("original_file_path") == local.original_file_path:
                    # Yes. Capture the important arguments and create
                    # a make_template() function.
                    env = args[0]
                    globals = args[2] if len(args) >= 3 else kwargs["globals"]

                    def make_template(in_str):
                        env.add_extension(SnapshotExtension)
                        return env.from_string(in_str, globals=globals)

                    local.make_template = make_template

        return old_from_string(*args, **kwargs)

    def _unsafe_process(self, fname, config=None):
        osmosis_dbt_project = dbt_project_container().get_project_by_root_dir(
            config.get_section((self.templater_selector, self.name, "project_dir"))
        )
        node = self._find_node(fname, osmosis_dbt_project)
        local.original_file_path = os.path.relpath(
            fname, start=osmosis_dbt_project.args.project_dir
        )
        local.make_template = None
        try:
            if not isinstance(node, CompiledModelNode):
                compiled_node = osmosis_dbt_project.compile_node(node)
            else:
                compiled_node = DbtAdapterCompilationResult(
                    raw_sql=getattr(node, RAW_SQL_ATTRIBUTE),
                    compiled_sql=node.compiled_sql,
                    node=node,
                    injected_sql=getattr(node, "injected_sql", None),
                )
        except Exception as err:
            raise SQLFluffSkipFile(  # pragma: no cover
                f"Skipped file {fname} because dbt raised a fatal "
                f"exception during compilation: {err!s}"
            ) from err
        finally:
            local.original_file_path = None

        if compiled_node.injected_sql:
            # If injected SQL is present, it contains a better picture
            # of what will actually hit the database (e.g. with tests).
            # However it's not always present.
            compiled_sql = compiled_node.injected_sql
        else:
            compiled_sql = compiled_node.compiled_sql

        raw_sql = compiled_node.raw_sql

        if not compiled_sql:  # pragma: no cover
            raise SQLTemplaterError(
                "dbt templater compilation failed silently, check your "
                "configuration by running `dbt compile` directly."
            )

        with open(fname) as source_dbt_model:
            source_dbt_sql = source_dbt_model.read()

        if not source_dbt_sql.rstrip().endswith("-%}"):
            n_trailing_newlines = len(source_dbt_sql) - len(source_dbt_sql.rstrip("\n"))
        else:
            # Source file ends with right whitespace stripping, so there's
            # no need to preserve/restore trailing newlines, as they would
            # have been removed regardless of dbt'skeep_trailing_newlines=False
            # behavior.
            n_trailing_newlines = 0

        templater_logger.debug(
            "    Trailing newline count in source dbt model: %r",
            n_trailing_newlines,
        )
        templater_logger.debug("    Raw SQL before compile: %r", source_dbt_sql)
        templater_logger.debug("    Node raw SQL: %r", raw_sql)
        templater_logger.debug("    Node compiled SQL: %r", compiled_sql)

        # Adjust for dbt Jinja removing trailing newlines. For more details on
        # this, see the similar code in sqlfluff-templater.dbt.
        compiled_node.raw_sql = source_dbt_sql
        compiled_sql = compiled_sql + "\n" * n_trailing_newlines
        raw_sliced, sliced_file, templated_sql = self.slice_file(
            source_dbt_sql,
            compiled_sql,
            config=config,
            make_template=local.make_template,
            append_to_templated="\n" if n_trailing_newlines else "",
        )
        return (
            TemplatedFile(
                source_str=source_dbt_sql,
                templated_str=templated_sql,
                fname=fname,
                sliced_file=sliced_file,
                raw_sliced=raw_sliced,
            ),
            # No violations returned in this way.
            [],
        )


# Monkeypatch Environment.from_string(). OsmosisDbtTemplater uses this to
# intercept Jinja compilation and capture a template trace.
old_from_string = Environment.from_string
Environment.from_string = OsmosisDbtTemplater.from_string


class SnapshotExtension(StandaloneTag):
    """Dummy "snapshot" tags so raw dbt templates will parse.

    For more context, see sqlfluff-templater-dbt.
    """

    tags = {"snapshot", "endsnapshot"}

    def render(self, format_string=None):
        """Dummy method that renders the tag."""
        return ""
