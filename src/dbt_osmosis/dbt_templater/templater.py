"""Defines the dbt_osmosis templater."""

from contextlib import contextmanager
import os
import os.path
import logging
import threading
from typing import Optional

from dbt.version import get_installed_version
from dbt.adapters.factory import get_adapter
from dbt.exceptions import (
    CompilationException as DbtCompilationException,
    FailedToConnectException as DbtFailedToConnectException,
)
from dbt.contracts.graph.compiled import CompiledModelNode
from jinja2 import Environment
from jinja2_simple_tags import StandaloneTag

from sqlfluff.core.cached_property import cached_property
from sqlfluff.core.errors import SQLTemplaterError, SQLFluffSkipFile
from sqlfluff.core.templaters.base import TemplatedFile, large_file_check
from sqlfluff.core.templaters.jinja import JinjaTemplater

from dbt_osmosis.core.osmosis import DbtAdapterCompilationResult

# Instantiate the logger
templater_logger = logging.getLogger("dbt_osmosis.dbt_templater")

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

# Below, we monkeypatch Environment.from_string() to intercept when dbt
# compiles (i.e. runs Jinja) to expand the "node" corresponding to fname.
# We do this to capture the Jinja context at the time of compilation, i.e.:
# - Jinja Environment object
# - Jinja "globals" dictionary
#
# This info is captured by the "make_template()" function, which in
# turn is used by our parent class' (JinjaTemplater) slice_file()
# function.
old_from_string = Environment.from_string


class OsmosisDbtTemplater(JinjaTemplater):
    """dbt templater for dbt-osmosis.

    Based on sqlfluff-templater-dbt.
    """

    name = "dbt"
    sequential_fail_limit = 3

    def __init__(self, **kwargs):
        self.sqlfluff_config = None
        self.formatter = None
        self.working_dir = os.getcwd()
        self._sequential_fails = 0
        self.connection_acquired = False
        super().__init__(**kwargs)

    def config_pairs(self):  # pragma: no cover TODO?
        """Returns info about the given templater for output by the cli."""
        return [("templater", self.name), ("dbt", self.dbt_version)]

    @property
    def dbt_version(self):
        """Gets the dbt version."""
        return DBT_VERSION_STRING

    @property
    def dbt_version_tuple(self):
        """Gets the dbt version as a tuple on (major, minor)."""
        return DBT_VERSION_TUPLE

    @property
    def dbt_config(self):
        """Loads the dbt config."""
        from dbt_osmosis.core.server_v2 import app

        return app.state.dbt_project_container["dbt_project"].config

    @property
    def dbt_manifest(self):
        """Returns the dbt manifest."""
        from dbt_osmosis.core.server_v2 import app

        return app.state.dbt_project_container["dbt_project"].dbt

    def _get_profile(self):
        """Get a dbt profile name from the configuration."""
        return self.sqlfluff_config.get_section((self.templater_selector, self.name, "profile"))

    def _get_target(self):
        """Get a dbt target name from the configuration."""
        return self.sqlfluff_config.get_section((self.templater_selector, self.name, "target"))

    @large_file_check
    def process(self, *, fname, in_str=None, config=None, formatter=None):
        """Compile a dbt model and return the compiled SQL.

        Args:
            fname (:obj:`str`): Path to dbt model(s)
            in_str (:obj:`str`, optional): This is ignored for dbt
            config (:obj:`FluffConfig`, optional): A specific config to use for this
                templating operation. Only necessary for some templaters.
            formatter (:obj:`CallbackFormatter`): Optional object for output.
        """
        # Stash the formatter if provided to use in cached methods.
        self.formatter = formatter
        self.sqlfluff_config = config
        fname_absolute_path = os.path.abspath(fname)

        try:
            processed_result = self._unsafe_process(fname_absolute_path, in_str, config)
            # Reset the fail counter
            self._sequential_fails = 0
            return processed_result
        except DbtCompilationException as e:
            # Increment the counter
            self._sequential_fails += 1
            if e.node:
                return None, [
                    SQLTemplaterError(
                        f"dbt compilation error on file '{e.node.original_file_path}', " f"{e.msg}",
                        # It's fatal if we're over the limit
                        fatal=self._sequential_fails > self.sequential_fail_limit,
                    )
                ]
            else:
                raise  # pragma: no cover
        except DbtFailedToConnectException as e:
            return None, [
                SQLTemplaterError(
                    "dbt tried to connect to the database and failed: you could use "
                    "'execute' to skip the database calls. See"
                    "https://docs.getdbt.com/reference/dbt-jinja-functions/execute/ "
                    f"Error: {e.msg}",
                    fatal=True,
                )
            ]
        # If a SQLFluff error is raised, just pass it through
        except SQLTemplaterError as e:  # pragma: no cover
            return None, [e]

    def _find_node(self, fname, config=None):
        if not fname:  # pragma: no cover
            raise ValueError("For the dbt templater, the `process()` method requires a file name")
        elif fname == "stdin":  # pragma: no cover
            raise ValueError(
                "The dbt templater does not support stdin input, provide a path instead"
            )
        from dbt_osmosis.core.server_v2 import app

        osmosis_dbt_project = app.state.dbt_project_container["dbt_project"]
        expected_node_path = os.path.relpath(
            fname, start=os.path.abspath(osmosis_dbt_project.args.project_dir)
        )
        found_node = None
        for node in osmosis_dbt_project.dbt.nodes.values():
            # TODO: Scans all nodes. Could be slow for large projects. Is there
            # a better way to do this?
            if node.original_file_path == expected_node_path:
                found_node = node
                break
        if not found_node:
            skip_reason = self._find_skip_reason(expected_node_path)
            if skip_reason:
                raise SQLFluffSkipFile(f"Skipped file {fname} because it is {skip_reason}")
            raise SQLFluffSkipFile(
                "File %s was not found in dbt project" % fname
            )  # pragma: no cover
        return found_node

    def _find_skip_reason(self, expected_node_path) -> Optional[str]:
        """Return string reason if model okay to skip, otherwise None."""
        # Scan macros.
        for macro in self.dbt_manifest.macros.values():
            if macro.original_file_path == expected_node_path:
                return "a macro"

        # Scan disabled nodes.
        for nodes in self.dbt_manifest.disabled.values():
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

    # Apply the monkeypatch. To avoid issues with multiple threads, we never
    # remove the patch.
    Environment.from_string = from_string

    def _unsafe_process(self, fname, in_str=None, config=None):
        from dbt_osmosis.core.server_v2 import app

        osmosis_dbt_project = app.state.dbt_project_container["dbt_project"]
        node = self._find_node(fname, config)
        templater_logger.debug(
            "_find_node for path %r returned object of type %s.", fname, type(node)
        )

        save_ephemeral_nodes = dict(
            (k, v)
            for k, v in self.dbt_manifest.nodes.items()
            if v.config.materialized == "ephemeral" and not getattr(v, "compiled", False)
        )
        with self.connection():
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
                templater_logger.exception(
                    "Fatal dbt compilation error on %s. This occurs most often "
                    "during incorrect sorting of ephemeral models before linting. "
                    "Please report this error on github at "
                    "https://github.com/sqlfluff/sqlfluff/issues, including "
                    "both the raw and compiled sql for the model affected.",
                    fname,
                )
                # Additional error logging in case we get a fatal dbt error.
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
                # have been removed regardless of dbt's
                # keep_trailing_newlines=False behavior.
                n_trailing_newlines = 0

            templater_logger.debug(
                "    Trailing newline count in source dbt model: %r",
                n_trailing_newlines,
            )
            templater_logger.debug("    Raw SQL before compile: %r", source_dbt_sql)
            templater_logger.debug("    Node raw SQL: %r", raw_sql)
            templater_logger.debug("    Node compiled SQL: %r", compiled_sql)

            # When using dbt-templater, trailing newlines are ALWAYS REMOVED during
            # compiling. Unless fixed (like below), this will cause:
            #    1. Assertion errors in TemplatedFile, when it sanity checks the
            #       contents of the sliced_file array.
            #    2. L009 linting errors when running "sqlfluff lint foo_bar.sql"
            #       since the linter will use the compiled code with the newlines
            #       removed.
            #    3. "No newline at end of file" warnings in Git/GitHub since
            #       sqlfluff uses the compiled SQL to write fixes back to the
            #       source SQL in the dbt model.
            #
            # The solution is (note that both the raw and compiled files have
            # had trailing newline(s) removed by the dbt-templater.
            #    1. Check for trailing newlines before compiling by looking at the
            #       raw SQL in the source dbt file. Remember the count of trailing
            #       newlines.
            #    2. Set node.raw_sql/node.raw_code to the original source file contents.
            #    3. Append the count from #1 above to compiled_sql. (In
            #       production, slice_file() does not usually use this string,
            #       but some test scenarios do.
            compiled_node.raw_sql = source_dbt_sql
            compiled_sql = compiled_sql + "\n" * n_trailing_newlines

            # TRICKY: dbt configures Jinja2 with keep_trailing_newline=False.
            # As documented (https://jinja.palletsprojects.com/en/3.0.x/api/),
            # this flag's behavior is: "Preserve the trailing newline when
            # rendering templates. The default is False, which causes a single
            # newline, if present, to be stripped from the end of the template."
            #
            # Below, we use "append_to_templated" to effectively "undo" this.
            raw_sliced, sliced_file, templated_sql = self.slice_file(
                source_dbt_sql,
                compiled_sql,
                config=config,
                make_template=local.make_template,
                append_to_templated="\n" if n_trailing_newlines else "",
            )
        # :HACK: If calling compile_node() compiled any ephemeral nodes,
        # restore them to their earlier state. This prevents a runtime error
        # in the dbt "_inject_ctes_into_sql()" function that occurs with
        # 2nd-level ephemeral model dependencies (e.g. A -> B -> C, where
        # both B and C are ephemeral). Perhaps there is a better way to do
        # this, but this seems good enough for now.
        for k, v in save_ephemeral_nodes.items():
            if getattr(self.dbt_manifest.nodes[k], "compiled", False):
                self.dbt_manifest.nodes[k] = v
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

    @contextmanager
    def connection(self):
        """Context manager that manages a dbt connection, if needed."""
        # We have to register the connection in dbt >= 1.0.0 ourselves
        # In previous versions, we relied on the functionality removed in
        # https://github.com/dbt-labs/dbt-core/pull/4062.
        if not self.connection_acquired:
            from dbt_osmosis.core.server_v2 import app

            adapter = get_adapter(self.dbt_config)
            adapter.acquire_connection("master")
            adapter.set_relations_cache(self.dbt_manifest)
            self.connection_acquired = True
        yield
        # :TRICKY: Once connected, we never disconnect. Making multiple
        # connections during linting has proven to cause major performance
        # issues.


class SnapshotExtension(StandaloneTag):
    """Dummy "snapshot" tags so raw dbt templates will parse.

    Context: dbt snapshots
    (https://docs.getdbt.com/docs/building-a-dbt-project/snapshots/#example)
    use custom Jinja "snapshot" and "endsnapshot" tags. However, dbt does not
    actually register those tags with Jinja. Instead, it finds and removes these
    tags during a preprocessing step. However, DbtTemplater needs those tags to
    actually parse, because JinjaTracer creates and uses Jinja to process
    another template similar to the original one.
    """

    tags = {"snapshot", "endsnapshot"}

    def render(self, format_string=None):
        """Dummy method that renders the tag."""
        return ""
