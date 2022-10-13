"""Tests for the dbt templater."""

import glob
import os
import logging
import shutil
from pathlib import Path

import pytest

from sqlfluff.core import FluffConfig, Lexer, Linter
from sqlfluff.core.errors import SQLFluffSkipFile
from dbt_osmosis.core.osmosis import DBT_MAJOR_VER, DBT_MINOR_VER
from tests.sqlfluff_templater.fixtures.dbt.templater import (  # noqa: F401
    DBT_FLUFF_CONFIG,
    dbt_templater,
    project_dir,
)


def test__templater_dbt_missing(dbt_templater, project_dir):  # noqa: F811
    """Check that a nice error is returned when dbt module is missing."""
    try:
        import dbt  # noqa: F401

        pytest.skip(msg="dbt is installed")
    except ModuleNotFoundError:
        pass

    with pytest.raises(ModuleNotFoundError, match=r"pip install sqlfluff\[dbt\]"):
        dbt_templater.process(
            in_str="",
            fname=os.path.join(project_dir, "models/my_new_project/test.sql"),
            config=FluffConfig(configs=DBT_FLUFF_CONFIG),
        )


@pytest.mark.parametrize(
    "fname",
    [
        # dbt_utils
        "use_dbt_utils.sql",
        # macro calling another macro
        "macro_in_macro.sql",
        # config.get(...)
        "use_headers.sql",
        # var(...)
        "use_var.sql",
        # {# {{ 1 + 2 }} #}
        "templated_inside_comment.sql",
        # {{ dbt_utils.last_day(
        "last_day.sql",
        # Many newlines at end, tests templater newline handling
        "trailing_newlines.sql",
        # Ends with whitespace stripping, so trailing newline handling should
        # be disabled
        "ends_with_whitespace_stripping.sql",
    ],
)
def test__templater_dbt_templating_result(project_dir, dbt_templater, fname):  # noqa: F811
    """Test that input sql file gets templated into output sql file."""
    _run_templater_and_verify_result(dbt_templater, project_dir, fname)


def test_dbt_profiles_dir_env_var_uppercase(
    project_dir, dbt_templater, tmpdir, monkeypatch  # noqa: F811
):
    """Tests specifying the dbt profile dir with env var."""
    profiles_dir = tmpdir.mkdir("SUBDIR")  # Use uppercase to test issue 2253
    monkeypatch.setenv("DBT_PROFILES_DIR", str(profiles_dir))
    shutil.copy(os.path.join(project_dir, "../profiles_yml/profiles.yml"), str(profiles_dir))
    _run_templater_and_verify_result(dbt_templater, project_dir, "use_dbt_utils.sql")


def _run_templater_and_verify_result(dbt_templater, project_dir, fname):  # noqa: F811
    templated_file, _ = dbt_templater.process(
        in_str=None,
        fname=os.path.join(project_dir, "models/my_new_project/", fname),
        config=FluffConfig(configs=DBT_FLUFF_CONFIG),
    )
    template_output_folder_path = Path("tests/sqlfluff_templater/fixtures/dbt/templated_output/")
    fixture_path = _get_fixture_path(template_output_folder_path, fname)
    assert str(templated_file) == fixture_path.read_text()


def _get_fixture_path(template_output_folder_path, fname):
    fixture_path: Path = template_output_folder_path / fname  # Default fixture location
    # Is there a version-specific version of the fixture file?
    if (DBT_MAJOR_VER, DBT_MINOR_VER) >= (1, 0):
        dbt_version_specific_fixture_folder = "dbt_utils_0.8.0"
    else:
        dbt_version_specific_fixture_folder = None

    if dbt_version_specific_fixture_folder:
        # Maybe. Determine where it would exist.
        version_specific_path = (
            Path(template_output_folder_path) / dbt_version_specific_fixture_folder / fname
        )
        if version_specific_path.is_file():
            # Ok, it exists. Use this path instead.
            fixture_path = version_specific_path
    return fixture_path


@pytest.mark.parametrize(
    "raw_file,templated_file,result",
    [
        (
            "select * from a",
            """
with dbt__CTE__INTERNAL_test as (
select * from a
)select count(*) from dbt__CTE__INTERNAL_test
""",
            # The unwrapper should trim the ends.
            [
                ("literal", slice(0, 15, None), slice(0, 15, None)),
            ],
        )
    ],
)
def test__templater_dbt_slice_file_wrapped_test(
    raw_file, templated_file, result, dbt_templater, caplog  # noqa: F811
):
    """Test that wrapped queries are sliced safely using _check_for_wrapped()."""
    with caplog.at_level(logging.DEBUG, logger="sqlfluff.templater"):
        _, resp, _ = dbt_templater.slice_file(
            raw_file,
            templated_file,
        )
    assert resp == result


@pytest.mark.parametrize(
    "fname",
    [
        "tests/test.sql",
        "models/my_new_project/single_trailing_newline.sql",
        "models/my_new_project/multiple_trailing_newline.sql",
    ],
)
def test__templater_dbt_templating_test_lex(project_dir, dbt_templater, fname):  # noqa: F811
    """Demonstrate the lexer works on both dbt models and dbt tests.

    Handle any number of newlines.
    """
    source_fpath = os.path.join(project_dir, fname)
    with open(source_fpath, "r") as source_dbt_model:
        source_dbt_sql = source_dbt_model.read()
    n_trailing_newlines = len(source_dbt_sql) - len(source_dbt_sql.rstrip("\n"))
    lexer = Lexer(config=FluffConfig(configs=DBT_FLUFF_CONFIG))
    templated_file, _ = dbt_templater.process(
        in_str=None,
        fname=os.path.join(project_dir, fname),
        config=FluffConfig(configs=DBT_FLUFF_CONFIG),
    )
    tokens, lex_vs = lexer.lex(templated_file)
    assert templated_file.source_str == "select a\nfrom table_a" + "\n" * n_trailing_newlines
    assert templated_file.templated_str == "select a\nfrom table_a" + "\n" * n_trailing_newlines


@pytest.mark.parametrize(
    "fname",
    [
        "use_var.sql",
        "incremental.sql",
        "single_trailing_newline.sql",
        "L034_test.sql",
    ],
)
def test__dbt_templated_models_do_not_raise_lint_error(project_dir, fname):  # noqa: F811
    """Test that templated dbt models do not raise a linting error."""
    lntr = Linter(config=FluffConfig(configs=DBT_FLUFF_CONFIG))
    lnt = lntr.lint_path(path=os.path.join(project_dir, "models/my_new_project/", fname))
    violations = lnt.check_tuples()
    assert len(violations) == 0


def _clean_path(glob_expression):
    """Clear out files matching the provided glob expression."""
    for fsp in glob.glob(glob_expression):
        os.remove(fsp)


def test__templater_dbt_templating_absolute_path(project_dir, dbt_templater):  # noqa: F811
    """Test that absolute path of input path does not cause RuntimeError."""
    try:
        dbt_templater.process(
            in_str="",
            fname=os.path.abspath(os.path.join(project_dir, "models/my_new_project/use_var.sql")),
            config=FluffConfig(configs=DBT_FLUFF_CONFIG),
        )
    except Exception as e:
        pytest.fail(f"Unexpected RuntimeError: {e}")
