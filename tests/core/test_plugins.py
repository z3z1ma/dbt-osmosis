# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

from dbt_osmosis.core.plugins import (
    get_plugin_manager,
    FuzzyCaseMatching,
    FuzzyPrefixMatching,
)


def test_plugin_manager_hooks():
    """
    Ensure FuzzyCaseMatching and FuzzyPrefixMatching are registered by default,
    and that get_candidates works as expected.
    """
    pm = get_plugin_manager()
    # We can search for the classes
    plugins = pm.get_plugins()
    has_case = any(isinstance(p, FuzzyCaseMatching) for p in plugins)
    has_prefix = any(isinstance(p, FuzzyPrefixMatching) for p in plugins)
    assert has_case
    assert has_prefix

    # We'll manually trigger the hook
    # Typically: pm.hook.get_candidates(name="my_col", node=<some node>, context=<ctx>)
    results = pm.hook.get_candidates(name="my_col", node=None, context=None)
    # results is a list of lists from each plugin => flatten them
    combined = [variant for sublist in results for variant in sublist]
    # Expect e.g. my_col => MY_COL => myCol => MyCol from FuzzyCaseMatching
    # FuzzyPrefixMatching might do nothing unless we set prefix
    assert "my_col" in combined
    assert "MY_COL" in combined
    assert "myCol" in combined
    assert "MyCol" in combined
