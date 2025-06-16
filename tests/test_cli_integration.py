"""
Simple integration test to verify the CLI flag parsing works correctly.
"""

from click.testing import CliRunner
from dbt_osmosis.cli.main import cli


def test_cli_output_to_upper_flag_recognized():
    """Test that the CLI recognizes the --output-to-upper flag."""
    runner = CliRunner()
    
    # Test that the flag shows up in help for refactor command
    result = runner.invoke(cli, ['yaml', 'refactor', '--help'])
    assert result.exit_code == 0
    assert '--output-to-upper' in result.output
    assert 'uppercase if possible.' in result.output
    
    # Test that the flag shows up in help for document command
    result = runner.invoke(cli, ['yaml', 'document', '--help'])
    assert result.exit_code == 0
    assert '--output-to-upper' in result.output
    assert 'uppercase if possible.' in result.output


def test_cli_both_flags_available():
    """Test that both --output-to-lower and --output-to-upper flags are available."""
    runner = CliRunner()
    
    # Test refactor command
    result = runner.invoke(cli, ['yaml', 'refactor', '--help'])
    assert result.exit_code == 0
    assert '--output-to-lower' in result.output
    assert '--output-to-upper' in result.output
    
    # Test document command  
    result = runner.invoke(cli, ['yaml', 'document', '--help'])
    assert result.exit_code == 0
    assert '--output-to-lower' in result.output
    assert '--output-to-upper' in result.output


def test_cli_flag_parsing_no_errors():
    """Test that the CLI can parse the flags without errors (dry run)."""
    runner = CliRunner()
    
    # Test that the command accepts the flag without erroring on argument parsing
    # Using --dry-run and --disable-introspection to avoid actual execution
    result = runner.invoke(cli, [
        'yaml', 'refactor', 
        '--dry-run', 
        '--disable-introspection', 
        '--output-to-upper',
        '--project-dir', '/tmp'  # This will fail later but argument parsing should work
    ])
    
    # We expect this to fail due to missing project setup, but the exit code 
    # should not be 2 (which indicates argument parsing errors)
    assert result.exit_code != 2, f"Argument parsing failed. Output: {result.output}"


def test_organize_command_no_output_flags():
    """Test that organize command doesn't have the output transformation flags."""
    runner = CliRunner()
    
    result = runner.invoke(cli, ['yaml', 'organize', '--help'])
    assert result.exit_code == 0
    assert '--output-to-lower' not in result.output
    assert '--output-to-upper' not in result.output
