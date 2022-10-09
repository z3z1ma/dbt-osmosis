import sys

import dbt_osmosis.main

if __name__ == "__main__":
    dbt_osmosis.main.cli(sys.argv[1:])
