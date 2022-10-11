from dbt_osmosis.core.osmosis import DbtOsmosis

if __name__ == "__main__":
    dbt = DbtOsmosis(
        profiles_dir="/Users/alexanderbutler/Documents/harness/analytics-pipelines/transform/profiles/bigquery",
        project_dir="/Users/alexanderbutler/Documents/harness/analytics-pipelines/transform/",
        target="prod",
    )
