---
sidebar_position: 5
---
# Workflow

## YAML Files

dbt-osmosis **manages** your YAML files in a **declarative** manner based on the configuration in your `dbt_project.yml`. In many cases, you don’t even have to manually create YAML files; dbt-osmosis will generate them on your behalf if they don’t exist. Once established, any changes you make to these config rules are automatically enforced—dbt-osmosis will move or merge YAML files accordingly.

### Sources

By default, **dbt-osmosis** watches your sources from the **dbt manifest**. If you declare specific source paths under `vars: dbt-osmosis: sources`, dbt-osmosis can:

- **Create** those YAML files if they don’t exist yet (bootstrapping).
- **Synchronize** them with the **actual** database schema (pulling in missing columns unless you specify `--skip-add-source-columns`).
- **Migrate** them if you change where they’re supposed to live (e.g., if you update `path: "some/new/location.yml"` in `dbt_project.yml`, dbt-osmosis will move it).

**Key benefit**: You never have to scaffold source YAML by hand. Merely define the path under `vars: dbt-osmosis:` (and optionally a custom schema). If your source changes (columns, new tables), dbt-osmosis can detect and update the YAML to match.

```yaml title="dbt_project.yml"
vars:
  dbt-osmosis:
    sources:
      salesforce:
        path: "staging/salesforce/source.yml"
        schema: "salesforce_v2"
      marketo: "staging/customer/marketo.yml"
```

### Models

Similarly, **models** are managed based on your `+dbt-osmosis` directives in `dbt_project.yml`. For each folder (or subfolder) of models:

```yaml title="dbt_project.yml"
models:
  my_project:
    +dbt-osmosis: "{parent}.yml"

    intermediate:
      +dbt-osmosis: "{node.config[materialized]}/{model}.yml"

    # etc.

seeds:
  my_project:
    # DON'T FORGET: seeds need a +dbt-osmosis rule too!
    +dbt-osmosis: "_schema.yml"
```

When you run `dbt-osmosis yaml` commands (like `refactor`, `organize`, `document`):

- **Missing** YAML files are automatically **bootstrapped**.
- dbt-osmosis merges or updates any **existing** ones.
- If you rename or move a model to a different folder (and thus a different `+dbt-osmosis` rule), dbt-osmosis merges or moves the corresponding YAML to match.

Because dbt-osmosis enforces your declared file paths, you **won’t** inadvertently end up with duplicate or out-of-date YAML references.

---

## Running dbt-osmosis

Whether you are focusing on daily doc updates or large-scale refactors, dbt-osmosis can be triggered in **three** common ways. You can pick whichever method suits your team’s workflow best; they’re not mutually exclusive.

### 1. On-demand ⭐️

**Simplest approach**: occasionally run dbt-osmosis when you want to tidy things up or ensure docs are current. For instance:

```bash
# Example: refactor and see if changes occur
dbt-osmosis yaml refactor --target prod --check
```

**Recommended usage**:

- **Monthly or quarterly** “cleanups”
- Ad hoc runs on a **feature branch** (review & merge the changes if they look good)
- Let developers manually run it whenever they have significantly changed schemas

A single execution often yields **substantial** value by updating or reorganizing everything according to your rules.

### 2. Pre-commit hook ⭐️⭐️

To **automate** doc and schema alignment, you can add dbt-osmosis to your team’s **pre-commit** hooks. It will run automatically whenever you commit any `.sql` files in, say, `models/`.

```yaml title=".pre-commit-config.yaml"
repos:
  - repo: https://github.com/z3z1ma/dbt-osmosis
    rev: v1.1.5
    hooks:
      - id: dbt-osmosis
        files: ^models/
        # Optionally specify any arguments, e.g. production target:
        args: [--target=prod]
        additional_dependencies: [dbt-<adapter>]
```

**Pro**: Docs never go stale because every commit updates them.
**Con**: Slight overhead on each commit, but typically manageable if you filter only changed models.

### 3. CI/CD ⭐️⭐️⭐️

You can also integrate dbt-osmosis into your **continuous integration** pipeline. For example, a GitHub Action or a standalone script might:

1. Clone your repo into a CI environment.
2. Run `dbt-osmosis yaml refactor`.
3. Commit any resulting changes back to a branch or open a pull request.

```bash title="example.sh"
git clone https://github.com/my-org/my-dbt-project.git
cd my-dbt-project
git checkout -b dbt-osmosis-refactor

dbt-osmosis yaml refactor --target=prod

git commit -am "✨ dbt-osmosis refactor"
git push origin -f
gh pr create
```

**Pros**:

- Automated and **reviewable** in a PR.
- Takes the load off dev machines by running it in a controlled environment.

**Cons**:

- Requires some CI setup.
- Devs must remember to review and merge the PR.

---

**In summary**, dbt-osmosis fits a wide range of workflows. Whether you run it on-demand, as a pre-commit hook, or integrated into a CI pipeline, you’ll enjoy a consistent, automated approach to maintaining and updating your dbt YAML files for both **sources** and **models**.
