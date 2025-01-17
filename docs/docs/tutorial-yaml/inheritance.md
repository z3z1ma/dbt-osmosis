---
sidebar_position: 3
---
# Inheritance

## Overview

One of the most powerful features of dbt-osmosis is **multi-level column documentation inheritance**. This means **all** relevant tags, descriptions, and meta fields for your columns can **cascade** from **any** upstream node to your current model.

For instance, if:

1. **Source** `salesforce` defines a column `contact_id` with a thorough description and some compliance tags (e.g. `["GDPR", "PII"]`).
2. A **staging model** aliases it to the same column name (`contact_id`) but doesn’t bother re-describing it.
3. A subsequent **intermediate** model references that staging model without modifying or renaming the column…

dbt-osmosis can detect that the final model’s `contact_id` inherits documentation from the source. This approach ensures **DRY** docs: define them once, pass them everywhere.

## How It Works

### 1. Building an Ancestor Tree

When dbt-osmosis looks at a given node (model, seed, or source), it first builds an **ancestor tree** by traversing all upstream dependencies—including **multiple** levels of parents. Concretely:

- It ignores ephemeral nodes (which typically aren’t documented or part of the lineage).
- It collects every **unique** ancestor node (source, seed, or model).
- If a node has more than one ancestor, it includes them **all** (not just the direct parent).

Internally, this is driven by `_build_node_ancestor_tree(...)` in the code. The result is a structure that might look like:

```
{
  "generation_0": ["model.my_project.intermediate.my_node"],       # the current node
  "generation_1": ["model.my_project.staging.salesforce_contacts"],
  "generation_2": ["source.my_project.salesforce"]                 # the farthest removed ancestor
}
```

### 2. Aggregating a “Knowledge Graph”

dbt-osmosis then **reverses** that tree so it can start from the **oldest** ancestor (like your source) and move **forward** in time through staging or intermediate nodes. It effectively layers each node’s documentation onto the final child. This is handled by `_build_column_knowledge_graph(...)`, which:

- Examines **every** column in your final node.
- Looks for matches in each ancestor’s columns.
- **Merges** any descriptions, tags, meta fields, or other specified keys from the ancestor into the child’s doc if the child’s doc is missing or is a placeholder.
- Skips ephemeral nodes and merges from all others.

As it merges, it respects your config flags, e.g. `--skip-merge-meta` or `--skip-add-tags`. If the child already has a **non-empty** description, it won’t overwrite it (unless you used `--force-inherit-descriptions`). If the child has only a placeholder, that is replaced with the upstream doc.

### 3. Handling Renames and Fuzzy Matching

By default, dbt-osmosis expects **exact** column name matches to pass doc down. However, the code also includes:

- **FuzzyCaseMatching**: handles uppercase vs. lowercase variations.
- **FuzzyPrefixMatching**: can strip known prefixes, e.g. if your staging model renamed `contact_id` to `stg_contact_id` but you still want to treat them as the same column.

In the future, these hooks may expand further for more advanced fuzzy matches or ignoring certain patterns. You can also implement your own plugin to handle custom rename rules.

### 4. Updating the Child Node

Finally, once the “knowledge graph” is built, dbt-osmosis updates each column in the child node. This step is done by `inherit_upstream_column_knowledge(...)`. If the child column is missing data—like a description—it’s **populated** with the best available doc from an upstream ancestor.

## Example

Imagine a lineage chain:

```
source.salesforce → staging.salesforce_contacts → intermediate.int_contacts → marts.int_contacts_reporting
```

Your `source.salesforce` might say:

```yaml
columns:
  - name: contact_id
    description: "Unique ID for each contact from Salesforce"
    tags: ["PII", "GDPR"]
  - name: email
    description: "Primary email address of the contact"
```

Your `staging.salesforce_contacts` might not redefine `contact_id` or `email`. So dbt-osmosis sees they are “inherited.” Then your `intermediate.int_contacts` has `contact_id`, but changes the description to “Contact ID, updated daily.” So, by the time we get to `marts.int_contacts_reporting`, that new description has priority. We still preserve the original compliance tags from the source unless you removed or modified them along the way.

## When Columns Diverge

If at some point you rename columns or drastically change their meaning (while still using the same name upstream), dbt-osmosis stops short of forcing them to match. You can:

- **Manually** override the doc in the child to reflect the new meaning.
- If you truly want them to be distinct, you give it a different name, thus skipping inheritance for that new column name.
- Or leverage fuzzy prefix logic if you’re systematically prefixing them and still want doc to pass down.

## Config Flags That Control Inheritance

Below are some relevant flags to refine how much doc merges:

- `--skip-merge-meta`: Skip inheriting `meta` fields from parent nodes.
- `--skip-add-tags`: Skip inheriting tags from upstream.
- `--force-inherit-descriptions`: Overwrite the child’s existing (but possibly placeholder) descriptions with the parent’s doc.
- `--add-progenitor-to-meta`: Mark each column with a `meta.osmosis_progenitor` field, so you can see *which* node it was inherited from.
- `--add-inheritance-for-specified-keys=policy_tags` (and so on): Inherit additional custom fields from your upstream docs.

## Key Benefits

- **Eliminate Repetitive Docs**: Document columns once at the source (or the earliest staging) and let everything downstream reuse it.
- **Propagate Compliance Tags**: If `GDPR` or `PII` tags are attached to certain columns, they follow that column across your entire pipeline.
- **Promote Consistency**: No more “different descriptions for the same field.”
- **Extendable**: You can create your own fuzzy matching logic if your naming patterns are more complex.

## Future Enhancements

- **Robust Fuzzy Matching**: Extending column name matching to handle more patterns or partial renames.
- **External Data Dictionary**: Ingest a CSV or JSON dictionary of columns and doc them as if they were an upstream node.
- **Integration with LLMs**: Combining the knowledge graph with an LLM to automatically fill out doc for columns that have none—**dbt-osmosis** already supports `--synthesize` for OpenAI, but deeper integration is possible.
- **Ignoring Common Prefixes**: For example, ignoring “stg_” or “dim_” while linking columns between child and parent.

---

### Takeaways

**Multi-level column inheritance** is a central feature of dbt-osmosis that drastically cuts down on repetitive documentation. By building a knowledge graph from all upstream nodes, dbt-osmosis ensures each model or source gets the most complete doc possible for every column, while letting you override or skip certain merges as needed. Whether you want to unify compliance tags, pass down descriptions, or unify large parts of your lineage, **inheritance** is how you keep your doc consistent without repeating yourself.
