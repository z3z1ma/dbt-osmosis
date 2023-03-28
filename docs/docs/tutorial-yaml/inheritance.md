---
sidebar_position: 3
---
# Inheritance

## Overview

A really clutch feature of dbt-osmosis is the ability to inherit documentation from parent nodes. This is especially useful when you have a large number of models that share the same documentation. For example, if you have a large number of models that are all derived from a single source table, you can define the documentation for the source table once and then inherit it for all of the models that are derived from it. This means you are able to be more DRY with your documentation. Alternatives such as dbt-codegen only go up one level of inheritance. dbt-osmosis traverses the entire hierarchy of your dbt project and inherits documentation from all parent nodes for the specific node being documented. 

## Details

dbt-osmosis accumulates a knowledge graph for a specfic model by traversing the edges until it reaches the furthest removed ancestors of a node. It then caches all the documentation into a dictionary. It then traverses the edges in the opposite direction, starting from the furthest removed ancestors and working its way down to the node being documented merging in documentation. Once we have built the graph we can lean on it for any undocumented columns. 

The crux of the value proposition is that we often alias columns in our staging models and use them many times in many places without changing the name. This means, within the context of a specific models family tree, we should be able to inherit that knowledge. This inheritance can include tags and descriptions. This permits propagating PII, GDPR, and other compliance related tags for example. When a column is used in a model and its definition is semantically different while the column name is the same (which is a questionable practice), you should update the definition for that column in that model. The inheritors will use the updated definition if they pull from said model. 

## Takeways

While this obviously has some limitations, it is a powerful feature that can help you be more DRY with your documentation. It is also a great way to ensure that your documentation is consistent across your entire dbt project. It captures +80% of the meat of the repetitive documentation for a model.

## Future Work

In the future we would like to support fuzzy string matching across specific boundaries so that documentation inheritance can flow across models where renames are expected such as staging models. We should also support ignoring prefixes when inheriting documentation. This would allow us to inherit documentation from a parent model even if the child purposefully renames all the columns with a prefix for explicitness.

We should also support LLMs by using the knowledge graph as a prompt for ChatGPT or some such so that a project can be intelligently self-documenting.

Lastly, how can we support a data dictionary as an input. Something like a CSV or JSON file that can be used to populate the knowledge graph. This would allow for a more structured approach to documentation.
