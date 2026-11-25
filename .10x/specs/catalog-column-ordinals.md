Status: active
Created: 2026-11-25
Updated: 2026-11-25

# Preserve catalog ordinals through Core metadata conversion

## Contract
For an accepted catalog v1 resource, catalog-specific conversion into the metadata consumed by column transforms MUST retain each column's original integer index. The CLI's existing database-order setting MUST therefore sort by the supplied ordinal regardless of JSON object member insertion order on every supported Core baseline. Conversion MUST preserve names, types and comments and MUST NOT mutate the loaded artifact or input file.

The implementation MUST leave the current shared metadata-class behavior, public mapping iteration order, warehouse enumeration/flattening, ignore-pattern behavior, relation matching and explicit alphabetical setting unchanged. It MUST NOT sort catalog objects globally or infer ordinal constraints not present in the current schema. Sparse, negative, zero and tied integer ordinals remain accepted. Equal positions retain the existing stable transform sorting behavior; do not manufacture an ordering between ties. Missing/non-integer indexes retain current input validation errors. Raw producer fixtures MUST remain byte-for-byte unchanged; reordered copies are explicitly derived consumer inputs.

## Scenarios and acceptance
- Given a valid recorded catalog and a derived copy with only reordered column object members, database-order CLI output for models and sources has the same ordinal order and preserved descriptions/types/semantic YAML fields.
- Given Core1.8's distinct catalog metadata class, conversion retains original indexes rather than replacing them with enumeration offsets. Given newer Core's shared class, output behavior is unchanged.
- Given explicit alphabetical sorting, output remains alphabetical regardless of catalog ordinal/member order.
- Given negative/sparse/zero/tied integer positions, ordering follows existing stable numeric transform semantics without normalization or rejection.
- Given warehouse columns or flattened children, current enumeration indexes are unchanged.
- Given disabled introspection and supplied catalog, tests need no warehouse connection; CLI tests assert successful user-visible YAML and artifact preservation.

## Authority and scope
Current https://docs.getdbt.com/reference/artifacts/catalog-json identifies index as ordinal. https://schemas.getdbt.com/dbt/catalog/v1.json defines a columns object with required integer index, no minimum/default/uniqueness constraint. Existing sort_columns_as_in_database sorts by metadata.index and explicit alphabetical configuration is separately authoritative. The actual Core1.8 CLI counterexample and newer-Core controls are recorded in `.10x/tickets/2026-11-25-catalog-cli-boundary.md`. User-authorized offline roadmap correctness maintenance supplies scope; parent approved this bounded direction after measured evidence. Historical #482/#485 zero-based observations remain evidence, now superseded as intended behavior by the reproduced CLI defect.

No support-policy/dependency changes, warehouse execution, release, #468 action or broader runtime equivalence claim.
