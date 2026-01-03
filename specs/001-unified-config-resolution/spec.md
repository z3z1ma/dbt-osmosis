# Feature Specification: Unified Configuration Resolution System

**Feature Branch**: `001-unified-config-resolution`
**Created**: 2026-01-02
**Status**: Draft
**Input**: User description: "In dbt osmosis there are probably three places we can specify configuration we can specify configuration in the dbt project dot yaml and we use either custom keys inside of the models or sources block basically like custom keys which go like plus dbt dash osmosis or we can put the dbt osmosis key inside of a plus config sort of thing so we can nest it in plus config or possibly in the meta too you might also check meta of a model so there's top level keys which are going to be deprecated by dbt then there's inside of the config so like plus config or there's inside of um meta finally I also want to support a way to specify configuration outside of the dbt hot path and like a supplementary yaml file configuration that can appear in all these different places I want to create a unified configuration resolver sort of interface and all we should need to do is ask for a key possibly scoped to a specific node like a model or seed or source and that key should be resolved with some precedence based on all the different places config can exist. This should naturally create a mechanism where pre dbt 1.10 where top level keys are allowed on 1.11 whatever it is where top level keys are allowed they work and then after when they stop working people just nest their config under plus config or plus meta and that also works just fine because we resolve it from all these places. So this unified resolution is super important and we also need unified resolution for column descriptions and really just any model property. A good example of this and what I'm talking about specifically is we get column descriptions from the in memory manifest object but those already have jinja rendered. We have a feature which lets people propagate unrendered jinja which is really useful if people use doc blocks so we want to be able to support that and right now it's some switch statements somewhere it's not centralized so we also need to centralize like a uniform interface for accessing model properties either from the YAML or from the in memory manifest depending on what the user wants."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Cross-DBT-Version Configuration Compatibility (Priority: P1)

As a dbt-osmosis user, I want to specify dbt-osmosis configuration in multiple locations (dbt_project.yml top-level keys, +config blocks, meta fields, and supplementary YAML files) so that my configuration works across different dbt versions without needing to refactor when dbt deprecates top-level custom keys.

**Why this priority**: This is the core value proposition - users need forward and backward compatibility as dbt's configuration model evolves. Without this, users face breaking changes when upgrading dbt versions.

**Independent Test**: Can be fully tested by configuring the same setting in different locations and verifying that the resolver respects the precedence order and returns the correct value regardless of dbt version.

**Acceptance Scenarios**:

1. **Given** a dbt 1.8 project with top-level `+dbt-osmosis` keys in dbt_project.yml, **When** I run dbt-osmosis, **Then** the configuration is resolved correctly from top-level keys
2. **Given** a dbt 1.11+ project where top-level keys are deprecated, **When** I specify config under `+config` or `+meta`, **Then** the configuration is resolved correctly from these nested locations
3. **Given** configuration specified in both top-level and nested locations, **When** I query for a setting, **Then** the nested location takes precedence (following the documented precedence rules)
4. **Given** a project using both dbt 1.8 and 1.11 syntax across different models, **When** I run dbt-osmosis, **Then** all models resolve their configuration correctly regardless of syntax style

---

### User Story 2 - Unified Model Property Access (Priority: P1)

As a dbt-osmosis user, I want to access model properties (descriptions, meta, tags, data types) from either the parsed manifest or the raw YAML files through a single interface so that I can choose between rendered and unrendered values without duplicating logic.

**Why this priority**: This eliminates scattered switch statements and provides a single source of truth for property access. It enables the unrendered jinja feature (doc blocks) to work consistently across the codebase.

**Independent Test**: Can be fully tested by querying the same property through the unified interface with different flags (rendered vs unrendered) and verifying the correct source is used.

**Acceptance Scenarios**:

1. **Given** a column description containing `{{ doc('my_doc') }}` in the YAML, **When** I access the property with `prefer_unrendered=True`, **Then** I receive the raw string with the jinja template intact
2. **Given** a column description in the manifest (already rendered), **When** I access the property with `prefer_unrendered=False`, **Then** I receive the rendered description from the manifest
3. **Given** both manifest and YAML available, **When** I access a column description, **Then** the system automatically selects the appropriate source based on the `prefer_unrendered` flag
4. **Given** a model with tags in both manifest and YAML, **When** I access the tags property, **Then** I receive the value from the specified source

---

### User Story 3 - Supplementary Configuration Files (Priority: P2)

As a dbt-osmosis user, I want to specify dbt-osmosis configuration in a separate `dbt-osmosis.yml` file outside of dbt_project.yml so that I can keep my dbt project configuration clean and avoid polluting the dbt hot path.

**Why this priority**: This improves user experience by providing a dedicated configuration space. It's lower priority because the existing configuration methods work, but this enhances organization.

**Independent Test**: Can be fully tested by creating a `dbt-osmosis.yml` file with configuration and verifying that settings are resolved from this file with the correct precedence relative to other sources.

**Acceptance Scenarios**:

1. **Given** a `dbt-osmosis.yml` file in the project root, **When** I run dbt-osmosis, **Then** configuration values are read from this file
2. **Given** configuration specified in both `dbt-osmosis.yml` and dbt_project.yml, **When** I query for a setting, **Then** the `dbt-osmosis.yml` value takes precedence
3. **Given** node-level configuration in model YAML and project-level in `dbt-osmosis.yml`, **When** I query for a setting on that node, **Then** the node-level configuration takes precedence
4. **Given** no `dbt-osmosis.yml` file exists, **When** I run dbt-osmosis, **Then** the system operates normally without errors (file is optional)

---

### User Story 4 - Column-Level Configuration Override (Priority: P2)

As a dbt-osmosis user, I want to override dbt-osmosis settings for specific columns through column-level meta configuration so that I can have fine-grained control over how individual columns are processed.

**Why this priority**: This provides essential flexibility for edge cases where column-specific behavior is needed. It's P2 because node-level configuration covers most use cases.

**Independent Test**: Can be fully tested by configuring a setting at the node level, then overriding it for a specific column, and verifying the column uses the overridden value.

**Acceptance Scenarios**:

1. **Given** a model with `skip-add-tags: true` at the node level, **When** I set `skip-add-tags: false` on a specific column's meta, **Then** that column has tags added while other columns do not
2. **Given** a model with `output-to-lower: true` at the node level, **When** I set `output-to-lower: false` for a specific column, **Then** that column preserves its original case
3. **Given** column-level meta configuration, **When** I query for a setting on that column, **Then** the column-level value is returned instead of the node-level value
4. **Given** no column-level override exists, **When** I query for a setting on a column, **Then** the node-level or project-level value is returned as fallback

---

### User Story 5 - Developer-Facing Resolution API (Priority: P3)

As a dbt-osmosis developer, I want a simple, consistent API for resolving configuration and model properties so that I don't need to remember which function to call or where the logic lives.

**Why this priority**: This is an internal developer experience improvement that reduces technical debt. It's P3 because the existing scattered patterns work, even if they're not ideal.

**Independent Test**: Can be fully tested by replacing existing scattered configuration access with the unified API and verifying that all tests still pass.

**Acceptance Scenarios**:

1. **Given** I need to resolve a configuration value, **When** I call `ConfigResolver.resolve(key, node)`, **Then** I receive the correct value from the highest-priority source
2. **Given** I need to access a model property, **When** I call `PropertyAccessor.get(property, node, source='yaml')`, **Then** I receive the value from the specified source
3. **Given** I need to check if a setting exists, **When** I call `ConfigResolver.has(key, node)`, **Then** I receive a boolean indicating presence
4. **Given** legacy code using `_get_setting_for_node`, **When** I migrate to the new API, **Then** the behavior remains identical (backward compatibility)

---

### Edge Cases

- What happens when a configuration key is specified in multiple sources with conflicting values?
  - **Resolution**: The highest-priority source wins according to the documented precedence chain. The system logs which source provided the value.

- What happens when a supplementary `dbt-osmosis.yml` file contains invalid YAML?
  - **Resolution**: The system fails fast with a clear error message indicating the file location and YAML parsing error. The error message suggests validating the YAML file.

- What happens when a node has no configuration set for a requested key?
  - **Resolution**: The system returns the provided fallback value (or None if no fallback). The caller is responsible for handling None values appropriately.

- What happens when accessing properties on ephemeral models that may not have YAML files?
  - **Resolution**: The system falls back to manifest-only access for ephemeral models. A warning is logged if YAML access is requested but unavailable.

- What happens when a column exists in the database but not in the YAML file?
  - **Resolution**: The property accessor returns manifest values for the column. Configuration resolution falls back to node-level settings since no column-level meta exists.

- What happens when both rendered (manifest) and unrendered (YAML) descriptions exist but are different?
  - **Resolution**: The `prefer_unrendered` flag determines which source is used. If the flag is not set, the system defaults to rendered (manifest) values for consistency.

- What happens when dbt project variables (vars) contain dbt-osmosis configuration that conflicts with other sources?
  - **Resolution**: Project-level vars have the lowest precedence (above fallback only). Node-level and column-level configuration always takes precedence over project-level vars.

- What happens when the `unrendered_config` field is not available on a node (older dbt versions)?
  - **Resolution**: The system gracefully handles missing fields by skipping that source in the resolution chain and continuing to the next source.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST resolve configuration values from all supported sources in a single call
- **FR-002**: System MUST support configuration from the following sources, in order of precedence (highest to lowest):
  1. Column-level meta (when column specified)
  2. Node-level meta
  3. Node-level config.extra
  4. Node-level config.meta (dbt 1.10+)
  5. Project-level vars in dbt_project.yml
  6. Supplementary `dbt-osmosis.yml` file (optional)
  7. Fallback value
- **FR-003**: System MUST support both kebab-case (`dbt-osmosis-setting-name`) and snake_case (`dbt_osmosis_setting_name`) key variants
- **FR-004**: System MUST support both `dbt-osmosis-` and `dbt_osmosis_` prefixes for all configuration keys
- **FR-005**: System MUST support nested configuration via `dbt-osmosis-options` and `dbt_osmosis_options` dictionaries
- **FR-006**: System MUST provide a unified interface for accessing model properties from both manifest and YAML sources
- **FR-007**: System MUST allow users to specify preference for rendered (manifest) vs unrendered (YAML) property values
- **FR-008**: System MUST support reading project-level configuration from a `dbt-osmosis.yml` file in the project root
- **FR-009**: System MUST maintain backward compatibility with the existing `_get_setting_for_node` function
- **FR-010**: System MUST log which source provided the resolved configuration value for debugging
- **FR-011**: System MUST handle missing sources gracefully (e.g., no YAML file, no unrendered_config field)
- **FR-012**: System MUST provide a `has()` method to check if a configuration key exists without providing a fallback
- **FR-013**: System MUST resolve column-level settings before falling back to node-level settings
- **FR-014**: System MUST allow access to properties that may not exist in the manifest (e.g., unrendered jinja templates)
- **FR-015**: System MUST support top-level custom keys for dbt versions < 1.11 (deprecated but supported)
- **FR-016**: System MUST support nested keys under `+config` for dbt versions >= 1.11
- **FR-017**: System MUST support keys under `+meta` at all dbt versions

### Key Entities

- **ConfigResolver**: The main configuration resolution interface that accepts a key, optional node, optional column, and fallback value; returns the resolved setting from the highest-priority source
- **PropertyAccessor**: The unified property access interface that retrieves model properties (descriptions, tags, meta, etc.) from either manifest or YAML based on a source preference
- **ConfigurationSource**: Represents a location where configuration can be specified (column meta, node meta, config.extra, project vars, supplementary file)
- **PropertySource**: Represents where a model property is stored (manifest, YAML file, database)
- **ResolutionContext**: Contains the dbt project context, manifest, YAML buffer cache, and settings needed for resolution
- **ConfigurationKey**: A setting name with optional prefix (e.g., `skip-add-tags`, `dbt-osmosis-skip-add-tags`, `dbt_osmosis_skip_add_tags`)
- **PropertyKey**: A model property identifier (e.g., `description`, `tags`, `meta`, `data_type`)
- **PrecedenceRule**: Defines the order in which sources are checked for configuration values

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Configuration resolution from any supported source completes in under 10ms per query (measured by performance benchmarks)
- **SC-002**: 100% of existing `_get_setting_for_node` call sites are migrated to the new `ConfigResolver` API without breaking existing functionality
- **SC-003**: All 32+ scattered configuration access patterns across the codebase are consolidated into a single resolution interface
- **SC-004**: 100% of test suite passes with the new resolution system, confirming backward compatibility
- **SC-005**: Users can specify configuration in any supported source (top-level, +config, +meta, supplementary file) and receive correct values according to documented precedence
- **SC-006**: Unrendered jinja templates (`{{ doc(...) }}`, `{% docs %}`) are preserved when `prefer_unrendered=True` is set
- **SC-007**: Configuration resolution works identically across dbt versions 1.8 through 1.11+ without version-specific code paths
- **SC-008**: Developer onboarding time for understanding configuration resolution is reduced by 50% (measured by documentation readability and code complexity metrics)
- **SC-009**: Zero increase in memory footprint compared to the current implementation (resolution is stateless and uses existing caches)
- **SC-010**: All configuration sources are documented with clear precedence rules in the user-facing documentation

## Assumptions

- The existing `SettingsResolver` class in `introspection.py` provides a partial implementation but only covers node/column meta and config.extra sources
- The `_get_setting_for_node` function is widely used across the codebase and must remain backward compatible
- The `unrendered_config` field is available in dbt 1.10+ nodes but may not exist in older dbt versions
- The `dbt-osmosis.yml` file is optional and its absence should not cause errors
- Property access for unrendered values requires reading raw YAML files from disk (via the existing YAML buffer cache)
- The manifest contains pre-rendered jinja templates (e.g., `{{ doc('foo') }}` is already resolved to its content)
- Users may mix and match configuration styles within a single project (e.g., some models use top-level keys, others use +config)
- Configuration precedence should follow the principle: more specific > less specific (column > node > project)
- The system should be extensible to support future configuration sources without breaking existing code
- YAML parsing errors in `dbt-osmosis.yml` should be treated as fatal errors (fail fast) to avoid silent misconfiguration
