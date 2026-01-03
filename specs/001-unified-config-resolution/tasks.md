# Tasks: Unified Configuration Resolution System

**Input**: Design documents from `/specs/001-unified-config-resolution/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/config-resolver.py

**Tests**: Tests are REQUIRED per constitution (Section IV: Test-Driven Development is NON-NEGOTIABLE). All tests must be written first and must fail before implementation.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- All paths are relative to repository root

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create tests/core/test_config_resolution.py test file structure
- [ ] T002 Create tests/core/test_property_accessor.py test file structure
- [ ] T003 [P] Add demo_duckdb/dbt-osmosis.yml example configuration file

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T004 Create ConfigurationError exception class in src/dbt_osmosis/core/introspection.py
- [ ] T005 [P] Add ConfigSourceName enum to src/dbt_osmosis/core/introspection.py
- [ ] T006 [P] Add PropertySource enum to src/dbt_osmosis/core/introspection.py
- [ ] T007 Create base ConfigurationSource abstract class in src/dbt_osmosis/core/introspection.py

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Cross-DBT-Version Configuration Compatibility (Priority: P1) 🎯 MVP

**Goal**: Resolve configuration from all sources (column meta, node meta, config.extra, config.meta, unrendered_config, project vars, dbt-osmosis.yml) with proper precedence for dbt 1.8-1.11+ compatibility

**Independent Test**: Configure same setting in different locations and verify resolver respects precedence order regardless of dbt version

### Tests for User Story 1 (TDD - Write FIRST, ensure FAIL before implementation)

- [ ] T008 [P] [US1] Write failing test for ConfigMetaSource in tests/core/test_config_resolution.py (dbt 1.10+ field handling)
- [ ] T009 [P] [US1] Write failing test for UnrenderedConfigSource in tests/core/test_config_resolution.py (dbt 1.10+ field handling)
- [ ] T010 [P] [US1] Write failing test for ProjectVarsSource in tests/core/test_config_resolution.py (vars to_dict access)
- [ ] T011 [P] [US1] Write failing test for SupplementaryFileSource in tests/core/test_config_resolution.py (dbt-osmosis.yml reading)
- [ ] T012 [P] [US1] Write failing test for full precedence chain in tests/core/test_config_resolution.py (all 7 sources in order)
- [ ] T013 [P] [US1] Write failing test for kebab-case/snake_case key normalization in tests/core/test_config_resolution.py
- [ ] T014 [P] [US1] Write failing test for prefix handling (dbt-osmosis-, dbt_osmosis_) in tests/core/test_config_resolution.py
- [ ] T015 [P] [US1] Write failing test for has() method in tests/core/test_config_resolution.py
- [ ] T016 [P] [US1] Write failing test for get_precedence_chain() method in tests/core/test_config_resolution.py
- [ ] T017 [P] [US1] Write failing test for dbt version compatibility (1.8 vs 1.11) in tests/core/test_config_resolution.py
- [ ] T018 [P] [US1] Write failing test for missing source graceful handling in tests/core/test_config_resolution.py
- [ ] T019 [P] [US1] Write failing integration test with demo_duckdb project in tests/integration/test_config_resolution_integration.py

### Implementation for User Story 1

- [ ] T020 [P] [US1] Implement ConfigMetaSource class in src/dbt_osmosis/core/introspection.py (node.config.meta access with hasattr check)
- [ ] T021 [P] [US1] Implement UnrenderedConfigSource class in src/dbt_osmosis/core/introspection.py (node.unrendered_config access with hasattr check)
- [ ] T022 [P] [US1] Implement ProjectVarsSource class in src/dbt_osmosis/core/introspection.py (runtime_cfg.vars.to_dict access)
- [ ] T023 [P] [US1] Implement SupplementaryFileSource class in src/dbt_osmosis/core/introspection.py (read dbt-osmosis.yml via _YAML_BUFFER_CACHE)
- [ ] T024 [US1] Update SettingsResolver.resolve() method in src/dbt_osmosis/core/introspection.py (add new sources to precedence chain)
- [ ] T025 [US1] Implement SettingsResolver.has() method in src/dbt_osmosis/core/introspection.py
- [ ] T026 [US1] Implement SettingsResolver.get_precedence_chain() method in src/dbt_osmosis/core/introspection.py
- [ ] T027 [US1] Add debug logging for source resolution in src/dbt_osmosis/core/introspection.py (FR-010 compliance)
- [ ] T028 [US1] Update _get_setting_for_node() to delegate to extended SettingsResolver in src/dbt_osmosis/core/introspection.py (backward compatibility)

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently - all 7 configuration sources resolve correctly with precedence

---

## Phase 4: User Story 2 - Unified Model Property Access (Priority: P1)

**Goal**: Unified interface for accessing model properties (descriptions, tags, meta) from manifest (rendered) or YAML (unrendered) sources

**Independent Test**: Query same property through unified interface with different flags and verify correct source is used

### Tests for User Story 2 (TDD - Write FIRST, ensure FAIL before implementation)

- [ ] T029 [P] [US2] Write failing test for PropertyAccessor.get() with source='manifest' in tests/core/test_property_accessor.py
- [ ] T030 [P] [US2] Write failing test for PropertyAccessor.get() with source='yaml' in tests/core/test_property_accessor.py
- [ ] T031 [P] [US2] Write failing test for PropertyAccessor.get() with source='auto' in tests/core/test_property_accessor.py (unrendered jinja detection)
- [ ] T032 [P] [US2] Write failing test for get_description() convenience method in tests/core/test_property_accessor.py
- [ ] T033 [P] [US2] Write failing test for get_meta() convenience method in tests/core/test_property_accessor.py
- [ ] T034 [P] [US2] Write failing test for has_property() method in tests/core/test_property_accessor.py
- [ ] T035 [P] [US2] Write failing test for missing YAML file graceful handling in tests/core/test_property_accessor.py
- [ ] T036 [P] [US2] Write failing test for ephemeral model (no YAML) handling in tests/core/test_property_accessor.py
- [ ] T037 [P] [US2] Write failing test for unrendered jinja template preservation in tests/core/test_property_accessor.py

### Implementation for User Story 2

- [ ] T038 [P] [US2] Create PropertyAccessor class structure in src/dbt_osmosis/core/introspection.py (accept YamlRefactorContext)
- [ ] T039 [P] [US2] Implement PropertyAccessor._get_from_manifest() method in src/dbt_osmosis/core/introspection.py
- [ ] T040 [P] [US2] Implement PropertyAccessor._get_from_yaml() method in src/dbt_osmosis/core/introspection.py (use _get_node_yaml from inheritance.py)
- [ ] T041 [US2] Implement PropertyAccessor._has_unrendered_jinja() helper in src/dbt_osmosis/core/introspection.py (detect {{ doc(, {% docs %})
- [ ] T042 [US2] Implement PropertyAccessor.get() method in src/dbt_osmosis/core/introspection.py (handle manifest/yaml/auto sources)
- [ ] T043 [US2] Implement PropertyAccessor.get_description() method in src/dbt_osmosis/core/introspection.py
- [ ] T044 [US2] Implement PropertyAccessor.get_meta() method in src/dbt_osmosis/core/introspection.py
- [ ] T045 [US2] Implement PropertyAccessor.has_property() method in src/dbt_osmosis/core/introspection.py
- [ ] T046 [US2] Add YAML file missing warning handling in src/dbt_osmosis/core/introspection.py

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently - properties accessible from both manifest and YAML with unrendered jinja preservation

---

## Phase 5: User Story 3 - Supplementary Configuration Files (Priority: P2)

**Goal**: Support dbt-osmosis.yml file for project-level configuration outside dbt's hot path

**Independent Test**: Create dbt-osmosis.yml with configuration and verify settings resolve with correct precedence

### Tests for User Story 3 (TDD - Write FIRST, ensure FAIL before implementation)

- [ ] T047 [P] [US3] Write failing test for dbt-osmosis.yml reading in tests/core/test_config_resolution.py
- [ ] T048 [P] [US3] Write failing test for dbt-osmosis.yml precedence over project vars in tests/core/test_config_resolution.py
- [ ] T049 [P] [US3] Write failing test for node-level config precedence over dbt-osmosis.yml in tests/core/test_config_resolution.py
- [ ] T050 [P] [US3] Write failing test for missing dbt-osmosis.yml handling (no errors) in tests/core/test_config_resolution.py
- [ ] T051 [P] [US3] Write failing test for invalid YAML error handling in tests/core/test_config_resolution.py
- [ ] T052 [P] [US3] Write failing integration test with demo_duckdb dbt-osmosis.yml in tests/integration/test_config_resolution_integration.py

### Implementation for User Story 3

**Note**: SupplementaryFileSource was already implemented in T023 (US1). This phase adds validation and integration.

- [ ] T053 [US3] Add ConfigurationError for invalid dbt-osmosis.yml in src/dbt_osmosis/core/introspection.py
- [ ] T054 [US3] Add dbt-osmosis.yml schema validation in src/dbt_osmosis/core/introspection.py (optional file, fail on invalid YAML)
- [ ] T055 [US3] Update demo_duckdb/dbt-osmosis.yml with comprehensive example configuration
- [ ] T056 [US3] Update demo_duckdb/dbt_project.yml with reference to dbt-osmosis.yml in comments

**Checkpoint**: At this point, dbt-osmosis.yml is fully functional - reads correctly, proper precedence, graceful missing file handling

---

## Phase 6: User Story 4 - Column-Level Configuration Override (Priority: P2)

**Goal**: Override dbt-osmosis settings for specific columns through column-level meta configuration

**Independent Test**: Configure setting at node level, override for specific column, verify column uses overridden value

### Tests for User Story 4 (TDD - Write FIRST, ensure FAIL before implementation)

- [ ] T057 [P] [US4] Write failing test for column meta taking precedence over node meta in tests/core/test_config_resolution.py
- [ ] T058 [P] [US4] Write failing test for column-specific config inheritance in tests/core/test_config_resolution.py
- [ ] T059 [P] [US4] Write failing test for column fallback to node-level when no column override in tests/core/test_config_resolution.py
- [ ] T060 [P] [US4] Write failing test for multiple columns with different overrides in tests/core/test_config_resolution.py
- [ ] T061 [P] [US4] Write failing integration test with demo_duckdb column configs in tests/integration/test_config_resolution_integration.py

### Implementation for User Story 4

**Note**: ColumnMetaSource and column precedence logic was already implemented in US1 (T020, T024). This phase adds validation.

- [ ] T062 [US4] Validate column-level override precedence in existing SettingsResolver.resolve() in src/dbt_osmosis/core/introspection.py
- [ ] T063 [US4] Add column-specific test scenarios to demo_duckdb project in demo_duckdb/models/schema.yml
- [ ] T064 [US4] Update documentation with column override examples in CLAUDE.md

**Checkpoint**: At this point, column-level configuration overrides work correctly - columns can override node-level settings

---

## Phase 7: User Story 5 - Developer-Facing Resolution API (Priority: P3)

**Goal**: Simple, consistent API for resolving configuration and model properties

**Independent Test**: Replace scattered configuration access with unified API and verify all tests still pass

### Tests for User Story 5 (TDD - Write FIRST, ensure FAIL before implementation)

- [ ] T065 [P] [US5] Write failing test for ConfigResolver public API contract in tests/core/test_config_resolution.py
- [ ] T066 [P] [US5] Write failing test for PropertyAccessor public API contract in tests/core/test_property_accessor.py
- [ ] T067 [P] [US5] Write failing test for backward compatibility with _get_setting_for_node in tests/core/test_config_resolution.py
- [ ] T068 [P] [US5] Write failing test for all 24 existing _get_setting_for_node call sites in tests/integration/test_backward_compatibility.py

### Implementation for User Story 5

- [ ] T069 [P] [US5] Add ConfigResolver and PropertyAccessor to __all__ in src/dbt_osmosis/core/introspection.py
- [ ] T070 [P] [US5] Add ConfigResolver and PropertyAccessor to __all__ in src/dbt_osmosis/core/osmosis.py (public exports)
- [ ] T071 [US5] Verify _get_setting_for_node backward compatibility wrapper in src/dbt_osmosis/core/introspection.py (ensure delegation works)
- [ ] T072 [P] [US5] Update CLAUDE.md with ConfigResolver usage documentation
- [ ] T073 [P] [US5] Update CLAUDE.md with PropertyAccessor usage documentation
- [ ] T074 [P] [US5] Add API examples to CLAUDE.md (resolve(), has(), get(), get_description())
- [ ] T075 [US5] Create configuration precedence documentation in CLAUDE.md (document 7-source precedence chain)
- [ ] T076 [US5] Add quickstart.md reference to CLAUDE.md

**Checkpoint**: At this point, developer API is complete and documented - all user stories functional with clean public interface

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T077 [P] Run full test suite and ensure 100% pass rate (task test)
- [ ] T078 [P] Run coverage check and ensure >70% coverage maintained
- [ ] T079 [P] Run ruff format and fix any formatting issues
- [ ] T080 [P] Run ruff check and fix any linting issues
- [ ] T081 [P] Verify all test files have fresh_caches fixture where needed
- [ ] T082 [P] Add type hints to all new public methods
- [ ] T083 Update CHANGELOG.md with new features and breaking changes (none expected)
- [ ] T084 [P] Add performance benchmark for ConfigResolver.resolve() (verify <10ms per query)
- [ ] T085 [P] Add memory footprint verification (ensure zero increase vs baseline)
- [ ] T086 Validate SC-002: check all 24 _get_setting_for_node call sites still work
- [ ] T087 [P] Validate SC-005: test all configuration sources with demo_duckdb
- [ ] T088 [P] Validate SC-006: verify unrendered jinja templates preserved end-to-end
- [ ] T089 [P] Validate SC-007: test across dbt versions 1.8, 1.9, 1.10 in CI matrix

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-7)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P1 → P2 → P2 → P3)
- **Polish (Phase 8)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories. Implements ConfigResolver with all 7 sources.
- **User Story 2 (P1)**: Can start after Foundational (Phase 2) - No dependencies on US1. Implements PropertyAccessor independently.
- **User Story 3 (P2)**: Depends on US1 T023 (SupplementaryFileSource already implemented) - adds validation and integration
- **User Story 4 (P2)**: Depends on US1 T020, T024 (column sources already implemented) - adds validation and examples
- **User Story 5 (P3)**: Depends on US1 (ConfigResolver) and US2 (PropertyAccessor) being complete - adds exports and documentation

### Within Each User Story

- Tests MUST be written and FAIL before implementation (TDD per constitution)
- Source classes can be implemented in parallel (marked [P])
- Core resolve() method depends on source classes
- Integration tests depend on unit tests passing

### Parallel Opportunities

**Setup Phase (Phase 1)**:
```bash
# Can run in parallel:
Task T001: Create test_config_resolution.py
Task T002: Create test_property_accessor.py
Task T003: Add demo_duckdb/dbt-osmosis.yml
```

**Foundational Phase (Phase 2)**:
```bash
# Can run in parallel after Setup:
Task T005: Add ConfigSourceName enum
Task T006: Add PropertySource enum
# (T004, T007 have dependencies)
```

**User Story 1 Tests (Phase 3)**:
```bash
# All tests can run in parallel (write first, ensure FAIL):
Task T008 through T019: All US1 test tasks
```

**User Story 1 Implementation (Phase 3)**:
```bash
# Source classes can run in parallel:
Task T020: ConfigMetaSource
Task T021: UnrenderedConfigSource
Task T022: ProjectVarsSource
Task T023: SupplementaryFileSource
# (T024-T028 have dependencies on these)
```

**User Story 2 Tests (Phase 4)**:
```bash
# All tests can run in parallel (write first, ensure FAIL):
Task T029 through T037: All US2 test tasks
```

**User Story 2 Implementation (Phase 4)**:
```bash
# Can run in parallel:
Task T039: _get_from_manifest()
Task T040: _get_from_yaml()
# (T041-T046 have dependencies)
```

**After Foundational Phase**:
```bash
# Different user stories can be worked on in parallel by different team members:
Developer A: User Story 1 (ConfigResolver)
Developer B: User Story 2 (PropertyAccessor)
# Both need Phase 2 complete, but are independent of each other
```

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together (TDD - write FIRST):
Task T008: "Write failing test for ConfigMetaSource in tests/core/test_config_resolution.py"
Task T009: "Write failing test for UnrenderedConfigSource in tests/core/test_config_resolution.py"
Task T010: "Write failing test for ProjectVarsSource in tests/core/test_config_resolution.py"
Task T011: "Write failing test for SupplementaryFileSource in tests/core/test_config_resolution.py"
# ... and so on for T012-T019
# Ensure all tests FAIL before proceeding to implementation

# Launch all source classes for User Story 1 together:
Task T020: "Implement ConfigMetaSource class in src/dbt_osmosis/core/introspection.py"
Task T021: "Implement UnrenderedConfigSource class in src/dbt_osmosis/core/introspection.py"
Task T022: "Implement ProjectVarsSource class in src/dbt_osmosis/core/introspection.py"
Task T023: "Implement SupplementaryFileSource class in src/dbt_osmosis/core/introspection.py"
# All independent - can run in parallel
```

---

## Implementation Strategy

### MVP First (User Story 1 Only - Core ConfigResolver)

**Minimum Viable Product**: Cross-dbt-version configuration compatibility

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
   - Write all tests (T008-T019) - ensure they FAIL
   - Implement all sources (T020-T023) - in parallel
   - Integrate sources into resolver (T024-T028)
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Verify: All 7 configuration sources resolve correctly with precedence

**MVP Delivers**: Configuration resolution from all 7 sources works across dbt 1.8-1.11+

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!) ✅
3. Add User Story 2 → Test independently → Deploy/Demo ✅
4. Add User Story 3 → Test independently → Deploy/Demo ✅
5. Add User Story 4 → Test independently → Deploy/Demo ✅
6. Add User Story 5 → Test independently → Deploy/Demo ✅
7. Polish → Final release

Each story adds value without breaking previous stories.

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (ConfigResolver) - Phase 3
   - Developer B: User Story 2 (PropertyAccessor) - Phase 4
3. Once US1 and US2 complete:
   - Developer A: User Story 3 (Supplementary file validation) - Phase 5
   - Developer B: User Story 4 (Column validation) - Phase 6
4. Team completes User Story 5 (Documentation) together - Phase 7
5. Team completes Polish together - Phase 8

Stories complete and integrate independently.

---

## Summary

- **Total Tasks**: 89
- **Setup Tasks**: 3
- **Foundational Tasks**: 4
- **User Story 1 (P1)**: 22 tasks (11 tests + 11 implementation)
- **User Story 2 (P1)**: 18 tasks (9 tests + 9 implementation)
- **User Story 3 (P2)**: 10 tasks (6 tests + 4 implementation)
- **User Story 4 (P2)**: 8 tasks (5 tests + 3 implementation)
- **User Story 5 (P3)**: 12 tasks (4 tests + 8 implementation)
- **Polish**: 13 tasks

**Parallel Opportunities**: 42 tasks marked [P] can run in parallel with appropriate team size

**Independent Test Criteria**:
- US1: Configure same setting in different locations → verify precedence
- US2: Query property with different flags → verify source selection
- US3: Create dbt-osmosis.yml → verify file read and precedence
- US4: Configure node + column → verify column override
- US5: Call new APIs → verify they work and old code still works

**Suggested MVP Scope**: User Story 1 (Phase 3) - Core ConfigResolver with all 7 sources. This delivers the primary value: configuration resolution across dbt versions 1.8-1.11+ with full precedence support.

**Format Validation**: All tasks follow checklist format: `- [ ] [ID] [P?] [Story?] Description with file path`
