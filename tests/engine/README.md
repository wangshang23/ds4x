# Engine tests

`ds4_test` now contains only single-request engine invariants that do not need
a model checkpoint. Full-model precision, packed checkpoint, prefill, and
long-context decode validation live under `tests/integration/`.

```bash
make ds4_test
./ds4_test --all
```

`ds4_test.c` is the standalone aggregation unit for the engine checks.

The protocol/server unit group does not require a model:

```bash
make ds4_test
./ds4_test --server
```

Model-backed groups are selected explicitly and use `DS4_TEST_MODEL` plus the
fixture variables printed by `./ds4_test --help`.
