# Engine tests

`ds4_test.c` is a thin test-suite aggregation unit. Its `parts/` directory
separates CUDA/shared helpers, retained Apple reference fixtures, model/vector
tests, speculative decoding checks, and the test runner.

The protocol/server unit group does not require a model:

```bash
make ds4_test
./ds4_test --server
```

Model-backed groups are selected explicitly and use `DS4_TEST_MODEL` plus the
fixture variables printed by `./ds4_test --help`.
