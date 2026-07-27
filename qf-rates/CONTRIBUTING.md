# Contributing

Use a focused branch, add tests with every behavioral change, and preserve the public header/source
separation. Public APIs use `snake_case` functions and `PascalCase` types. Inputs are validated at
module boundaries and failures use `ValidationError` or `NumericalError`.

Before opening a change:

```bash
cmake -S . -B build -DQF_WARNINGS_AS_ERRORS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
clang-format --dry-run --Werror include/qf/**/*.hpp src/**/*.cpp tests/*.cpp
```

Pricing changes must include at least one reference or property test, an explicit tolerance and a
short model-risk note when an approximation changes.

