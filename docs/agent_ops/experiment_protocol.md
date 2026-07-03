# Experiment Protocol

Every experiment record should include:

- git commit hash;
- data split and embryo grouping;
- exact command;
- config JSON/YAML;
- runtime;
- predicted nodes per dataset;
- node recall when GT exists;
- edge TP/FP/FN;
- adjusted edge Jaccard;
- division TP/FP/FN/Jaccard;
- final score;
- failure notes and screenshots/reports if useful.

Store experiment notes under `docs/experiments/` and keep raw artifacts under ignored `runs/` or `outputs/`.

