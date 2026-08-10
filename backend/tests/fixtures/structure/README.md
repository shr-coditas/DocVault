# Structure fixture corpus

These files are synthetic, non-sensitive fixtures for deterministic extraction,
chunking, and retrieval evaluation. DOCX and PDF variants are generated in tests
so their binary contents remain reproducible. Tests also generate multi-page,
repeated-band, empty, malformed, and image-only PDF cases.

The `markdown/` directory contains upload-ready examples for manual testing:

- hierarchical policy content with lists, quotes, and tables;
- an incident runbook with Bash, SQL, and Python fences;
- a wide equipment table;
- deeply nested project sections and lists;
- long paragraphs for token-boundary testing;
- mixed-language text for evaluation experiments.
