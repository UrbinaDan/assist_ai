S: Our auth API p95 was 420ms due to DB hotspots and an N+1 query pattern.
T: Reduce p95 below 250ms while keeping correctness.
A: Added targeted indexes, rewrote the query, introduced a read-replica, and added a short cache. Used tracing with OpenTelemetry to verify impact.
R: p95 dropped to 240ms, errors -30%, infra cost -12%.
