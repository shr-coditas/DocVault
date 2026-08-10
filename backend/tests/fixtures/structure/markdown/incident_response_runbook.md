# Incident Response Runbook

Use this runbook for a synthetic service called Atlas.

## Severity levels

| Severity | Definition | Initial response target |
| --- | --- | --- |
| SEV-1 | Complete outage or confirmed data exposure | 10 minutes |
| SEV-2 | Major degradation affecting many users | 30 minutes |
| SEV-3 | Limited degradation with a workaround | 4 hours |

## Initial triage

1. Confirm the alert is current.
2. Open an incident channel.
3. Assign roles:
   - Incident commander
   - Operations lead
   - Communications lead
4. Record the start time and affected components.

```bash
curl --fail http://atlas-api:8000/health/ready
```

## Database checks

```sql
SELECT state, count(*)
FROM pg_stat_activity
GROUP BY state
ORDER BY state;
```

Do not run write queries during triage unless the incident commander approves them.

## Recovery verification

```python
def recovery_is_stable(error_rates: list[float]) -> bool:
    return len(error_rates) >= 5 and max(error_rates[-5:]) < 0.01
```

### Exit criteria

- Health checks remain successful for fifteen minutes.
- Error rate remains below one percent.
- The incident timeline contains the mitigation and verification evidence.

> Preserve logs and timestamps. Never paste credentials into the incident channel.
