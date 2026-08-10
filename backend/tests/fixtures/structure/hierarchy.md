# Employee Handbook

> This fixture is synthetic and contains no sensitive data.

## Leave

Submit these items:

1. A request form
   - Include the requested dates
   - Name the approver
2. A coverage plan

| Leave type | Days |
| --- | ---: |
| Annual | 20 |
| Personal | 5 |

```python
def days_available(annual: int, used: int) -> int:
    return annual - used
```

## Conduct

Respect colleagues and protect confidential information.
