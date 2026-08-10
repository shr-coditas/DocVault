# Security Guide

## Access reviews

An access review examines whether each account still needs its assigned permissions, whether the permissions match the person's current responsibilities, whether temporary access has expired, and whether service accounts have an identified owner. Reviewers should make decisions from current evidence rather than copying the previous review. When evidence is incomplete, the reviewer should record the missing information and assign a follow-up instead of silently approving access.

The review record should identify the system, account, reviewer, decision, supporting evidence, and completion time. A rejected entitlement should be removed through the system's normal change process. Emergency removal may use an expedited process, but the resulting change still needs a durable record.

## Credential handling

Credentials must not appear in source files, document text, screenshots, tickets, or chat messages. Examples used in training should be obvious placeholders and should not resemble working production values. If a credential is exposed, rotation takes priority over determining who caused the exposure.

### Rotation sequence

1. Create the replacement credential.
2. Update authorized consumers.
3. Verify that consumers work with the replacement.
4. Revoke the previous credential.
5. Record completion without recording either secret value.

## Audit evidence

Audit evidence should be sufficient for another reviewer to understand what happened without granting that reviewer access to unrelated confidential data. Evidence retention should follow the applicable policy, and deletion should use an approved lifecycle rather than an ad hoc local copy.
