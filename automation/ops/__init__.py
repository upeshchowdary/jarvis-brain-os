"""PRIVATE operation implementations, registered with ``automation.gateway``.

Do not import these modules from anywhere except ``automation.gateway`` — call
``gateway.execute(ActionRequest(...))`` instead. ``tests/test_gateway_isolation.py``
fails if any other module imports ``automation.ops``.
"""

from automation.ops import apps, computer, messaging, shell, system  # noqa: F401
