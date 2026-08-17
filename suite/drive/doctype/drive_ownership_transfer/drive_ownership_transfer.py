# Copyright (c) 2026, Frappe Technologies Pvt. Ltd. and contributors
# For license information, please see license.txt

import frappe
from frappe.model.document import Document

MAX_ERROR_LOG = 20_000


class DriveOwnershipTransfer(Document):
    """Audit row and resume point for a bulk handover.

    The work list is never stored: `_pending_items` re-derives it from the
    source folder on every run, so a job that dies halfway can simply be run
    again and will pick up whatever is still owned by `from_user`.
    """

    def validate(self):
        if self.from_user == self.to_user:
            frappe.throw("Cannot hand a user's files over to themselves.", frappe.ValidationError)
        if not frappe.db.exists("User", self.to_user):
            frappe.throw(f"No such user: {self.to_user}", frappe.DoesNotExistError)
        if frappe.db.get_value("User", self.to_user, "enabled") == 0:
            frappe.throw(f"{self.to_user} is disabled and cannot receive files.", frappe.ValidationError)

    def record_error(self, entity: str, exc: Exception, repaired: list[str] | None = None) -> None:
        """Append one failure and keep going. Truncated from the front so a run
        that fails on every item still ends with a readable tail.

        `repaired` names rows whose `file_url` had to be re-pointed at bytes a
        half-finished relocation had already moved. Logged rather than silently
        fixed: it is the one thing here that changed data outside the failed item,
        and an admin reading this log should see it.
        """
        line = f"[{frappe.utils.now()}] {entity}: {type(exc).__name__}: {exc}"
        if repaired:
            line += f" | re-pointed {len(repaired)} relocated blob(s): {', '.join(repaired[:5])}"
            self.db_set("files_repaired", (self.files_repaired or 0) + len(repaired), update_modified=False)
        log = f"{self.error_log or ''}\n{line}".strip()
        self.db_set("error_log", log[-MAX_ERROR_LOG:], update_modified=False)
        self.db_set("files_failed", (self.files_failed or 0) + 1, update_modified=False)
