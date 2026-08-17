"""Handing a file, a folder or a whole Drive over to somebody else.

Ownership in Drive is two independent things, and a transfer has to move both:

1. ``File.owner`` — the framework field. ``get_user_access_for_user`` short-circuits
   on it, granting the owner everything and bypassing any ``deny`` on the path. It
   is per entity: it does not inherit to children.
2. The ``Drive Permission`` row on the owner's private root folder, planted by
   ``grant_owner_access``. *This* is what actually makes the files inside reachable,
   because it inherits all the way down. Ordinary files get no row of their own.

So flipping ``owner`` alone grants the recipient access without taking it from the
giver — the subtree is still sitting under the giver's root folder, whose row keeps
granting them everything. A transfer is only real once the subtree leaves that
folder, which is why ``transfer_ownership`` reparents as well as reassigns.

Files that live *outside* the giver's private folder — in the shared ``Drive`` tree,
or inside a folder someone else shared with them — are not in anyone's private space,
so there is nothing to move: reassigning ``owner`` is the whole transfer.
"""

from contextlib import contextmanager

import frappe
from frappe import _

from suite.drive.api.activity import create_new_activity_log
from suite.drive.api.notifications import create_notification
from suite.drive.api.permissions import is_drive_admin
from suite.drive.api.storage import acquire_owner_storage_lock, get_storage_usage
from suite.drive.utils import (
    APP_FOLDERS,
    ATTACHMENT_CONTENT_DOCTYPE,
    FRAMEWORK_FOLDERS,
    GENERAL_USER,
    GROUP_PREFIX,
    PERMISSION_TYPES,
    ROOT_FOLDER,
    STATUS_ACTIVE,
    USERS_FOLDER,
    WRITER_CONTENT_DOCTYPE,
    create_drive_file,
    get_ancestors_of,
    get_new_file_name,
    get_user_folder,
    grant_owner_access,
)
from suite.drive.utils.files import FileManager, storage_key

# What the previous owner is left with when `keep_previous_access` is on: they can
# still open what they handed over, and nothing more. Not `write` — a handover that
# left them able to keep changing the file would make the new owner's control
# nominal; not `share` — passing it on again is the owner's prerogative.
# `comment` rides with `read` to match what the share dialog calls "Can view"
# (`ShareDialog.getAccess`), so the row round-trips through that dropdown unchanged.
VIEWER_ACCESS = dict.fromkeys(PERMISSION_TYPES, 0) | {"read": 1, "comment": 1}


@contextmanager
def _elevated():
    """Run one step as Administrator.

    `File.move()` authorizes the *session* user against the destination folder, which
    is the wrong question during a transfer: the giver legitimately holds no `upload`
    on the recipient's private folder, and the operation as a whole was already
    authorized by `_assert_can_transfer`. Only `session.user` is swapped, not
    `frappe.set_user` — that also clears `form_dict` and the session data, which we
    are in the middle of serving a request from.
    """
    original = frappe.session.user
    frappe.session.user = "Administrator"
    try:
        yield
    finally:
        frappe.session.user = original


@contextmanager
def _quiet_share_notifications():
    """Suppress `DrivePermission.after_insert`'s share email for rows this module
    plants. The recipient gets an explicit `Transfer` notification instead, and the
    previous owner should not be emailed "someone shared a file with you" about a
    file they just gave away — least of all during an offboarding, when they may
    already be gone."""
    previous = frappe.flags.get("in_ownership_transfer")
    frappe.flags.in_ownership_transfer = True
    try:
        yield
    finally:
        frappe.flags.in_ownership_transfer = previous


def _subtree(root: str) -> list[frappe._dict]:
    """`root` and every descendant, in one pass.

    Trashed children are included deliberately: they are still the owner's, they
    still show on their Trash page (`api/list.py` filters it by `owner`), and leaving
    them behind would strand them in the previous owner's trash with no way back.
    """
    return frappe.db.sql(
        """
        WITH RECURSIVE tree AS (
            SELECT name FROM `tabFile` WHERE name = %(root)s
            UNION ALL
            SELECT f.name FROM `tabFile` f JOIN tree ON f.folder = tree.name
        )
        SELECT
            f.name, f.owner, f.is_folder, f.status, f.file_size,
            f.file_type, f.file_url, f.content_doctype, f.content_docname
        FROM `tabFile` f JOIN tree ON f.name = tree.name
        """,
        {"root": root},
        as_dict=True,
    )


# ----------------------------------------------------------------------------
# Blob integrity
#
# On non-flat storage the backend key *is* the path — `get_disk_path` builds it
# from the parent's key plus the file name — so relocating a subtree rewrites
# every key under it. S3 has no rename, so `recursive_path_move` issues one
# copy_object + delete_object per descendant, interleaved with DB writes and
# spanning no transaction. Interrupt it halfway and the objects already copied
# sit at their new keys while the rolled-back rows still name the old ones:
# readable bytes, unreachable files.
#
# Two defences, because neither is sufficient alone:
#   `_verify_blobs` refuses to start a doomed relocation (cheap, catches the
#   common case of data that was already missing), and `_capture_urls` /
#   `_repair_urls` clean up after one that died in flight.
# ----------------------------------------------------------------------------


def _blob_expected(row: frappe._dict) -> bool:
    """Whether this row should have a blob sitting at its `file_url` right now.

    Mirrors `File._not_in_disk`, plus trashed rows: their bytes live under
    `.trash/<id>`, and `recursive_path_move` deliberately leaves them there.
    """
    if row.status != STATUS_ACTIVE or not row.file_url or row.file_type == "Link":
        return False
    # Slides/Sheets keep their content in their own doctype; only Writer's is a blob.
    return not (row.content_doctype and row.content_doctype != WRITER_CONTENT_DOCTYPE)


def _blob_exists(manager, file_url: str) -> bool:
    key = storage_key(file_url)
    if manager.s3_enabled:
        try:
            manager.conn.head_object(Bucket=manager.bucket, Key=key)
            return True
        except Exception:
            # A folder marker is stored with a trailing slash; a bare miss on the
            # unslashed form is not proof of absence.
            try:
                manager.conn.head_object(Bucket=manager.bucket, Key=key.rstrip("/") + "/")
                return True
            except Exception:
                return False
    try:
        return manager.get_local_path(file_url).exists()
    except Exception:
        return False


def _verify_blobs(rows: list[frappe._dict]) -> None:
    """Refuse a relocation whose bytes are not where the database says they are.

    Better to decline before touching anything than to discover it four objects
    into a subtree, with the earlier ones already moved.
    """
    manager = FileManager()
    missing = [r.name for r in rows if _blob_expected(r) and not _blob_exists(manager, r.file_url)]
    if missing:
        frappe.throw(
            _("{0} file(s) are missing from storage and cannot be moved: {1}").format(
                len(missing), ", ".join(missing[:5])
            ),
            frappe.ValidationError,
        )


def _capture_urls(names: list[str]) -> dict[str, str]:
    """Where the (possibly half-finished) move believes each blob now lives.

    Read *before* the rollback: afterwards this information is gone, and without
    it a partially relocated subtree cannot be pointed back at its own bytes.
    """
    if not names:
        return {}
    rows = frappe.get_all("File", filters={"name": ("in", names)}, fields=["name", "file_url"])
    return {r.name: r.file_url for r in rows}


def _repair_urls(attempted: dict[str, str]) -> list[str]:
    """Re-point rows whose bytes were relocated before the move failed.

    Runs after the rollback, so the rows name their original keys again. Where
    that key is now empty and the attempted one holds the object, the row is
    corrected — the file stays readable and the item can simply be retried.
    """
    if not attempted:
        return []
    manager = FileManager()
    repaired = []
    for name, moved_url in attempted.items():
        row = frappe.db.get_value(
            "File", name, ["file_url", "status", "file_type", "content_doctype"], as_dict=True
        )
        if not row or not moved_url or row.file_url == moved_url:
            continue
        if not _blob_expected(frappe._dict(row, name=name)):
            continue
        if _blob_exists(manager, row.file_url) or not _blob_exists(manager, moved_url):
            continue
        frappe.db.set_value("File", name, "file_url", moved_url, update_modified=False)
        repaired.append(name)
    return repaired


def _active_bytes(rows: list[frappe._dict]) -> int:
    return sum(r.file_size or 0 for r in rows if not r.is_folder and r.status == STATUS_ACTIVE)


def _user_root(user: str) -> str | None:
    """The user's private root folder, read straight off `Drive Settings`.

    Deliberately not `get_user_folder`, which creates one on a miss — during an
    offboarding the source user may already be deleted, and conjuring a fresh folder
    for them is the opposite of what we want.
    """
    return frappe.db.get_value("Drive Settings", user, "user_folder")


def _is_inside(entity: str, ancestor: str | None) -> bool:
    return bool(ancestor) and ancestor in get_ancestors_of(entity)


def _is_structural(entity: str) -> bool:
    """Scaffolding that belongs to the site rather than to a person.

    Drive's two roots and the framework/app folders are shared by everyone, and a
    user's home folder is bound to them by `Drive Settings.user_folder` — handing one
    over would leave two users pointing at the same folder. Their *contents* transfer
    perfectly well; that is exactly what `transfer_all_owned` does.
    """
    if entity in (ROOT_FOLDER, USERS_FOLDER) or entity in FRAMEWORK_FOLDERS or entity in APP_FOLDERS:
        return True
    return bool(frappe.db.exists("Drive Settings", {"user_folder": entity}))


def _assert_valid_recipient(new_owner: str) -> None:
    if not new_owner or new_owner in ("", GENERAL_USER, "Guest") or new_owner.startswith(GROUP_PREFIX):
        frappe.throw(
            _("{0} is not a user and cannot own files.").format(new_owner or "''"),
            frappe.ValidationError,
        )
    if not frappe.db.exists("User", new_owner):
        frappe.throw(_("No such user: {0}").format(new_owner), frappe.DoesNotExistError)
    if not frappe.db.get_value("User", new_owner, "enabled"):
        frappe.throw(_("{0} is disabled and cannot receive files.").format(new_owner), frappe.ValidationError)


def _assert_can_transfer(doc, new_owner: str, rows: list[frappe._dict]) -> None:
    caller = frappe.session.user

    # Ownership is strictly more than `share`: the owner bypasses denies, and only
    # they can give the file away again. A share-holder must not be able to hand
    # somebody else's file to a third party — same reasoning as the grant ceiling.
    if not (is_drive_admin(caller) or doc.owner == caller):
        frappe.throw(
            _("Only the owner of this file can transfer it."),
            frappe.PermissionError,
        )

    _assert_valid_recipient(new_owner)

    if doc.owner == new_owner:
        frappe.throw(_("{0} already owns this.").format(new_owner), frappe.ValidationError)

    if _is_structural(doc.name):
        frappe.throw(
            _("Structural and home folders cannot be transferred. Transfer what's inside instead."),
            frappe.PermissionError,
        )

    if doc.status != STATUS_ACTIVE:
        frappe.throw(_("Only active files can be transferred."), frappe.ValidationError)

    _assert_within_quota(new_owner, _active_bytes(rows))


def _assert_within_quota(new_owner: str, incoming: int) -> None:
    """Storage is billed to `File.owner` (`api/storage.py`), so a transfer moves the
    cost onto the recipient. Checked up front against the whole subtree rather than
    per file, so a large handover fails before it has moved half of itself."""
    usage = get_storage_usage(new_owner)
    if usage["limit"] and (usage["limit"] - usage["total_size"]) < incoming:
        frappe.throw(
            _("{0} does not have enough storage to receive this ({1} needed).").format(
                new_owner, _human_bytes(incoming)
            ),
            ValueError,
        )


def _human_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size) < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0


def _reassign(rows: list[frappe._dict], new_owner: str) -> None:
    """Point every row in the subtree at the new owner, content documents included.

    `content_has_permission` already delegates access to the backing File, so the
    content document's own `owner` does not gate access — but it does drive listings
    in every content app: Sheets sorts, filters and trashes on `tabSheet.owner`,
    Slides lists on `Presentation.owner`, Writer's home lists on `File.owner`. Leave
    it behind and a transferred document keeps showing up for its previous owner.
    """
    for row in rows:
        frappe.db.set_value("File", row.name, "owner", new_owner, update_modified=False)
        if (
            row.content_doctype
            and row.content_docname
            and row.content_doctype != ATTACHMENT_CONTENT_DOCTYPE
            and frappe.db.exists(row.content_doctype, row.content_docname)
        ):
            frappe.db.set_value(
                row.content_doctype, row.content_docname, "owner", new_owner, update_modified=False
            )


def _grant(entity: str, user: str, access: dict) -> None:
    """Insert or update one `Drive Permission` row, bypassing `File.share`.

    `share()` resolves the grant ceiling against the *session* user, which asks the
    wrong question here for the same reason `_elevated` exists: mid-transfer the
    giver may already have lost the levels being granted.
    """
    name = frappe.db.get_value("Drive Permission", {"entity": entity, "user": user})
    if name:
        perm = frappe.get_doc("Drive Permission", name)
    else:
        perm = frappe.new_doc("Drive Permission")
        perm.update({"entity": entity, "user": user})
    perm.update({**access, "deny": 0})
    perm.save(ignore_permissions=True)


def _revoke(entity: str, user: str) -> None:
    name = frappe.db.get_value("Drive Permission", {"entity": entity, "user": user})
    if name:
        frappe.delete_doc("Drive Permission", name, ignore_permissions=True)


@frappe.whitelist()
def transfer_ownership(entity: str, new_owner: str, keep_previous_access: bool = True) -> dict:
    """Hand `entity` and everything under it to `new_owner`.

    Returns a summary of what moved.
    """
    # No manual coercion of `keep_previous_access`: `frappe.whitelist` wraps every
    # method in `validate_argument_types`, which validates each argument against its
    # annotation with pydantic before the body runs.
    doc = frappe.get_doc("File", entity)
    rows = _subtree(doc.name)
    _assert_can_transfer(doc, new_owner, rows)

    old_owner = doc.owner
    acquire_owner_storage_lock(old_owner)
    acquire_owner_storage_lock(new_owner)

    return _do_transfer(doc, rows, old_owner, new_owner, keep_previous_access)


def _do_transfer(doc, rows, old_owner, new_owner, keep_previous_access, destination=None) -> dict:
    """The transfer itself, with the authorization already established.

    Split out so the bulk path can reuse it per item without re-running the
    per-entity permission check against a session user who may be neither party.
    """
    source_root = _user_root(old_owner)
    reparent = _is_inside(doc.name, source_root)

    with _quiet_share_notifications():
        if reparent:
            # `move()` owns the parts that are easy to get wrong: the blob relocation
            # (a real copy on S3, and `recursive_path_move` for every descendant),
            # the `file_name` dedup if the destination already has that name, and the
            # `update_file_size` rollups on both parents. Never reproduce it by hand.
            destination = destination or get_user_folder(new_owner).name
            # Flat storage keys by id, so nothing relocates and there is nothing to
            # verify; checking anyway would refuse transfers over pre-existing gaps
            # that a flat-mode transfer does not care about.
            if not FileManager().flat:
                _verify_blobs(rows)
            with _elevated():
                doc.move(destination)

        _reassign(rows, new_owner)

        # Mirror what a user's own root folder carries, so the transfer survives the
        # file later being moved somewhere that grants the new owner nothing.
        grant_owner_access(doc.name, new_owner)

        if reparent:
            # Only the reparent takes access away. Left where it was, the previous
            # owner's access came from rows we never touched.
            if keep_previous_access:
                _grant(doc.name, old_owner, VIEWER_ACCESS)
            else:
                _revoke(doc.name, old_owner)

    _log_transfer(doc, old_owner, new_owner)
    frappe.db.set_value("File", doc.name, "file_modified", frappe.utils.now(), update_modified=False)

    return {
        "entity": doc.name,
        "previous_owner": old_owner,
        "new_owner": new_owner,
        "files": len(rows),
        "bytes": _active_bytes(rows),
        "moved": bool(reparent),
        "destination": destination if reparent else None,
    }


def _log_transfer(doc, old_owner: str, new_owner: str) -> None:
    actor = frappe.db.get_value("User", frappe.session.user, "full_name") or frappe.session.user
    to_name = frappe.db.get_value("User", new_owner, "full_name") or new_owner
    create_new_activity_log(
        entity=doc.name,
        activity_type="transfer",
        activity_message=f"{actor} transferred ownership of {doc.file_name} to {to_name}",
        document_field="owner",
        field_old_value=old_owner,
        field_new_value=new_owner,
    )
    create_notification(
        frappe.session.user,
        new_owner,
        "Transfer",
        doc,
        f"{actor} made you the owner of {doc.file_name}",
    )


@frappe.whitelist()
def get_transfer_preview(entity: str) -> dict:
    """What a transfer of `entity` would move — for the confirmation dialog."""
    doc = frappe.get_doc("File", entity)
    if not (is_drive_admin() or doc.owner == frappe.session.user):
        frappe.throw(_("Only the owner of this file can transfer it."), frappe.PermissionError)
    rows = _subtree(doc.name)
    return {
        "entity": doc.name,
        "file_name": doc.file_name,
        "owner": doc.owner,
        "files": len([r for r in rows if not r.is_folder]),
        "folders": len([r for r in rows if r.is_folder]),
        "bytes": _active_bytes(rows),
        "leaves_your_drive": _is_inside(doc.name, _user_root(doc.owner)),
    }


# ----------------------------------------------------------------------------
# Bulk handover
# ----------------------------------------------------------------------------


@frappe.whitelist()
def transfer_all_owned(from_user: str, to_user: str, keep_previous_access: bool = True) -> str:
    """Queue a handover of everything `from_user` owns to `to_user`.

    Admin-only: this reassigns somebody's entire private Drive, which no ordinary
    user should be able to do to another. Returns the `Drive Ownership Transfer`
    name to poll.
    """
    if not is_drive_admin():
        frappe.throw(_("Only a Drive admin can transfer another user's files."), frappe.PermissionError)

    _assert_valid_recipient(to_user)
    if from_user == to_user:
        frappe.throw(_("Cannot hand a user's files over to themselves."), frappe.ValidationError)

    running = frappe.db.exists(
        "Drive Ownership Transfer", {"from_user": from_user, "status": ["in", ("Queued", "Running")]}
    )
    if running:
        frappe.throw(
            _("A handover for {0} is already in progress ({1}). Retry that one instead.").format(
                from_user, running
            ),
            frappe.ValidationError,
        )

    transfer = frappe.get_doc(
        {
            "doctype": "Drive Ownership Transfer",
            "from_user": from_user,
            "to_user": to_user,
            "keep_previous_access": int(bool(keep_previous_access)),
        }
    ).insert(ignore_permissions=True)

    frappe.enqueue(
        "suite.drive.api.ownership.run_bulk_transfer",
        queue="long",
        timeout=6 * 60 * 60,
        job_id=f"drive_transfer_{transfer.name}",
        deduplicate=True,
        enqueue_after_commit=True,
        transfer=transfer.name,
    )
    return transfer.name


@frappe.whitelist()
def retry_transfer(transfer: str) -> str:
    """Re-queue a handover that failed or was interrupted.

    The point of re-deriving the work list every run is that this is safe: whatever
    already changed hands is simply no longer pending. It also releases a row stuck
    on `Running` after a worker was killed, which would otherwise block the user's
    next handover forever through the concurrency check.
    """
    if not is_drive_admin():
        frappe.throw(_("Only a Drive admin can transfer another user's files."), frappe.PermissionError)

    doc = frappe.get_doc("Drive Ownership Transfer", transfer)
    if doc.status == "Completed":
        frappe.throw(_("That handover already finished."), frappe.ValidationError)

    doc.db_set("status", "Queued", update_modified=False)
    frappe.enqueue(
        "suite.drive.api.ownership.run_bulk_transfer",
        queue="long",
        timeout=6 * 60 * 60,
        job_id=f"drive_transfer_{doc.name}",
        deduplicate=True,
        enqueue_after_commit=True,
        transfer=doc.name,
    )
    return doc.name


@frappe.whitelist()
def get_transfer_status(transfer: str) -> dict:
    if not is_drive_admin():
        frappe.throw(_("Not permitted."), frappe.PermissionError)
    return frappe.db.get_value(
        "Drive Ownership Transfer",
        transfer,
        [
            "name",
            "from_user",
            "to_user",
            "status",
            "total_files",
            "files_moved",
            "files_failed",
            "files_repaired",
            "destination_folder",
            "error_log",
        ],
        as_dict=True,
    )


def _pending_items(from_user: str, source_root: str | None) -> list[str]:
    """Top-level things still to hand over, re-derived rather than remembered.

    Two groups, in order:
      1. children of the source user's own root folder — these get relocated;
      2. anything else they still own — files in the shared tree, or inside a folder
         somebody else shared with them. Those are not in anyone's private space, so
         they are reassigned where they stand (see `_do_transfer`).

    Descendants are excluded: transferring a folder carries everything under it, and
    listing children separately would transfer them twice.
    """
    items = []
    if source_root:
        items += frappe.get_all(
            "File",
            filters={"folder": source_root, "owner": from_user, "status": STATUS_ACTIVE},
            pluck="name",
            order_by="is_folder desc, file_name asc",
        )

    outside = frappe.get_all(
        "File",
        filters={"owner": from_user, "status": STATUS_ACTIVE, "name": ["not in", items or [""]]},
        pluck="name",
    )
    # Keep only the topmost of any chain — a file inside an already-listed folder
    # rides along with its parent.
    listed = set(items)
    outside_set = set(outside)
    for name in outside:
        ancestors = set(get_ancestors_of(name))
        if ancestors & listed or ancestors & outside_set:
            continue
        if source_root and source_root in ancestors:
            continue
        # Someone else's home folder, or site scaffolding they happen to own — the
        # per-file path refuses these and so must this one.
        if _is_structural(name):
            continue
        items.append(name)
        listed.add(name)
    return [n for n in items if not _is_structural(n)]


def _destination_folder(transfer) -> str:
    """The single folder in the recipient's Drive that receives the handover.

    Created lazily and remembered on the transfer row, so a resumed run drops its
    remaining items into the same place rather than making a second folder.
    """
    if transfer.destination_folder and frappe.db.exists("File", transfer.destination_folder):
        return transfer.destination_folder

    from suite.drive.utils.files import FileManager

    root = get_user_folder(transfer.to_user).name
    # The source user may be gone; fall back to the address rather than "None's files".
    label = frappe.db.get_value("User", transfer.from_user, "full_name") or transfer.from_user
    title = get_new_file_name(f"{label}'s files", root, True)

    manager = FileManager()
    folder = create_drive_file(
        title, root, "Folder", lambda f: manager.create_folder(f), owner=transfer.to_user
    )
    with _quiet_share_notifications():
        grant_owner_access(folder.name, transfer.to_user)
    transfer.db_set("destination_folder", folder.name, update_modified=False)
    return folder.name


def run_bulk_transfer(transfer: str) -> None:
    """Job body. Safe to re-run: the work list is re-derived every time.

    Anything that escapes the per-item handler kills the whole run, so the row is
    marked `Failed` rather than being left on `Running` forever — a stuck `Running`
    would block every future handover for that user through the concurrency check.
    """
    try:
        _run_bulk_transfer(transfer)
    except Exception as exc:
        frappe.db.rollback()
        doc = frappe.get_doc("Drive Ownership Transfer", transfer)
        doc.record_error("<whole run>", exc)
        doc.db_set("status", "Failed", update_modified=False)
        frappe.log_error(f"Drive handover {transfer} aborted", frappe.get_traceback())
        frappe.db.commit()
        raise


def _run_bulk_transfer(transfer: str) -> None:
    doc = frappe.get_doc("Drive Ownership Transfer", transfer)
    doc.db_set("status", "Running", update_modified=False)

    source_root = _user_root(doc.from_user)
    items = _pending_items(doc.from_user, source_root)
    doc.db_set("total_files", len(items), update_modified=False)
    frappe.db.commit()

    if not items:
        doc.db_set("status", "Completed", update_modified=False)
        frappe.db.commit()
        return

    destination = _destination_folder(doc)
    frappe.db.commit()

    for name in items:
        rows = []
        try:
            entity = frappe.get_doc("File", name)
            rows = _subtree(entity.name)
            _assert_within_quota(doc.to_user, _active_bytes(rows))
            # Not elevated here: the job runs as the admin who queued it, `move()`'s
            # own check clears them via `is_drive_admin`, and leaving the session user
            # alone keeps the activity log naming the person who ran the handover
            # rather than "Administrator".
            _do_transfer(
                entity,
                rows,
                doc.from_user,
                doc.to_user,
                # Granted once on the destination folder below instead, so the previous
                # owner gets one row and one notification rather than one per item.
                keep_previous_access=False,
                destination=destination,
            )
            doc.db_set("files_moved", (doc.files_moved or 0) + 1, update_modified=False)
            # Per item, so a crash keeps everything already moved and the rest simply
            # reappears in `_pending_items` on the next run.
            frappe.db.commit()
        except Exception as exc:
            # Capture before the rollback, or the only record of where a half-finished
            # relocation put the bytes is thrown away with it. On S3 that is the
            # difference between a retryable item and permanently unreachable files.
            attempted = _capture_urls([r.name for r in rows]) if rows else {}
            frappe.db.rollback()
            repaired = _repair_urls(attempted)
            doc.reload()
            doc.record_error(name, exc, repaired=repaired)
            frappe.log_error(f"Drive handover {doc.name}: {name}", frappe.get_traceback())
            frappe.db.commit()

    if doc.keep_previous_access and frappe.db.exists("User", doc.from_user):
        # One grant on the container covers everything inside it: nearest row wins,
        # and nothing between it and the leaves names the previous owner.
        with _quiet_share_notifications():
            _grant(destination, doc.from_user, VIEWER_ACCESS)

    doc.reload()
    doc.db_set(
        "status",
        "Completed With Errors" if doc.files_failed else "Completed",
        update_modified=False,
    )
    frappe.db.commit()
