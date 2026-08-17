"""Ownership transfer: per-file and bulk.

The load-bearing assertion in this file is `test_previous_owner_loses_access`: an
implementation that only flips `File.owner` passes almost everything else here and
still leaves the previous owner with full access, because the subtree is still
sitting under their home folder. See `suite/drive/api/ownership.py`.
"""

from unittest.mock import patch

import frappe
from frappe.tests import IntegrationTestCase

from suite.drive.api.list import trash
from suite.drive.api.ownership import (
    _pending_items,
    _subtree,
    get_transfer_preview,
    run_bulk_transfer,
    transfer_all_owned,
    transfer_ownership,
)
from suite.drive.api.permissions import get_user_access_for_user
from suite.drive.api.storage import get_storage_usage
from suite.drive.utils import (
    GENERAL_USER,
    GROUP_PREFIX,
    ROOT_FOLDER,
    STATUS_ACTIVE,
    STATUS_TRASHED,
    USERS_FOLDER,
    create_drive_file,
    get_ancestors_of,
    get_root_folder,
    get_user_folder,
)
from suite.drive.utils.files import FileManager, storage_key
from suite.tests.utils import ensure_user

ALICE = "drive-xfer-alice@example.com"
BOB = "drive-xfer-bob@example.com"
CAROL = "drive-xfer-carol@example.com"
ADMIN = "drive-xfer-admin@example.com"


class OwnershipTestBase(IntegrationTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        for email in (ALICE, BOB, CAROL, ADMIN):
            ensure_user(email)
        admin = frappe.get_doc("User", ADMIN)
        if "Suite Admin" not in [r.role for r in admin.roles]:
            admin.append("roles", {"role": "Suite Admin"})
            admin.save(ignore_permissions=True)
        cls.alice_home = get_user_folder(ALICE).name
        cls.bob_home = get_user_folder(BOB).name
        cls.carol_home = get_user_folder(CAROL).name

    def setUp(self):
        frappe.flags.mute_drive_activity_log = True
        self.addCleanup(self._reset_flags)

    def _reset_flags(self):
        frappe.flags.mute_drive_activity_log = False
        frappe.flags.in_ownership_transfer = None

    # -- fixtures ---------------------------------------------------------
    def make_folder(self, parent, name=None, owner=ALICE):
        manager = FileManager()
        folder = create_drive_file(
            name or frappe.generate_hash(8),
            parent,
            "Folder",
            lambda f: manager.create_folder(f),
            owner=owner,
        )
        # The rollback between tests reverts the rows but not the directory this
        # just made. Left behind, the next run's `shutil.move` finds its destination
        # already occupied and fails — a collision that cannot happen in production,
        # where `move()` dedupes `file_name` before deriving the path.
        self.addCleanup(self._rmtree, folder.name, folder.file_url)
        return folder

    def _rmtree(self, name, original_url):
        import shutil

        manager = FileManager()
        urls = {original_url, frappe.db.get_value("File", name, "file_url")}
        for url in filter(None, urls):
            try:
                shutil.rmtree(manager.site_folder / storage_key(url), ignore_errors=True)
            except Exception:
                pass

    def make_file(self, parent, name=None, content=b"drive-test-bytes", owner=ALICE, size=None):
        """A file with real bytes on disk.

        Not optional: on non-flat storage `FileManager.move` throws
        "This file doesn't exist on disk" if the blob is missing, so a fixture without
        bytes would exercise a code path no real transfer ever takes.
        """
        manager = FileManager()
        doc = create_drive_file(
            name or f"{frappe.generate_hash(8)}.txt",
            parent,
            "Text",
            lambda f: "/" + str(manager.get_disk_path(f)),
            "text/plain",
            len(content) if size is None else size,
            owner=owner,
        )
        path = manager.site_folder / storage_key(doc.file_url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        self.addCleanup(lambda p=path: p.unlink(missing_ok=True))
        return doc

    # -- assertions -------------------------------------------------------
    def access(self, entity, user):
        return get_user_access_for_user(entity, user)

    def assertNoAccess(self, entity, user, msg=None):
        acc = self.access(entity, user)
        self.assertEqual(
            {k: acc[k] for k in ("read", "write", "share")},
            {"read": 0, "write": 0, "share": 0},
            msg or f"{user} should have lost access to {entity}",
        )

    def assertFullAccess(self, entity, user, msg=None):
        acc = self.access(entity, user)
        self.assertEqual(
            {k: acc[k] for k in ("read", "write", "share")},
            {"read": 1, "write": 1, "share": 1},
            msg or f"{user} should hold everything on {entity}",
        )

    def assertViewerAccess(self, entity, user):
        """What the previous owner is left with: able to open it, nothing more."""
        acc = self.access(entity, user)
        self.assertEqual(acc["read"], 1, f"{user} should still read {entity}")
        self.assertEqual(acc["write"], 0, f"{user} must not be able to change {entity}")
        self.assertEqual(acc["upload"], 0, f"{user} must not be able to add to {entity}")
        self.assertEqual(acc["share"], 0, f"{user} must not be able to re-share {entity}")

    def owner_of(self, entity):
        return frappe.db.get_value("File", entity, "owner")

    def make_content_doc(self, doctype, owner=ALICE):
        """A real content-backed file: `after_insert` creates the Drive File."""
        with self.set_user(owner):
            doc = frappe.get_doc({"doctype": doctype, "title": frappe.generate_hash(8)}).insert()
        file = frappe.db.get_value("File", {"content_doctype": doctype, "content_docname": doc.name})
        self.assertIsNotNone(file, f"{doctype} should be backed by a Drive File")
        return doc, file


class TestTransferOwnership(OwnershipTestBase):
    """The per-file / per-folder primitive."""

    def test_previous_owner_loses_access(self):
        """The whole point. An owner-flip-only implementation fails here and only here.

        Alice's file stays inside her home folder unless the transfer moves it, and
        that folder's `Drive Permission` row keeps granting her everything below it.
        """
        file = self.make_file(self.alice_home)
        self.assertFullAccess(file.name, ALICE)

        with self.set_user(ALICE):
            transfer_ownership(file.name, BOB, keep_previous_access=False)

        self.assertNoAccess(file.name, ALICE)
        self.assertFullAccess(file.name, BOB)
        self.assertEqual(self.owner_of(file.name), BOB)

    def test_file_leaves_previous_owners_home_folder(self):
        """The mechanism behind the test above, asserted directly so a regression
        names its own cause instead of showing up as a mysterious access failure."""
        file = self.make_file(self.alice_home)
        with self.set_user(ALICE):
            transfer_ownership(file.name, BOB)

        self.assertEqual(frappe.db.get_value("File", file.name, "folder"), self.bob_home)
        self.assertNotIn(self.alice_home, get_ancestors_of(file.name))

    def test_previous_owner_keeps_view_access_by_default(self):
        file = self.make_file(self.alice_home)
        with self.set_user(ALICE):
            transfer_ownership(file.name, BOB)

        self.assertViewerAccess(file.name, ALICE)
        self.assertFullAccess(file.name, BOB)

    def test_previous_owner_cannot_write_after_transfer(self):
        """The core of the view-only rule. Leaving them `write` would make the new
        owner's control nominal — the old owner could keep changing the file, and
        `rename`/`move` both gate on `write`."""
        file = self.make_file(self.alice_home)
        with self.set_user(ALICE):
            transfer_ownership(file.name, BOB)

        with self.set_user(ALICE):
            with self.assertRaises(frappe.PermissionError):
                frappe.get_doc("File", file.name).rename("alice-renamed-it.txt")

    def test_previous_owner_cannot_upload_into_a_transferred_folder(self):
        folder = self.make_folder(self.alice_home)
        with self.set_user(ALICE):
            transfer_ownership(folder.name, BOB)

        self.assertEqual(self.access(folder.name, ALICE)["upload"], 0)

    def test_previous_owner_cannot_reshare_after_transfer(self):
        """`share` is what lets someone hand a file on. Keeping it would make the
        transfer cosmetic."""
        file = self.make_file(self.alice_home)
        with self.set_user(ALICE):
            transfer_ownership(file.name, BOB)

        with self.set_user(ALICE):
            with self.assertRaises(frappe.PermissionError):
                frappe.get_doc("File", file.name).share(user=CAROL, read=1)

    def test_transfer_is_recursive(self):
        folder = self.make_folder(self.alice_home)
        inner = self.make_folder(folder.name)
        leaf = self.make_file(inner.name)

        with self.set_user(ALICE):
            result = transfer_ownership(folder.name, BOB, keep_previous_access=False)

        for name in (folder.name, inner.name, leaf.name):
            self.assertEqual(self.owner_of(name), BOB, f"{name} should have changed hands")
            self.assertFullAccess(name, BOB)
            self.assertNoAccess(name, ALICE)
        self.assertEqual(result["files"], 3)

    def test_only_the_subtree_root_is_reparented(self):
        """Children ride along on their unchanged `folder` pointer. Reparenting each
        one would flatten the tree into the recipient's home folder."""
        folder = self.make_folder(self.alice_home)
        leaf = self.make_file(folder.name)

        with self.set_user(ALICE):
            transfer_ownership(folder.name, BOB)

        self.assertEqual(frappe.db.get_value("File", folder.name, "folder"), self.bob_home)
        self.assertEqual(frappe.db.get_value("File", leaf.name, "folder"), folder.name)

    def test_trashed_descendants_transfer_too(self):
        """Otherwise they are stranded in the previous owner's trash, which lists by
        `owner` (`api/list.trash`), with no way for anyone to restore them."""
        folder = self.make_folder(self.alice_home)
        doomed = self.make_file(folder.name)
        frappe.db.set_value("File", doomed.name, "status", STATUS_TRASHED)

        with self.set_user(ALICE):
            transfer_ownership(folder.name, BOB)

        self.assertEqual(self.owner_of(doomed.name), BOB)

    def test_file_outside_home_is_reassigned_in_place(self):
        """Nothing in the shared tree is in anyone's private space, so there is no
        folder to move it out of — flipping `owner` is the whole transfer."""
        shared = self.make_folder(get_root_folder().name, owner=ALICE)
        file = self.make_file(shared.name)
        before = frappe.db.get_value("File", file.name, "folder")

        with self.set_user(ALICE):
            result = transfer_ownership(file.name, BOB)

        self.assertEqual(frappe.db.get_value("File", file.name, "folder"), before)
        self.assertFalse(result["moved"])
        self.assertEqual(self.owner_of(file.name), BOB)

    def test_new_owner_gets_an_explicit_permission_row(self):
        """Mirrors what a home folder carries, so the grant survives the file later
        being moved somewhere that would grant the new owner nothing."""
        file = self.make_file(self.alice_home)
        with self.set_user(ALICE):
            transfer_ownership(file.name, BOB)

        row = frappe.db.get_value(
            "Drive Permission", {"entity": file.name, "user": BOB}, ["read", "write", "share"], as_dict=True
        )
        self.assertIsNotNone(row, "new owner should hold a permission row of their own")
        self.assertEqual((row.read, row.write, row.share), (1, 1, 1))

    def test_name_collision_is_deduped(self):
        """The recipient may already have a file by that name; `move()` renames."""
        self.make_file(self.bob_home, name="report.txt", owner=BOB)
        mine = self.make_file(self.alice_home, name="report.txt")

        with self.set_user(ALICE):
            transfer_ownership(mine.name, BOB)

        self.assertNotEqual(frappe.db.get_value("File", mine.name, "file_name"), "report.txt")

    def test_blob_follows_the_file_on_disk(self):
        """Non-flat storage mirrors the tree, so a reparent is a real move. If the
        url and the bytes disagree the file becomes unreadable."""
        file = self.make_file(self.alice_home, content=b"canary-bytes")
        with self.set_user(ALICE):
            transfer_ownership(file.name, BOB)

        manager = FileManager()
        if manager.flat:
            self.skipTest("flat storage keys by id; a move never touches disk")
        new_url = frappe.db.get_value("File", file.name, "file_url")
        path = manager.site_folder / storage_key(new_url)
        self.assertTrue(path.exists(), f"blob should have followed to {new_url}")
        self.assertEqual(path.read_bytes(), b"canary-bytes")

    def test_flat_storage_transfer_touches_no_blob(self):
        """Flat keys by id, so the same transfer must succeed without a disk move."""
        settings = frappe.get_single("Drive Disk Settings")
        original = settings.flat
        settings.db_set("flat", 1, update_modified=False)
        self.addCleanup(lambda: settings.db_set("flat", original, update_modified=False))

        file = self.make_file(self.alice_home)
        url_before = frappe.db.get_value("File", file.name, "file_url")

        with self.set_user(ALICE), patch.object(FileManager, "move") as disk_move:
            transfer_ownership(file.name, BOB)

        disk_move.assert_not_called()
        self.assertEqual(frappe.db.get_value("File", file.name, "file_url"), url_before)
        self.assertEqual(self.owner_of(file.name), BOB)

    def test_s3_transfer_copies_the_object(self):
        """S3 has no directories, so a reparent copies every blob in the subtree
        across. Mocked: there is no MinIO in CI."""
        file = self.make_file(self.alice_home)

        with (
            self.set_user(ALICE),
            patch.object(FileManager, "move", return_value="/x") as disk_move,
            patch.object(FileManager, "s3_enabled", True, create=True),
        ):
            transfer_ownership(file.name, BOB)

        self.assertTrue(disk_move.called, "S3 transfer must relocate the object")
        self.assertEqual(self.owner_of(file.name), BOB)


class TestTransferGuards(OwnershipTestBase):
    """Everything the primitive must refuse."""

    def test_non_owner_cannot_transfer(self):
        """Even with `share`. Ownership is strictly more than share — it bypasses
        denies and is what lets you give the file away again."""
        file = self.make_file(self.alice_home)
        with self.set_user(ALICE):
            frappe.get_doc("File", file.name).share(user=CAROL, read=1, write=1, share=1)

        with self.set_user(CAROL):
            with self.assertRaises(frappe.PermissionError):
                transfer_ownership(file.name, CAROL)

        self.assertEqual(self.owner_of(file.name), ALICE)

    def test_drive_admin_can_transfer_someone_elses_file(self):
        file = self.make_file(self.alice_home)
        with self.set_user(ADMIN):
            transfer_ownership(file.name, BOB, keep_previous_access=False)

        self.assertEqual(self.owner_of(file.name), BOB)
        self.assertNoAccess(file.name, ALICE)

    def test_cannot_transfer_to_unknown_user(self):
        file = self.make_file(self.alice_home)
        with self.set_user(ALICE):
            with self.assertRaises(frappe.DoesNotExistError):
                transfer_ownership(file.name, "nobody-at-all@example.com")

    def test_cannot_transfer_to_disabled_user(self):
        file = self.make_file(self.alice_home)
        frappe.db.set_value("User", CAROL, "enabled", 0)
        self.addCleanup(lambda: frappe.db.set_value("User", CAROL, "enabled", 1))

        with self.set_user(ALICE):
            with self.assertRaises(frappe.ValidationError):
                transfer_ownership(file.name, CAROL)

    def test_cannot_transfer_to_a_non_user_principal(self):
        """`Drive Permission.user` accepts these; ownership must not."""
        file = self.make_file(self.alice_home)
        for principal in (GENERAL_USER, f"{GROUP_PREFIX}Everyone", "", "Guest"):
            with self.subTest(principal=principal), self.set_user(ALICE):
                with self.assertRaises((frappe.ValidationError, frappe.DoesNotExistError)):
                    transfer_ownership(file.name, principal)
        self.assertEqual(self.owner_of(file.name), ALICE)

    def test_cannot_transfer_to_current_owner(self):
        file = self.make_file(self.alice_home)
        with self.set_user(ALICE):
            with self.assertRaises(frappe.ValidationError):
                transfer_ownership(file.name, ALICE)

    def test_cannot_transfer_a_home_folder(self):
        """It is bound to its user by `Drive Settings.user_folder`; handing it over
        would leave two users pointing at one folder."""
        with self.set_user(ALICE):
            with self.assertRaises(frappe.PermissionError):
                transfer_ownership(self.alice_home, BOB)

    def test_cannot_transfer_structural_roots(self):
        for root in (ROOT_FOLDER, USERS_FOLDER):
            with self.subTest(root=root), self.set_user(ADMIN):
                with self.assertRaises(frappe.PermissionError):
                    transfer_ownership(root, BOB)

    def test_cannot_transfer_a_trashed_file(self):
        file = self.make_file(self.alice_home)
        frappe.db.set_value("File", file.name, "status", STATUS_TRASHED)
        with self.set_user(ALICE):
            with self.assertRaises(frappe.ValidationError):
                transfer_ownership(file.name, BOB)

    def test_recipient_quota_is_enforced(self):
        """Storage is billed to `File.owner`, so a transfer moves the cost. Checked
        against the whole subtree up front, not per file."""
        big = self.make_file(self.alice_home, size=5 * 1024**2)
        frappe.db.set_value("Drive Settings", BOB, "quota", 1)  # 1 MB
        self.addCleanup(lambda: frappe.db.set_value("Drive Settings", BOB, "quota", 0))

        with self.set_user(ALICE):
            with self.assertRaises(ValueError):
                transfer_ownership(big.name, BOB)

        self.assertEqual(self.owner_of(big.name), ALICE, "a refused transfer must change nothing")

    def test_preview_requires_ownership(self):
        file = self.make_file(self.alice_home)
        with self.set_user(CAROL):
            with self.assertRaises(frappe.PermissionError):
                get_transfer_preview(file.name)


class TestTransferSideEffects(OwnershipTestBase):
    """Activity log, notifications, and the email that must not be sent."""

    def test_transfer_is_recorded_in_the_activity_log(self):
        frappe.flags.mute_drive_activity_log = False
        file = self.make_file(self.alice_home)
        with self.set_user(ALICE):
            transfer_ownership(file.name, BOB)

        log = frappe.get_all(
            "Drive Entity Activity Log",
            filters={"entity": file.name, "action_type": "transfer"},
            fields=["old_value", "new_value"],
        )
        self.assertEqual(len(log), 1)
        self.assertEqual((log[0].old_value, log[0].new_value), (ALICE, BOB))

    def test_new_owner_is_notified(self):
        file = self.make_file(self.alice_home)
        with self.set_user(ALICE):
            transfer_ownership(file.name, BOB)

        self.assertTrue(
            frappe.db.exists(
                "Drive Notification",
                {"to_user": BOB, "type": "Transfer", "notif_doctype_name": file.name},
            )
        )

    def test_previous_owner_is_not_emailed_a_share_notification(self):
        """`DrivePermission.after_insert` would otherwise mail them "someone shared a
        file with you" about a file they just gave away."""
        file = self.make_file(self.alice_home)
        with (
            self.set_user(ALICE),
            patch("suite.drive.doctype.drive_permission.drive_permission.frappe.enqueue") as enqueue,
        ):
            transfer_ownership(file.name, BOB)

        enqueue.assert_not_called()

    def test_flag_is_cleared_after_a_failed_transfer(self):
        """The suppression flag is process-global; leaking it would silence share
        emails for everything that followed in the same request."""
        file = self.make_file(self.alice_home)
        with self.set_user(ALICE), patch.object(FileManager, "move", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                transfer_ownership(file.name, BOB)

        self.assertFalse(frappe.flags.get("in_ownership_transfer"))

    def test_session_user_is_restored_after_a_failed_transfer(self):
        """`_elevated` swaps `session.user` to Administrator for the move. If an
        exception left it swapped, the rest of the request would run as root."""
        file = self.make_file(self.alice_home)
        with self.set_user(ALICE), patch.object(FileManager, "move", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                transfer_ownership(file.name, BOB)
            self.assertEqual(frappe.session.user, ALICE)


class TestTransferBlastRadius(OwnershipTestBase):
    """Everything outside Drive that reads `owner` and therefore moves with it."""

    def test_storage_usage_follows_the_new_owner(self):
        size = 3 * 1024
        file = self.make_file(self.alice_home, size=size)
        alice_before = get_storage_usage(ALICE)["total_size"]
        bob_before = get_storage_usage(BOB)["total_size"]

        with self.set_user(ALICE):
            transfer_ownership(file.name, BOB)

        self.assertEqual(get_storage_usage(ALICE)["total_size"], alice_before - size)
        self.assertEqual(get_storage_usage(BOB)["total_size"], bob_before + size)

    def test_trash_listing_follows_the_new_owner(self):
        """`api/list.trash` filters on `owner`, so a transferred file that is later
        trashed must appear for the new owner and not the old one."""
        file = self.make_file(self.alice_home)
        with self.set_user(ALICE):
            transfer_ownership(file.name, BOB)
        frappe.db.set_value("File", file.name, "status", STATUS_TRASHED)

        with self.set_user(BOB):
            self.assertIn(file.name, [r["name"] for r in trash()])
        with self.set_user(ALICE):
            self.assertNotIn(file.name, [r["name"] for r in trash()])

    def make_sheet(self, owner=ALICE):
        return self.make_content_doc("Sheet", owner)

    def test_content_document_owner_follows_the_file(self):
        """`content_has_permission` delegates *access* to the backing File, so a
        transfer is already correct there. Listings are not: Sheets filters, sorts and
        trashes on `tabSheet.owner`, Slides lists on `Presentation.owner`. Leave the
        content document behind and it keeps showing up for its previous owner.

        Both content apps go through the same generic branch of `_reassign`, so both
        are checked rather than trusting one to stand in for the other.
        """
        for doctype in ("Sheet", "Presentation"):
            with self.subTest(doctype=doctype):
                doc, file = self.make_content_doc(doctype)
                self.assertEqual(frappe.db.get_value(doctype, doc.name, "owner"), ALICE)

                with self.set_user(ALICE):
                    transfer_ownership(file, BOB)

                self.assertEqual(frappe.db.get_value(doctype, doc.name, "owner"), BOB)

    def test_attachment_content_links_are_left_alone(self):
        """`ATTACHMENT_CONTENT_DOCTYPE` rows point back at another *File* (the library
        original), not at a document this transfer owns. Rewriting that File's owner
        would hand over somebody else's upload as a side effect."""
        from suite.drive.utils import ATTACHMENT_CONTENT_DOCTYPE

        original = self.make_file(self.carol_home, owner=CAROL)
        copy = self.make_file(self.alice_home)
        copy.db_set({"content_doctype": ATTACHMENT_CONTENT_DOCTYPE, "content_docname": original.name})

        with self.set_user(ALICE):
            transfer_ownership(copy.name, BOB)

        self.assertEqual(self.owner_of(original.name), CAROL, "the library original must not move")
        self.assertEqual(self.owner_of(copy.name), BOB)

    def test_sheets_own_list_api_follows_the_transfer(self):
        """End to end through Sheets' own endpoint, not just the column."""
        from suite.sheets.api import list_sheets

        sheet, file = self.make_sheet()

        with self.set_user(ALICE):
            self.assertIn(sheet.name, [r["name"] for r in list_sheets(owner_filter="mine")["sheets"]])

        with self.set_user(ALICE):
            transfer_ownership(file, BOB)

        with self.set_user(BOB):
            self.assertIn(sheet.name, [r["name"] for r in list_sheets(owner_filter="mine")["sheets"]])
        with self.set_user(ALICE):
            self.assertNotIn(sheet.name, [r["name"] for r in list_sheets(owner_filter="mine")["sheets"]])

    def test_transferred_content_file_is_still_openable_by_its_new_owner(self):
        """The File is the permission anchor for content apps — if the transfer broke
        the delegation the new owner would own a document they cannot open."""
        from suite.drive.overrides.file import content_has_permission

        sheet, file = self.make_sheet()
        with self.set_user(ALICE):
            transfer_ownership(file, BOB)

        doc = frappe.get_doc("Sheet", sheet.name)
        self.assertTrue(content_has_permission(doc, "read", BOB))
        self.assertTrue(content_has_permission(doc, "write", BOB))

    def test_shared_with_me_does_not_list_your_own_files(self):
        """`_apply_shared_filter` excludes `DriveFile.owner == user`. The new owner
        holds an explicit permission row, so without that exclusion a transferred file
        would show up as something shared *with* them."""
        from suite.drive.api.list import shared

        file = self.make_file(self.alice_home)
        with self.set_user(ALICE):
            transfer_ownership(file.name, BOB)

        with self.set_user(BOB):
            self.assertNotIn(file.name, [r["name"] for r in shared(shared_type="with")])

    def test_previous_owner_sees_it_under_shared_with_me(self):
        """The flip side: the view-only grant should surface it there, since it is now
        genuinely somebody else's file that they can still open."""
        from suite.drive.api.list import shared

        file = self.make_file(self.alice_home)
        with self.set_user(ALICE):
            transfer_ownership(file.name, BOB)

        with self.set_user(ALICE):
            self.assertIn(file.name, [r["name"] for r in shared(shared_type="with")])


class TestBulkTransfer(OwnershipTestBase):
    """The offboarding path."""

    def setUp(self):
        super().setUp()
        # `run_bulk_transfer` commits after every item, on purpose: that is what makes
        # a crashed job resumable rather than all-or-nothing. Inside a test it would
        # escape the framework's per-test rollback and leak transfer rows and half-moved
        # files into every subsequent run. Both are neutered here and asserted on
        # directly instead — resumability is still exercised for real by
        # `test_is_resumable_and_idempotent`, which runs the job twice.
        self.commit = self._silence("commit")
        self.rollback = self._silence("rollback")

    def _silence(self, method):
        patcher = patch(f"suite.drive.api.ownership.frappe.db.{method}")
        mock = patcher.start()
        self.addCleanup(patcher.stop)
        return mock

    def run_bulk(self, from_user=ALICE, to_user=BOB, **kwargs):
        with self.set_user(ADMIN):
            name = transfer_all_owned(from_user, to_user, **kwargs)
        run_bulk_transfer(name)
        return frappe.get_doc("Drive Ownership Transfer", name)

    def test_everything_lands_in_one_folder_in_the_recipients_drive(self):
        folder = self.make_folder(self.alice_home)
        loose = self.make_file(self.alice_home)
        nested = self.make_file(folder.name)

        transfer = self.run_bulk()

        self.assertEqual(transfer.status, "Completed")
        destination = transfer.destination_folder
        self.assertEqual(frappe.db.get_value("File", destination, "folder"), self.bob_home)
        self.assertEqual(frappe.db.get_value("File", folder.name, "folder"), destination)
        self.assertEqual(frappe.db.get_value("File", loose.name, "folder"), destination)
        # nested rides along with its parent rather than being flattened
        self.assertEqual(frappe.db.get_value("File", nested.name, "folder"), folder.name)
        for name in (folder.name, loose.name, nested.name):
            self.assertEqual(self.owner_of(name), BOB)

    def test_destination_folder_is_named_after_the_previous_owner(self):
        self.make_file(self.alice_home)
        transfer = self.run_bulk()
        label = frappe.db.get_value("User", ALICE, "full_name")
        self.assertIn(label, frappe.db.get_value("File", transfer.destination_folder, "file_name"))

    def test_previous_owner_keeps_view_access_on_the_container_only(self):
        """One grant on the container covers everything inside it — nearest row wins.
        Granting per item would mean one row and one email per file."""
        folder = self.make_folder(self.alice_home)
        leaf = self.make_file(folder.name)

        transfer = self.run_bulk()

        self.assertViewerAccess(transfer.destination_folder, ALICE)
        self.assertViewerAccess(leaf.name, ALICE)
        self.assertEqual(
            frappe.db.count("Drive Permission", {"user": ALICE, "entity": leaf.name}),
            0,
            "no per-item rows for the previous owner",
        )

    def test_opting_out_leaves_the_previous_owner_nothing(self):
        file = self.make_file(self.alice_home)
        self.run_bulk(keep_previous_access=False)
        self.assertNoAccess(file.name, ALICE)

    def test_is_resumable_and_idempotent(self):
        """The work list is re-derived, never stored, so a crashed job is just re-run."""
        file = self.make_file(self.alice_home)
        transfer = self.run_bulk()
        destination = transfer.destination_folder

        run_bulk_transfer(transfer.name)
        transfer.reload()

        self.assertEqual(transfer.destination_folder, destination, "no second folder")
        self.assertEqual(frappe.db.get_value("File", file.name, "folder"), destination)
        self.assertEqual(transfer.files_failed, 0)

    def test_survives_the_source_user_being_already_deleted(self):
        """The case this feature exists for: someone has already been offboarded and
        their files are stranded behind admin-only access."""
        ghost = "drive-xfer-ghost@example.com"
        ensure_user(ghost)
        ghost_home = get_user_folder(ghost).name
        orphan = self.make_file(ghost_home, owner=ghost)
        frappe.delete_doc("User", ghost, ignore_permissions=True, force=True)
        self.assertFalse(frappe.db.exists("User", ghost))
        # the Drive rows outlive them — every Drive doctype is in `ignore_links_on_delete`
        self.assertTrue(frappe.db.exists("Drive Settings", ghost))

        transfer = self.run_bulk(from_user=ghost)

        self.assertEqual(transfer.status, "Completed")
        self.assertEqual(self.owner_of(orphan.name), BOB)
        self.assertFullAccess(orphan.name, BOB)

    def test_files_outside_the_home_folder_are_reassigned_in_place(self):
        shared_folder = self.make_folder(get_root_folder().name, owner=ALICE)
        parent_before = frappe.db.get_value("File", shared_folder.name, "folder")

        transfer = self.run_bulk()

        self.assertEqual(self.owner_of(shared_folder.name), BOB)
        self.assertEqual(
            frappe.db.get_value("File", shared_folder.name, "folder"),
            parent_before,
            "shared-tree content must not be dragged into a private folder",
        )
        self.assertNotEqual(shared_folder.name, transfer.destination_folder)

    def test_only_a_drive_admin_may_run_it(self):
        with self.set_user(CAROL):
            with self.assertRaises(frappe.PermissionError):
                transfer_all_owned(ALICE, CAROL)

    def test_refuses_a_second_concurrent_run(self):
        self.make_file(self.alice_home)
        with self.set_user(ADMIN):
            transfer_all_owned(ALICE, BOB)
            with self.assertRaises(frappe.ValidationError):
                transfer_all_owned(ALICE, CAROL)

    def test_refuses_a_handover_to_the_same_user(self):
        with self.set_user(ADMIN):
            with self.assertRaises(frappe.ValidationError):
                transfer_all_owned(ALICE, ALICE)

    def test_one_failure_does_not_abort_the_run(self):
        """A departing employee's Drive is thousands of files; a single bad blob must
        not strand the rest."""
        good = self.make_file(self.alice_home)
        bad = self.make_file(self.alice_home)
        # the blob is gone, so FileManager.move throws for this one only
        manager = FileManager()
        (manager.site_folder / storage_key(bad.file_url)).unlink(missing_ok=True)

        transfer = self.run_bulk()

        self.assertEqual(transfer.status, "Completed With Errors")
        self.assertEqual(transfer.files_failed, 1)
        self.assertEqual(self.owner_of(good.name), BOB, "the healthy file still transferred")
        self.assertEqual(self.owner_of(bad.name), ALICE, "the failed one changed nothing")
        self.assertIn(bad.name, transfer.error_log)
        self.assertTrue(self.rollback.called, "a failed item must not leave a partial move behind")

    def test_a_failed_item_reappears_on_the_next_run(self):
        """The failure path and the resume path have to agree, or a transient error
        would silently drop a file out of the handover for good."""
        bad = self.make_file(self.alice_home)
        manager = FileManager()
        blob = manager.site_folder / storage_key(bad.file_url)
        blob.unlink(missing_ok=True)

        transfer = self.run_bulk()
        self.assertEqual(transfer.files_failed, 1)
        self.assertIn(bad.name, _pending_items(ALICE, self.alice_home))

        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(b"restored")
        run_bulk_transfer(transfer.name)

        self.assertEqual(self.owner_of(bad.name), BOB)

    def test_never_touches_another_users_home_folder(self):
        """`_pending_items` must refuse structural nodes even when they are somehow
        owned by the source user."""
        frappe.db.set_value("File", self.carol_home, "owner", ALICE, update_modified=False)
        self.addCleanup(
            lambda: frappe.db.set_value("File", self.carol_home, "owner", CAROL, update_modified=False)
        )

        self.assertNotIn(self.carol_home, _pending_items(ALICE, self.alice_home))

    def test_empty_drive_completes_without_creating_a_folder(self):
        transfer = self.run_bulk(from_user=CAROL, to_user=BOB)
        self.assertEqual(transfer.status, "Completed")
        self.assertIsNone(transfer.destination_folder)


class TestSubtreeHelper(OwnershipTestBase):
    def test_subtree_includes_root_and_every_descendant(self):
        folder = self.make_folder(self.alice_home)
        inner = self.make_folder(folder.name)
        leaf = self.make_file(inner.name)

        names = {r.name for r in _subtree(folder.name)}
        self.assertEqual(names, {folder.name, inner.name, leaf.name})

    def test_subtree_of_a_leaf_is_just_itself(self):
        file = self.make_file(self.alice_home)
        self.assertEqual([r.name for r in _subtree(file.name)], [file.name])

    def test_preview_reports_what_would_move(self):
        folder = self.make_folder(self.alice_home)
        self.make_file(folder.name, size=100)
        self.make_file(folder.name, size=50)

        with self.set_user(ALICE):
            preview = get_transfer_preview(folder.name)

        self.assertEqual(preview["files"], 2)
        self.assertEqual(preview["folders"], 1)
        self.assertEqual(preview["bytes"], 150)
        self.assertTrue(preview["leaves_your_drive"])


class TestFileKinds(OwnershipTestBase):
    """Every kind of thing a Drive File can be.

    They differ in whether they have a blob at all, which decides what `move()`
    touches. A transfer that only ever ran against plain uploads would break on the
    first shortcut or presentation someone hands over.
    """

    def make_link(self, parent, url="https://frappe.io"):
        """A bookmark: no bytes anywhere, `_not_in_disk` is True."""
        return create_drive_file(
            f"{frappe.generate_hash(6)}-link", parent, "Link", url, "link", 0, owner=ALICE
        )

    def test_link_files_transfer_without_touching_storage(self):
        link = self.make_link(self.alice_home)
        with self.set_user(ALICE), patch.object(FileManager, "move") as disk_move:
            transfer_ownership(link.name, BOB)

        disk_move.assert_not_called()
        self.assertEqual(self.owner_of(link.name), BOB)
        self.assertFullAccess(link.name, BOB)

    def test_empty_folder_transfers(self):
        folder = self.make_folder(self.alice_home)
        with self.set_user(ALICE):
            result = transfer_ownership(folder.name, BOB)

        self.assertEqual(result["files"], 1)
        self.assertEqual(self.owner_of(folder.name), BOB)

    def test_zero_byte_file_transfers(self):
        empty = self.make_file(self.alice_home, content=b"")
        with self.set_user(ALICE):
            transfer_ownership(empty.name, BOB)
        self.assertEqual(self.owner_of(empty.name), BOB)

    def test_content_backed_file_without_a_blob_transfers(self):
        """Slides and Sheets keep their content in their own doctype, so the File has
        a url but nothing behind it. `_blob_expected` must not demand one."""
        doc, file = self.make_content_doc("Presentation")
        with self.set_user(ALICE):
            transfer_ownership(file, BOB)

        self.assertEqual(self.owner_of(file), BOB)
        self.assertEqual(frappe.db.get_value("Presentation", doc.name, "owner"), BOB)

    def test_mixed_subtree_of_every_kind(self):
        """The realistic case: one folder holding a bit of everything."""
        folder = self.make_folder(self.alice_home)
        upload = self.make_file(folder.name)
        link = self.make_link(folder.name)
        empty_sub = self.make_folder(folder.name)
        trashed = self.make_file(folder.name)
        frappe.db.set_value("File", trashed.name, "status", STATUS_TRASHED)

        with self.set_user(ALICE):
            transfer_ownership(folder.name, BOB, keep_previous_access=False)

        for name in (folder.name, upload.name, link.name, empty_sub.name, trashed.name):
            self.assertEqual(self.owner_of(name), BOB, f"{name} did not change hands")
        self.assertNoAccess(upload.name, ALICE)

    def test_unicode_and_separator_in_names_survive(self):
        """`escape_component` turns `/` into `%2F` so a name cannot forge a path
        level. The relocation has to round-trip that, or the blob is lost."""
        folder = self.make_folder(self.alice_home, name=f"Проект ✨ {frappe.generate_hash(6)}")
        odd = self.make_file(folder.name, name="a/b — copy.txt", content=b"unicode-bytes")

        with self.set_user(ALICE):
            transfer_ownership(folder.name, BOB)

        self.assertEqual(self.owner_of(odd.name), BOB)
        manager = FileManager()
        if not manager.flat:
            url = frappe.db.get_value("File", odd.name, "file_url")
            self.assertTrue(
                (manager.site_folder / storage_key(url)).exists(),
                f"blob lost for an escaped name: {url}",
            )

    def test_deeply_nested_tree_transfers_whole(self):
        parent = self.alice_home
        chain = []
        for _ in range(8):
            parent = self.make_folder(parent).name
            chain.append(parent)
        leaf = self.make_file(parent)

        with self.set_user(ALICE):
            result = transfer_ownership(chain[0], BOB, keep_previous_access=False)

        self.assertEqual(result["files"], 9)
        for name in [*chain, leaf.name]:
            self.assertEqual(self.owner_of(name), BOB)
        self.assertNoAccess(leaf.name, ALICE)

    def test_subtree_already_partly_owned_by_the_recipient(self):
        """A collaborator's file sitting inside the folder being handed over. It is
        already theirs; the transfer must be a no-op for it, not a downgrade."""
        folder = self.make_folder(self.alice_home)
        with self.set_user(ALICE):
            frappe.get_doc("File", folder.name).share(user=BOB, read=1, write=1, upload=1)
        theirs = self.make_file(folder.name, owner=BOB)

        with self.set_user(ALICE):
            transfer_ownership(folder.name, BOB)

        self.assertEqual(self.owner_of(theirs.name), BOB)
        self.assertFullAccess(theirs.name, BOB)

    def test_attachment_keeps_following_its_reference_document(self):
        """Attachments resolve access through `_ref_doc_access`, not the Drive tree.
        Ownership still moves, but the reference document stays in charge."""
        file = self.make_file(self.alice_home)
        file.db_set({"attached_to_doctype": "User", "attached_to_name": CAROL})

        with self.set_user(ALICE):
            transfer_ownership(file.name, BOB)

        self.assertEqual(self.owner_of(file.name), BOB)
        self.assertEqual(
            frappe.db.get_value("File", file.name, "attached_to_name"),
            CAROL,
            "the reference link must not be rewritten by a transfer",
        )


class TestFailureHandling(OwnershipTestBase):
    """What happens when storage does not cooperate.

    On non-flat storage the backend key encodes the path, so relocating a subtree
    rewrites every key under it — on S3 that is a copy + delete per descendant, with
    no transaction spanning the two systems. These cover the ways that goes wrong.
    """

    def test_missing_blob_is_refused_before_anything_moves(self):
        folder = self.make_folder(self.alice_home)
        good = self.make_file(folder.name)
        bad = self.make_file(folder.name)
        manager = FileManager()
        if manager.flat:
            self.skipTest("flat storage never relocates, so there is nothing to verify")
        (manager.site_folder / storage_key(bad.file_url)).unlink()

        with self.set_user(ALICE):
            with self.assertRaises(frappe.ValidationError) as caught:
                transfer_ownership(folder.name, BOB)

        self.assertIn(bad.name, str(caught.exception))
        # nothing at all moved
        self.assertEqual(self.owner_of(folder.name), ALICE)
        self.assertEqual(self.owner_of(good.name), ALICE)
        self.assertEqual(frappe.db.get_value("File", folder.name, "folder"), self.alice_home)

    def test_verification_ignores_kinds_that_have_no_blob(self):
        """Links, trashed rows and content-backed files legitimately have nothing at
        their url; demanding one would refuse perfectly good transfers."""
        from suite.drive.api.ownership import _blob_expected

        cases = {
            "link": {"status": STATUS_ACTIVE, "file_url": "https://x", "file_type": "Link"},
            "trashed": {"status": STATUS_TRASHED, "file_url": "/a", "file_type": "Text"},
            "no url": {"status": STATUS_ACTIVE, "file_url": "", "file_type": "Text"},
            "slides": {
                "status": STATUS_ACTIVE,
                "file_url": "/a",
                "file_type": "Presentation",
                "content_doctype": "Presentation",
            },
        }
        for label, row in cases.items():
            with self.subTest(kind=label):
                self.assertFalse(_blob_expected(frappe._dict(row)))

        self.assertTrue(
            _blob_expected(frappe._dict({"status": STATUS_ACTIVE, "file_url": "/a", "file_type": "Text"}))
        )

    def test_half_moved_subtree_is_repaired_not_left_dangling(self):
        """The failure that motivated `_capture_urls`/`_repair_urls`.

        A relocation that dies partway has already moved some blobs. Rolling the DB
        back points those rows at keys that are now empty — the bytes exist but the
        files are unreachable. The repair re-points them.
        """
        from suite.drive.api.ownership import _capture_urls, _repair_urls

        manager = FileManager()
        if manager.flat:
            self.skipTest("flat storage never relocates")

        file = self.make_file(self.alice_home, content=b"survivor")
        original_url = frappe.db.get_value("File", file.name, "file_url")

        # Simulate exactly what a half-finished move leaves behind: bytes at a new
        # key, and (after rollback) a row still naming the old one.
        moved_url = original_url + ".relocated"
        moved_path = manager.site_folder / storage_key(moved_url)
        moved_path.write_bytes(b"survivor")
        self.addCleanup(lambda: moved_path.unlink(missing_ok=True))
        (manager.site_folder / storage_key(original_url)).unlink()

        attempted = {file.name: moved_url}
        repaired = _repair_urls(attempted)

        self.assertEqual(repaired, [file.name])
        self.assertEqual(frappe.db.get_value("File", file.name, "file_url"), moved_url)

    def test_repair_leaves_healthy_rows_alone(self):
        """It must only ever rescue rows whose recorded blob is genuinely gone."""
        from suite.drive.api.ownership import _repair_urls

        if FileManager().flat:
            self.skipTest("flat storage never relocates")
        file = self.make_file(self.alice_home)
        url = frappe.db.get_value("File", file.name, "file_url")

        self.assertEqual(_repair_urls({file.name: url + ".elsewhere"}), [])
        self.assertEqual(frappe.db.get_value("File", file.name, "file_url"), url)

    def test_capture_urls_reads_the_uncommitted_move(self):
        """It has to run before the rollback — that is the only moment the new
        locations are knowable."""
        from suite.drive.api.ownership import _capture_urls

        file = self.make_file(self.alice_home)
        frappe.db.set_value("File", file.name, "file_url", "/private/files/moved-here")

        self.assertEqual(_capture_urls([file.name]), {file.name: "/private/files/moved-here"})

    def test_a_failed_transfer_leaves_the_file_wholly_untouched(self):
        file = self.make_file(self.alice_home)
        with self.set_user(ALICE), patch.object(FileManager, "move", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                transfer_ownership(file.name, BOB)

        self.assertEqual(self.owner_of(file.name), ALICE)
        self.assertEqual(frappe.db.get_value("File", file.name, "folder"), self.alice_home)
        self.assertFullAccess(file.name, ALICE)
        self.assertNoAccess(file.name, BOB)


class TestS3Semantics(OwnershipTestBase):
    """S3 keys encode the path, so a transfer is real object movement, not metadata.

    Everything here drives a fake client: there is no MinIO in CI (T5), and the point
    is the call *pattern* rather than any particular backend's behaviour.
    """

    class FakeS3:
        def __init__(self):
            self.objects = {}
            self.calls = []
            self.fail_after = None

        def put_object(self, Bucket, Key, Body=""):
            self.calls.append(("put", Key))
            self.objects[Key] = Body

        def head_object(self, Bucket, Key):
            self.calls.append(("head", Key))
            if Key not in self.objects:
                raise RuntimeError("404")

        def copy_object(self, Bucket, CopySource, Key):
            self.calls.append(("copy", CopySource["Key"], Key))
            if self.fail_after is not None and self.copies > self.fail_after:
                raise RuntimeError("S3 throttled")
            self.objects[Key] = self.objects.get(CopySource["Key"], b"")

        def delete_object(self, Bucket, Key):
            self.calls.append(("delete", Key))
            self.objects.pop(Key, None)

        @property
        def copies(self):
            return len([c for c in self.calls if c[0] == "copy"])

    def setUp(self):
        super().setUp()
        self.fake = self.FakeS3()
        real_init = FileManager.__init__

        def fake_init(manager):
            # Not the real __init__: it builds a boto3 client and needs a
            # decryptable aws_secret that no test site has.
            from pathlib import Path

            manager.settings = frappe.get_single("Drive Disk Settings")
            manager.s3_enabled = 1
            manager.flat = 0
            manager.bucket = "test-bucket"
            manager.site_folder = Path(frappe.get_site_path())
            manager.conn = self.fake

        patcher = patch.object(FileManager, "__init__", fake_init)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(lambda: setattr(FileManager, "__init__", real_init))

    def s3_file(self, parent, name=None):
        from suite.drive.utils.files import get_s3_url

        manager = FileManager()
        doc = create_drive_file(
            name or f"{frappe.generate_hash(8)}.txt",
            parent,
            "Text",
            lambda f: get_s3_url(str(manager.get_disk_path(f))),
            "text/plain",
            10,
            owner=ALICE,
        )
        self.fake.objects[str(manager.get_disk_path(doc))] = b"bytes"
        return doc

    def test_the_key_encodes_the_path_so_every_blob_is_copied(self):
        folder = self.make_folder(self.alice_home)
        a = self.s3_file(folder.name)
        b = self.s3_file(folder.name)

        self.fake.calls.clear()
        with self.set_user(ALICE):
            transfer_ownership(folder.name, BOB)

        copies = [c for c in self.fake.calls if c[0] == "copy"]
        deletes = [c for c in self.fake.calls if c[0] == "delete"]
        # the folder marker plus both files — S3 has no directories to rename
        self.assertEqual(len(copies), 3, f"expected 3 copies, got {self.fake.calls}")
        self.assertEqual(len(deletes), 3)
        for name in (a.name, b.name):
            url = frappe.db.get_value("File", name, "file_url")
            self.assertIn(storage_key(url), self.fake.objects, f"{name} points at no object")

    def test_every_transferred_object_lands_under_the_recipient(self):
        folder = self.make_folder(self.alice_home)
        self.s3_file(folder.name)

        with self.set_user(ALICE):
            transfer_ownership(folder.name, BOB)

        for key in self.fake.objects:
            self.assertNotIn(ALICE, key, f"{key} still sits under the previous owner")

    def test_a_throttled_copy_does_not_strand_the_blobs_it_already_moved(self):
        """The regression this whole failure-handling section exists for."""
        from suite.drive.api.ownership import _capture_urls, _repair_urls

        folder = self.make_folder(self.alice_home)
        files = [self.s3_file(folder.name) for _ in range(3)]
        names = [folder.name] + [f.name for f in files]

        self.fake.fail_after = 2  # die partway through the subtree
        with self.set_user(ALICE):
            with self.assertRaises(Exception):
                transfer_ownership(folder.name, BOB)
            attempted = _capture_urls(names)

        repaired = _repair_urls(attempted)

        manager = FileManager()
        for name in [f.name for f in files]:
            url = frappe.db.get_value("File", name, "file_url")
            self.assertTrue(
                _blob_exists_for_test(manager, url),
                f"{name} was left pointing at nothing (repaired: {repaired})",
            )

    def test_flat_mode_really_is_metadata_only(self):
        """The one configuration where the hypothesis 'S3 stores blobs, so a transfer
        is only a database change' actually holds."""
        settings = frappe.get_single("Drive Disk Settings")
        original = settings.flat
        settings.db_set("flat", 1, update_modified=False)
        self.addCleanup(lambda: settings.db_set("flat", original, update_modified=False))

        real_init = FileManager.__init__

        def flat_init(manager):
            real_init(manager)
            manager.flat = 1

        with patch.object(FileManager, "__init__", flat_init):
            folder = self.make_folder(self.alice_home)
            self.s3_file(folder.name)
            self.fake.calls.clear()
            with self.set_user(ALICE):
                transfer_ownership(folder.name, BOB)

        self.assertEqual(
            [c for c in self.fake.calls if c[0] in ("copy", "delete")],
            [],
            "flat storage must not move a single object",
        )
        self.assertEqual(self.owner_of(folder.name), BOB)


def _blob_exists_for_test(manager, url):
    from suite.drive.api.ownership import _blob_exists

    return _blob_exists(manager, url)


class TestSharedWithListDedup(OwnershipTestBase):
    """`get_shared_with_list` inserts a synthetic owner row unconditionally.

    `_do_transfer` also plants a real `Drive Permission` row for the new owner
    (`grant_owner_access`), so the owner is now representable two ways. This is the
    display layer that has to collapse them back into one row.
    """

    def test_new_owner_appears_once_after_a_transfer(self):
        from suite.drive.api.permissions import get_shared_with_list

        file = self.make_file(self.alice_home)
        with self.set_user(ALICE):
            transfer_ownership(file.name, BOB)

        with self.set_user(BOB):
            rows = get_shared_with_list(file.name)

        matches = [r for r in rows if r.get("user") == BOB]
        self.assertEqual(len(matches), 1, f"BOB listed {len(matches)} times: {rows}")

    def test_a_users_own_home_folder_lists_them_once(self):
        """The pre-existing case the fix also has to cover: `get_user_folder` plants
        the same kind of explicit owner row at creation time, independent of any
        transfer."""
        from suite.drive.api.permissions import get_shared_with_list

        with self.set_user(ALICE):
            rows = get_shared_with_list(self.alice_home)

        matches = [r for r in rows if r.get("user") == ALICE]
        self.assertEqual(len(matches), 1, f"ALICE listed {len(matches)} times: {rows}")

    def test_other_collaborators_are_unaffected(self):
        from suite.drive.api.permissions import get_shared_with_list

        file = self.make_file(self.alice_home)
        with self.set_user(ALICE):
            frappe.get_doc("File", file.name).share(user=CAROL, read=1, write=1)
            transfer_ownership(file.name, BOB)

        with self.set_user(BOB):
            rows = get_shared_with_list(file.name)

        self.assertEqual(len([r for r in rows if r.get("user") == BOB]), 1)
        self.assertEqual(len([r for r in rows if r.get("user") == CAROL]), 1)
