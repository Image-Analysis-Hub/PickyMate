"""A thumbnail grid for the boxed-particle list.

Unlike a plain auto-flowing QListWidget, this grid keeps each particle's
thumbnail pinned to a stable "slot" that is assigned once (in bottom-left-
first order) and never reassigned just because another particle elsewhere
was added or removed -- deleting a particle leaves a gap. The "Organize
Grid" button (see ParticleListPanel) explicitly compacts the grid, removing
gaps while preserving relative order.

Particles are tracked by a caller-supplied unique id (uid), not by list
position, so slot assignment survives additions/removals anywhere else in
the particle list. A uid that reappears after being removed -- via undo, or
detach/reattach -- returns to its old slot rather than being appended at
the end. It is deliberately *not* auto-selected just for reappearing here
(a uid reappearing isn't always an undo -- see the point above); undoing a
delete instead has TomoBoxer explicitly select the particles it actually
restored, in _reinsert_boxes.

The "All" tab is the one exception to the stable-slot rule: it groups
particles by class (Recycle Bin and Confirmed first, then alphabetically),
via ParticleGrid.sync_particles_grouped, which fully recomputes slots on
every call so classes stay visually clustered as particles move between
them. Per-class tabs are unaffected and keep the stable/gap-preserving
behavior described above.

Because that full recompute touches every particle's thumbnail and layout,
ParticleListPanel.sync_particles defers it (via _pending_all_entries) the
same way it already defers per-class tabs that aren't on screen -- "All"
only gets rebuilt when it's the tab actually being looked at, not on every
single particle change regardless of which tab is visible.
"""
from __future__ import annotations

import weakref

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from . import categories
from .widgets import TOOLBAR_BUTTON_STYLE, mark_destructive, space_key_state

_LEAVE_ONE_OUT_HELP = (
    "Leave-one-out excludes each particle's own contribution from the reference "
    "distribution before scoring it against the rest of the set -- more correct "
    "for finding outliers (a particle can't be an outlier relative to a "
    "distribution that includes itself), but it means recomputing the whole "
    "distribution once per particle instead of once for the whole set.\n\n"
    "Real measured cost (independent of embedding dimensionality):\n"
    "    50 particles      ~10 seconds\n"
    "    200 particles     ~45 seconds\n"
    "    1,000 particles   ~4-5 minutes\n"
    "    3,000 particles   ~15 minutes\n\n"
    "Only available when the reference and target set are the same (reviewing a "
    "set against itself) -- comparing one set against a different one always "
    "uses the whole reference distribution, since a particle that isn't a "
    "member of the reference set was never part of the distribution it's being "
    "compared to in the first place."
)

THUMB_SIZE = 96
MIN_THUMB = 24
MAX_THUMB = 256
SPACING = 4
UID_ROLE = 0


class _PlainClickEvent:
    """Stand-in for a real mouse event, for gestures (arrow-key grid
    navigation) that should behave exactly like a plain click -- single-
    select and jump to it -- without actually involving a mouse.
    TomoBoxer.img_selected only ever calls .modifiers() on what it's
    given."""

    @staticmethod
    def modifiers():
        return QtCore.Qt.NoModifier

_label_style = None  # lazily-built (bg_brush, no_pen, text_brush, font) shared by every
                      # particle's number-id label -- see _get_label_style()


def _get_label_style():
    """The pen/brushes/font used for every particle's number-id label are
    identical every time -- build them once and share the same instances
    (safe: QPen/QBrush/QFont are implicitly-shared value types that every
    setter below copies from) instead of constructing fresh ones per
    particle, which added up when populating a grid with many thousands of
    thumbnails at once. Built lazily rather than at import time since these
    need at least a QGuiApplication to exist."""
    global _label_style
    if _label_style is None:
        bg_brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 160))
        no_pen = QtGui.QPen(QtCore.Qt.NoPen)
        text_brush = QtGui.QBrush(QtGui.QColor("white"))
        font = QtGui.QFont()
        font.setPointSize(8)
        _label_style = (bg_brush, no_pen, text_brush, font)
    return _label_style


class ParticleGrid(QtWidgets.QGraphicsView):
    """Grid view: mx_image_selected(event, [uid]) is emitted on a plain
    click (navigates to that particle, as before). selection_changed([uids])
    is emitted whenever the multi-selection changes (plain click -> single
    item selection; Ctrl+click -> toggle; Shift+click -> range select)."""

    mx_image_selected = QtCore.Signal(object, object)
    selection_changed = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)  # so Escape (clear selection) reaches keyPressEvent below

        self._thumb_size = THUMB_SIZE
        self._mouse_mode = "App"
        self._items = {}        # uid -> QGraphicsPixmapItem
        self._images = {}       # uid -> raw numpy image (kept for re-scaling on zoom)
        self._slot_of_uid = {}  # uid -> stable slot index
        self._occupied_slots = set()  # always == set(self._slot_of_uid.values()) -- a live
                                 # mirror kept incrementally so _add_item's "is this old slot
                                 # free again" check (below) is O(1) instead of an O(current
                                 # size) dict.values() scan repeated once per re-added
                                 # particle, which made re-populating a large grid (e.g. after
                                 # a hidden set becomes visible again) quadratic.
        self._known_slot = {}   # uid -> last slot it ever occupied (survives removal,
                                 # so a particle brought back by undo returns to the
                                 # same spot instead of being appended at the end)
        self._next_slot = 0
        self._selected = set()   # cache of MY currently-highlighted uids, kept in sync
                                  # via apply_selection() -- never set directly by a
                                  # gesture; see ParticleListPanel._commit_selection
        self._selection_rects = {}  # uid -> QGraphicsRectItem
        self._last_clicked_uid = None
        self._last_known_slot = None  # last slot navigated/clicked to, kept even
                                  # once that uid is gone (recycled/confirmed/deleted)
                                  # so arrow-key navigation resumes from the same
                                  # spot instead of jumping back to the start --
                                  # see _navigate/_jump_to
        self._labels = {}        # uid -> (bg QGraphicsRectItem, text QGraphicsSimpleTextItem)
        self._border_rects = {}  # uid -> QGraphicsRectItem (class-color border, grouped views only)
        self._border_colors = {}  # uid -> QColor
        self._pan_last = None    # last screen pos while right-button panning
        self._levels = None      # (lo, hi) contrast override; None = per-thumbnail autocontrast

        self._update_scene_rect()

    def set_mouse_mode(self, mode):
        self._mouse_mode = mode

    def set_levels(self, lo, hi):
        """Apply an explicit (lo, hi) contrast range to every thumbnail
        here, overriding the default per-thumbnail mean/std autocontrast --
        mirrors ImageView2D.set_manual_levels so the main window's contrast
        slider affects the gallery identically."""
        self._levels = (float(lo), float(hi))
        self._refresh_all_pixmaps()

    def clear_levels(self):
        self._levels = None
        self._refresh_all_pixmaps()

    def _refresh_all_pixmaps(self):
        for uid, img in self._images.items():
            item = self._items.get(uid)
            if item is not None:
                item.setPixmap(self._render_pixmap(img))
                self._apply_item_scale(uid)

    def selected_uids(self):
        return list(self._selected)

    def apply_selection(self, selected_uids):
        """Show/hide this grid's own selection rects to match the given
        selection -- the one place that actually renders "is this uid
        selected." Called by ParticleListPanel whenever the shared
        selection changes (from any grid, or the tomogram window, or an
        Escape key press) and whenever this grid's own items get
        resynced, so a particle moved into a new class picks up its
        highlight immediately here instead of just losing it, and one
        removed from a class it's leaving quietly stops being highlighted
        here without that meaning it's not selected anymore elsewhere."""
        selected_uids = set(selected_uids) & self._items.keys()
        for uid in list(self._selection_rects.keys()):
            if uid not in selected_uids:
                self._hide_selection(uid)
        for uid in selected_uids:
            self._show_selection(uid)
        self._selected = selected_uids

    def clear_selection(self):
        """Request the shared selection be cleared -- the actual clearing/
        rendering happens via apply_selection(), echoed back once
        ParticleListPanel updates the selection every grid shares."""
        self._last_clicked_uid = None
        self.selection_changed.emit([])


    # -- syncing particles -----------------------------------------------
    def sync_particles(self, uid_img_pairs):
        """Reconcile displayed items with an iterable of (uid, image) pairs.
        New uids are appended to the next free slot (bottom-left first);
        uids no longer present are removed, leaving their slot empty;
        uids already shown just get their thumbnail image refreshed. A uid
        that reappears after having been removed returns to its old slot --
        see _add_item. Always un-dims every item -- dimming only means
        anything relative to apply_score_order's display, and this is the
        path any ordinary particle change (add/move/delete) goes through,
        so a stale dimmed particle should never survive one."""
        wanted = dict(uid_img_pairs)
        for uid in list(self._items.keys()):
            if uid not in wanted:
                self._remove_item(uid)
        for uid, img in uid_img_pairs:
            if uid in self._items:
                self._update_pixmap(uid, img)
            else:
                self._add_item(uid, img)
            self.set_dimmed(uid, False)
        self._update_scene_rect()

    def sync_particles_grouped(self, entries, class_order):
        """Like sync_particles, but groups items by class: entries is an
        iterable of (uid, image, classid) and class_order lists classids in
        the order their groups should appear (see categories.category_order).
        Unlike the plain per-class grids, slots are fully recomputed on
        every call -- that's unavoidable to keep classes clustered together
        as particles move between them -- so this is only used for the
        "All" tab. Each item also gets a border in its class's color so
        categories stay visually distinct even without column headers."""
        entries = list(entries)
        wanted = {uid for uid, _img, _cid in entries}
        for uid in list(self._items.keys()):
            if uid not in wanted:
                self._remove_item(uid)

        by_class = {}
        for uid, img, cid in entries:
            by_class.setdefault(cid, []).append((uid, img))

        self._slot_of_uid = {}
        cols = self._cols()
        slot = 0
        for cid in class_order:
            group = by_class.get(cid)
            if not group:
                continue
            if slot % cols != 0:
                slot += cols - (slot % cols)
            color = categories.class_color(cid)
            for uid, img in group:
                if uid in self._items:
                    self._update_pixmap(uid, img)
                    self._slot_of_uid[uid] = slot
                    self._known_slot[uid] = slot
                else:
                    self._add_item(uid, img, slot=slot)
                self.set_border_color(uid, color)
                slot += 1
        self._next_slot = slot
        self._occupied_slots = set(self._slot_of_uid.values())
        self._relayout_all()
        self._update_scene_rect()

    def apply_score_order(self, scores):
        """One-shot re-layout of whatever's already in this grid (nothing
        added or removed -- see sync_particles for that), ordered by the
        last computed Mahalanobis score:

        1. Particles with an embedding but no score yet (never computed,
           or reset by a class change) come first, in original-id (uid)
           order.
        2. Then every scored particle, ascending by score -- closest to
           the reference distribution first.
        3. Then, after a blank divider row and dimmed, anything with no
           embedding at all -- it can never be scored, so it's never
           reordered by this and stays out of the way rather than
           interleaved with particles that can be.

        This is a single snapshot, not an ongoing mode: nothing here
        remembers that this grid was "sorted by score," so the next
        particle added or moved just appends normally at the end (see
        _add_item), the same as if this had never been clicked. Calling
        it again re-derives the order fresh from whatever's changed
        since. scores: {uid: (score, has_embedding)} as built by
        TomoBoxer.update_boximgs; a uid missing from it is treated the
        same as (None, False) -- no embedding, never scored."""
        unscored, scored, unscorable = [], [], []
        for uid in self._items:
            score, has_embedding = scores.get(uid, (None, False))
            if not has_embedding:
                unscorable.append(uid)
            elif score is None:
                unscored.append(uid)
            else:
                scored.append((uid, score))
        unscored.sort()
        scored.sort(key=lambda t: t[1])
        unscorable.sort()

        cols = self._cols()
        self._slot_of_uid = {}
        slot = 0
        for uid in unscored:
            self._slot_of_uid[uid] = slot
            self._known_slot[uid] = slot
            self.set_dimmed(uid, False)
            slot += 1
        for uid, _score in scored:
            self._slot_of_uid[uid] = slot
            self._known_slot[uid] = slot
            self.set_dimmed(uid, False)
            slot += 1
        if unscorable:
            if slot % cols != 0:
                slot += cols - (slot % cols)
            slot += cols  # one full blank row as an explicit divider
            for uid in unscorable:
                self._slot_of_uid[uid] = slot
                self._known_slot[uid] = slot
                self.set_dimmed(uid, True)
                slot += 1
        self._next_slot = slot
        self._occupied_slots = set(self._slot_of_uid.values())
        self._relayout_all()
        self._update_scene_rect()

    def apply_id_order(self):
        """One-shot re-layout by uid (original ID) ascending, undoing
        whatever apply_score_order last did -- including un-dimming
        everything, since dimming only ever means anything relative to a
        distance-based order. Like apply_score_order, this is a single
        snapshot, not an ongoing mode."""
        ordered = sorted(self._items.keys())
        self._slot_of_uid = {}
        self._next_slot = 0
        for uid in ordered:
            self._slot_of_uid[uid] = self._next_slot
            self._known_slot[uid] = self._next_slot
            self.set_dimmed(uid, False)
            self._next_slot += 1
        self._occupied_slots = set(self._slot_of_uid.values())
        self._relayout_all()
        self._update_scene_rect()

    def uids_from_slot(self, uid):
        """uid and every other currently-displayed uid at an equal or
        later slot, in slot order -- i.e. "this one and the rest of the
        list" as it's actually laid out on screen right now, whatever
        that currently is (stable insertion order, or a one-shot Sort by
        Score/Original ID snapshot -- see apply_score_order/
        apply_id_order). Used for Shift+Ctrl+click (see
        TomoBoxer.img_selected), which has to follow what's visually on
        screen, not some other order invisible to whoever's looking at
        it -- particularly since a per-class grid's own slots only ever
        contain that class's particles, so this also naturally keeps the
        gesture from reaching into a different class entirely."""
        start = self._slot_of_uid.get(uid)
        if start is None:
            return [uid]
        return [u for u, s in sorted(self._slot_of_uid.items(), key=lambda kv: kv[1]) if s >= start]

    def reorganize(self):
        """Compact the grid: drop gaps left by deleted particles while
        keeping the relative (bottom-left-first) order of what remains."""
        ordered = sorted(self._slot_of_uid.keys(), key=lambda u: self._slot_of_uid[u])
        self._slot_of_uid = {}
        self._next_slot = 0
        for uid in ordered:
            self._slot_of_uid[uid] = self._next_slot
            self._known_slot[uid] = self._next_slot
            self._next_slot += 1
        self._occupied_slots = set(self._slot_of_uid.values())
        for uid in ordered:
            self._position_item(uid)
        self._update_scene_rect()

    # -- internal item management -----------------------------------------
    def _add_item(self, uid, img, slot=None):
        self._images[uid] = img
        if slot is None:
            known = self._known_slot.get(uid)
            if known is not None and known not in self._occupied_slots:
                slot = known
            else:
                slot = self._next_slot
                self._next_slot += 1
        self._known_slot[uid] = slot
        self._slot_of_uid[uid] = slot
        self._occupied_slots.add(slot)
        item = QtWidgets.QGraphicsPixmapItem(self._render_pixmap(img))
        item.setTransformationMode(QtCore.Qt.SmoothTransformation)
        item.setData(UID_ROLE, uid)
        self._scene.addItem(item)
        self._items[uid] = item
        self._add_label(uid)
        self._position_item(uid)

    def _remove_item(self, uid):
        item = self._items.pop(uid, None)
        if item is not None:
            self._scene.removeItem(item)
        self._images.pop(uid, None)
        slot = self._slot_of_uid.pop(uid, None)
        if slot is not None:
            self._occupied_slots.discard(slot)
        self._hide_selection(uid)
        self._hide_label(uid)
        self._border_colors.pop(uid, None)
        rect = self._border_rects.pop(uid, None)
        if rect is not None:
            self._scene.removeItem(rect)
        # Only a local rendering-cache cleanup -- this uid may well still
        # be selected overall (see ParticleListPanel._selected_uids); it
        # just doesn't have an item here anymore to show that on.
        self._selected.discard(uid)

    def _update_pixmap(self, uid, img):
        self._images[uid] = img
        item = self._items.get(uid)
        if item is not None:
            item.setPixmap(self._render_pixmap(img))
            self._apply_item_scale(uid)

    def _render_pixmap(self, img):
        """The one place that actually touches pixel data: the contrast
        stretch from the raw particle crop, at its native (box-size)
        resolution -- no resize here. Zoom is a display-only concern (see
        _apply_item_scale) so it never needs to re-run this."""
        return _thumbnail_pixmap(img, levels=self._levels)

    def _apply_item_scale(self, uid):
        """Scale the item to the current thumbnail size via a transform
        (with SmoothTransformation set on it in _add_item, the same
        bilinear-quality resampling a manual .scaled() call would do) rather
        than regenerating its pixmap -- this is what makes zoom cheap: it
        no longer touches pixel data at all, just this one number."""
        item = self._items.get(uid)
        if item is None:
            return
        pix = item.pixmap()
        native = max(pix.width(), pix.height())
        if native > 0:
            item.setScale(self._thumb_size / native)

    def _cols(self):
        w = max(self.viewport().width(), self._thumb_size + SPACING)
        return max(1, (w - SPACING) // (self._thumb_size + SPACING))

    def _position_item(self, uid):
        item = self._items.get(uid)
        if item is None:
            return
        self._apply_item_scale(uid)
        slot = self._slot_of_uid[uid]
        cols = self._cols()
        row, col = divmod(slot, cols)
        step = self._thumb_size + SPACING
        # Row 0 sits at the bottom (y in [-thumb, 0]); higher rows get more
        # negative y, i.e. they grow upward, so the first particle's slot
        # never moves as later rows are added above it.
        x = SPACING + col * step
        y = -SPACING - row * step - self._thumb_size
        item.setPos(x, y)
        rect = self._selection_rects.get(uid)
        if rect is not None:
            rect.setRect(x - 2, y - 2, self._thumb_size + 4, self._thumb_size + 4)
        self._position_label(uid)
        self._update_border(uid)

    # -- particle number-id label -------------------------------------------
    def _add_label(self, uid):
        bg_brush, no_pen, text_brush, font = _get_label_style()
        bg = self._scene.addRect(0, 0, 0, 0, no_pen, bg_brush)
        bg.setZValue(6)
        bg.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        text = self._scene.addSimpleText(str(uid))
        text.setBrush(text_brush)
        text.setFont(font)
        text.setZValue(7)
        text.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        self._labels[uid] = (bg, text)
        self._position_label(uid)

    def _position_label(self, uid):
        item = self._items.get(uid)
        lbl = self._labels.get(uid)
        if item is None or lbl is None:
            return
        bg, text = lbl
        pos = item.pos()
        text.setPos(pos.x() + 2, pos.y() + 1)
        br = text.boundingRect()
        bg.setRect(pos.x() + 1, pos.y() + 1, br.width() + 3, br.height())

    def _hide_label(self, uid):
        lbl = self._labels.pop(uid, None)
        if lbl is not None:
            bg, text = lbl
            self._scene.removeItem(bg)
            self._scene.removeItem(text)

    # -- per-item class-color border (used by the grouped "All" view) -------
    def set_border_color(self, uid, color):
        self._border_colors[uid] = color
        self._update_border(uid)

    def _update_border(self, uid):
        color = self._border_colors.get(uid)
        item = self._items.get(uid)
        if item is None or color is None:
            return
        rect = self._border_rects.get(uid)
        if rect is None:
            pen = QtGui.QPen(color)
            pen.setWidth(2)
            rect = self._scene.addRect(0, 0, 0, 0, pen)
            rect.setZValue(1)
            rect.setAcceptedMouseButtons(QtCore.Qt.NoButton)
            self._border_rects[uid] = rect
        else:
            pen = rect.pen()
            pen.setColor(color)
            rect.setPen(pen)
        pos = item.pos()
        rect.setRect(pos.x() - 1, pos.y() - 1, self._thumb_size + 2, self._thumb_size + 2)

    # -- dimming (used by apply_score_order for particles with no
    # embedding at all, which can never be scored/reordered) --------------
    def set_dimmed(self, uid, dimmed):
        item = self._items.get(uid)
        if item is not None:
            item.setOpacity(0.35 if dimmed else 1.0)

    def _relayout_all(self):
        for uid in self._items:
            self._position_item(uid)
        self._update_scene_rect()

    def _update_scene_rect(self):
        if self._items:
            rect = self._scene.itemsBoundingRect().adjusted(-SPACING, -SPACING, SPACING, SPACING)
        else:
            rect = QtCore.QRectF(0, -self._thumb_size - 2 * SPACING, self._thumb_size + 2 * SPACING, self._thumb_size + 2 * SPACING)
        self._scene.setSceneRect(rect)

    def _show_selection(self, uid):
        item = self._items.get(uid)
        if item is None:
            return
        rect = self._selection_rects.get(uid)
        if rect is None:
            pen = QtGui.QPen(QtGui.QColor("yellow"))
            pen.setWidth(3)
            rect = self._scene.addRect(0, 0, 0, 0, pen)
            rect.setZValue(10)
            # The highlight sits visually on top of the thumbnail, but must
            # not intercept mouse clicks -- otherwise itemAt() returns this
            # rect (which has no UID_ROLE data) instead of the pixmap item
            # underneath once a particle is selected, silently breaking
            # every click gesture (including Shift-click delete) on already
            # -selected particles.
            rect.setAcceptedMouseButtons(QtCore.Qt.NoButton)
            rect.setAcceptHoverEvents(False)
            self._selection_rects[uid] = rect
        pos = item.pos()
        rect.setRect(pos.x() - 2, pos.y() - 2, self._thumb_size + 4, self._thumb_size + 4)

    def _hide_selection(self, uid):
        rect = self._selection_rects.pop(uid, None)
        if rect is not None:
            self._scene.removeItem(rect)

    # -- zoom (mouse wheel) -----------------------------------------------
    def zoom_in(self):
        self._thumb_size = min(MAX_THUMB, int(round(self._thumb_size * 1.15)))
        self._relayout_all()

    def zoom_out(self):
        self._thumb_size = max(MIN_THUMB, int(round(self._thumb_size / 1.15)))
        self._relayout_all()

    # -- events -------------------------------------------------------------
    def wheelEvent(self, event):
        # Plain wheel zooms the thumbnails (matches the tomogram views);
        # Shift+wheel falls back to normal scrolling.
        if event.modifiers() & QtCore.Qt.ShiftModifier:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta == 0:
            event.accept()
            return
        # Zooming re-lays out every item (their slot positions scale with
        # thumb size), which would otherwise leave the view centered on
        # whatever raw scene point it happened to be at before -- rarely
        # where you actually want to keep looking. If something's
        # selected, keep it in view instead (its own position after the
        # relayout, not before); otherwise keep the point under the
        # cursor centered.
        anchor_uid = self._last_clicked_uid if self._last_clicked_uid in self._selected else None
        if anchor_uid is None and self._selected:
            anchor_uid = next(iter(self._selected))
        cursor_scene_pos = None
        if anchor_uid is None:
            cursor_scene_pos = self.mapToScene(event.position().toPoint())
        if delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()
        if anchor_uid is not None:
            item = self._items.get(anchor_uid)
            if item is not None:
                self.centerOn(item.sceneBoundingRect().center())
        elif cursor_scene_pos is not None:
            self.centerOn(cursor_scene_pos)
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._relayout_all()

    def mousePressEvent(self, event):
        self.setFocus(QtCore.Qt.MouseFocusReason)  # so Escape works right after clicking here
        if event.button() == QtCore.Qt.RightButton:
            # Grab-and-pan: right-click and drag to scroll the grid, same
            # gesture as the tomogram views (ImageView2D). A plain
            # right-click with no drag does nothing else -- it does NOT
            # select, since that made right-click redundant with (and
            # easily confused for) a left-click.
            self._pan_last = (event.position().x(), event.position().y())
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            event.accept()
            return
        if event.button() == QtCore.Qt.LeftButton:
            # itemAt() returns the strict topmost item regardless of
            # acceptedMouseButtons/acceptHoverEvents, so a selection
            # highlight rect (also drawn at this position, on top so it's
            # visible) would shadow the pixmap item underneath once a
            # particle is selected. Walk all items at this position instead
            # and use the first one that actually carries a UID.
            uid = None
            for candidate in self.items(event.pos()):
                cand_uid = candidate.data(UID_ROLE)
                if cand_uid is not None:
                    uid = cand_uid
                    break
            if uid is not None:
                mods = event.modifiers()
                # Shift(+Ctrl)+click sends this particle (or it and
                # everything after it in the list) to the Recycle Bin;
                # Space+click sends it to Confirmed -- both handled by
                # TomoBoxer.img_selected. Bypass multi-select entirely and
                # forward the click as-is.
                if mods & QtCore.Qt.ShiftModifier or space_key_state().held:
                    # This particle is about to leave this grid entirely
                    # (recycled/confirmed) -- remember where it sat so a
                    # subsequent arrow-key press resumes from here instead
                    # of jumping back to the start (see _navigate).
                    self._last_known_slot = self._slot_of_uid.get(uid)
                    self.mx_image_selected.emit(event, [uid])
                    event.accept()
                    return
                # Multi-select uses Ctrl (toggle one item) and Alt
                # (range-select from the last clicked item), since Shift
                # and Space are reserved for the gestures above.
                if mods & QtCore.Qt.AltModifier and self._last_clicked_uid is not None:
                    self._range_select(self._last_clicked_uid, uid)
                elif mods & QtCore.Qt.ControlModifier:
                    sel = set(self._selected)
                    if uid in sel:
                        sel.discard(uid)
                    else:
                        sel.add(uid)
                    self._last_clicked_uid = uid
                    self._last_known_slot = self._slot_of_uid.get(uid)
                    self.selection_changed.emit(list(sel))
                else:
                    self._last_clicked_uid = uid
                    self._last_known_slot = self._slot_of_uid.get(uid)
                    self.selection_changed.emit([uid])
                    self.mx_image_selected.emit(event, [uid])
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan_last is not None and (event.buttons() & QtCore.Qt.RightButton):
            x, y = event.position().x(), event.position().y()
            dx, dy = x - self._pan_last[0], y - self._pan_last[1]
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(dx))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(dy))
            self._pan_last = (x, y)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.RightButton:
            self._pan_last = None
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    _ARROW_KEYS = (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right, QtCore.Qt.Key_Up, QtCore.Qt.Key_Down)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.clear_selection()
            event.accept()
            return
        if event.key() in self._ARROW_KEYS:
            self._navigate(event.key())
            event.accept()
            return
        super().keyPressEvent(event)

    def _navigate(self, key):
        """Arrow keys move to the neighboring particle in that direction
        and jump to it exactly like clicking it would (same as a plain
        click: single-select it and center the tomogram views on it).

        Left/Right move to the previous/next occupied slot in slot order
        (not strictly "same row"): since slots are assigned bottom-row-
        first, left-to-right (see the module docstring), this is exactly
        reading order, so Right off the end of a row continues onto the
        next row up rather than stopping dead at the row boundary. Up/
        Down move a full row within the same column instead -- row 0 is
        the bottom row (see _position_item), so Up means a numerically
        *higher* row and Down a lower one.

        The anchor is _last_known_slot, a plain slot number that outlives
        whatever uid was last there -- so once that particle leaves this
        grid entirely (recycled/confirmed/deleted), the next arrow press
        still resumes from the same spot instead of falling back to the
        very first particle. If nothing has ever been selected/navigated
        to here, the first (lowest-slot) particle is picked instead."""
        if not self._slot_of_uid:
            return
        if self._last_known_slot is None:
            self._jump_to(min(self._slot_of_uid, key=self._slot_of_uid.get))
            return
        anchor_slot = self._last_known_slot
        slot_to_uid = {s: u for u, s in self._slot_of_uid.items()}
        if key == QtCore.Qt.Key_Right:
            candidates = sorted(s for s in slot_to_uid if s > anchor_slot)
        elif key == QtCore.Qt.Key_Left:
            candidates = sorted((s for s in slot_to_uid if s < anchor_slot), reverse=True)
        else:
            cols = self._cols()
            row, col = divmod(anchor_slot, cols)
            if key == QtCore.Qt.Key_Up:
                candidates = [(row + 1) * cols + col]
            else:  # Key_Down
                candidates = [(row - 1) * cols + col] if row > 0 else []
        for slot in candidates:
            uid = slot_to_uid.get(slot)
            if uid is not None:
                self._jump_to(uid)
                return

    def _jump_to(self, uid):
        self._last_clicked_uid = uid
        self._last_known_slot = self._slot_of_uid.get(uid)
        self.selection_changed.emit([uid])
        self.mx_image_selected.emit(_PlainClickEvent(), [uid])

    def _range_select(self, from_uid, to_uid):
        self._last_known_slot = self._slot_of_uid.get(to_uid)
        if from_uid not in self._slot_of_uid or to_uid not in self._slot_of_uid:
            self.selection_changed.emit([to_uid])
            return
        lo, hi = sorted((self._slot_of_uid[from_uid], self._slot_of_uid[to_uid]))
        uids = [u for u, s in self._slot_of_uid.items() if lo <= s <= hi]
        uids.sort(key=lambda u: self._slot_of_uid[u])
        self._last_clicked_uid = to_uid
        self.selection_changed.emit(uids)


def _thumbnail_pixmap(img: np.ndarray, levels=None) -> QtGui.QPixmap:
    """levels, when given, is an explicit (lo, hi) display range (the
    contrast slider) used instead of this thumbnail's own mean/std -- so
    manual contrast applies identically here and in the tomogram views."""
    img = np.asarray(img, dtype=np.float32)
    if levels is not None:
        lo, hi = levels
    else:
        mean, std = float(img.mean()), float(img.std())
        lo, hi = (mean - 3 * std, mean + 3 * std) if std else (mean - 1, mean + 1)
    if hi <= lo:
        hi = lo + 1
    buf = np.clip((img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    buf = np.ascontiguousarray(buf)
    h, w = buf.shape
    qimg = QtGui.QImage(buf.tobytes(), w, h, w, QtGui.QImage.Format_Grayscale8).copy()
    return QtGui.QPixmap.fromImage(qimg)


class DetachedGridWindow(QtWidgets.QWidget):
    """A floating top-level window holding a single class's ParticleGrid
    that has been "detached" out of the tabbed panel. Closing it (via the
    window's [x] or the explicit Reattach button) sends the grid back to
    its tab in the parent ParticleListPanel."""

    closed = QtCore.Signal(int)  # classid

    def __init__(self, classid, grid, title, visibility_checkbox, color_swatch):
        super().__init__()
        self.classid = classid
        self.grid = grid
        self.setWindowTitle(title)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # The docked tab's visibility checkbox and color swatch both go
        # away with the tab when detaching (they're embedded in the tab
        # bar) -- without replacements here, there'd be no way to
        # show/hide this set or see/change its color while its window is
        # floating.
        toprow = QtWidgets.QHBoxLayout()
        self.visibility_checkbox = visibility_checkbox
        self.visibility_checkbox.setText("Visible in tomogram view")
        toprow.addWidget(self.visibility_checkbox)
        toprow.addStretch(1)
        self.color_swatch = color_swatch
        toprow.addWidget(self.color_swatch)
        layout.addLayout(toprow)

        layout.addWidget(grid)
        # QTabWidget/QStackedWidget explicitly hide()s a page when it is
        # removed (removeTab), and that explicit-hidden flag survives being
        # reparented into this window's layout -- so force it visible again.
        grid.setVisible(True)
        self.reattach_button = QtWidgets.QPushButton("Reattach")
        layout.addWidget(self.reattach_button)
        self.reattach_button.clicked.connect(self.close)
        self.resize(420, 480)

    def closeEvent(self, event):
        self.closed.emit(self.classid)
        super().closeEvent(event)


class _ConstrainedTabBar(QtWidgets.QTabBar):
    """A movable QTabBar that only lets normal-set tabs actually be picked
    up for dragging -- "All", Recycle Bin, Confirmed, and "+" stay exactly
    where ParticleListPanel puts them, without disabling movable tabs
    altogether. `locked_check(index)` and `drag_bounds()` are filled in by
    the owning panel once it exists (this class has no knowledge of
    classids itself).

    Blocking a locked tab from *starting* its own drag (mousePressEvent
    below) stops it from ever moving, but Qt's native drag-reorder still
    physically swaps whatever tab a dragged one is passing over -- so
    without more, dragging a normal tab far enough would still shove "+"
    or a fixed category out of place for the rest of the gesture (visibly
    flickering) before a corrective moveTab() elsewhere snapped it back.
    Clamping the event position actually fed to Qt's own drag handling in
    mouseMoveEvent, to a fixed pixel range recorded at press-time, keeps
    the drag from ever reaching those tabs in the first place -- nothing
    to correct after the fact, no flicker."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.locked_check = lambda index: False
        self.drag_bounds = lambda: None  # () -> (lo, hi) local-x clamp, or None
        self._clamp = None

    def mousePressEvent(self, event):
        idx = self.tabAt(event.pos())
        if idx >= 0 and self.locked_check(idx):
            self.setCurrentIndex(idx)
            event.accept()
            return
        self._clamp = self.drag_bounds() if idx >= 0 else None
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._clamp is not None:
            lo, hi = self._clamp
            pos = event.position()
            x = min(max(pos.x(), lo), hi)
            if x != pos.x():
                event = QtGui.QMouseEvent(
                    event.type(), QtCore.QPointF(x, pos.y()), event.globalPosition(),
                    event.button(), event.buttons(), event.modifiers())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._clamp = None
        super().mouseReleaseEvent(event)


class ParticleListPanel(QtWidgets.QWidget):
    """Wraps a tabbed set of ParticleGrids: an "All" tab showing every
    visible particle, one tab per class (rebuilt via rebuild_tabs()), and a
    permanent "+" tab at the end that prompts for a name and creates a new
    set instead of ever actually being shown. Any class tab can be
    "detached" into its own floating window with the Detach button, and
    reattached by closing that window.

    This is also where Sets management now lives, replacing the standalone
    Options window's Sets panel: New/Rename/Delete/visibility all act on
    tabs directly (a new tab, double-click to rename, a Delete Set button,
    a per-tab visibility checkbox), rather than duplicating that as a
    separate list somewhere else."""

    mx_image_selected = QtCore.Signal(object, object)
    selection_changed = QtCore.Signal(object)  # list of uids currently selected
    class_activated = QtCore.Signal(int)  # classid, when a normal class's tab becomes current
    active_tab_changed = QtCore.Signal(object)  # classid (any -- normal or fixed), or None for "All"

    def __init__(self, target=None, parent=None):
        super().__init__(parent)
        self._target = weakref.ref(target) if target is not None else None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        # Matches the flat "chip" look of the tomogram window's own
        # toolbars -- shared rather than redefined here since this is a
        # separate top-level window with no widget-hierarchy stylesheet
        # cascade from that one.
        self.setStyleSheet(self.styleSheet() + TOOLBAR_BUTTON_STYLE)

        # Toggle Colors / Show Active Set Only / Show All Sets: the
        # QActions themselves live on TomoBoxer (it owns sets_visible and
        # the color rotation), but the buttons for them live here instead
        # of the tomogram window -- a QAction isn't tied to whichever
        # toolbar displays it, so this is the exact same objects, just
        # shown in a different window, with no duplicated logic.
        if target is not None:
            self.view_toolbar = QtWidgets.QToolBar()
            self.view_toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
            self.view_toolbar.addAction(target.toggle_colors_action)
            self.view_toolbar.addSeparator()
            self.view_toolbar.addAction(target.view_only_selected_action)
            self.view_toolbar.addAction(target.show_all_sets_action)
            layout.addWidget(self.view_toolbar)

        # Deliberately the loudest thing in this window: computing/sorting
        # by Mahalanobis distance used to live as a File menu item (then a
        # popup dialog triggered from a button), which made it easy to miss
        # or fiddly to reach. Now it's its own group, inline, right here --
        # the reference set/embedding space/leave-one-out controls plus the
        # big green "go" button all live together with nothing modal in
        # between, so the whole feature reads and edits in place.
        self.compute_distance_group = QtWidgets.QWidget()
        group_layout = QtWidgets.QVBoxLayout(self.compute_distance_group)
        group_layout.setContentsMargins(0, 0, 0, 0)
        group_layout.setSpacing(3)

        # One row for everything: which set gets ranked (the current tab,
        # kept live-updated by _update_compute_distance_state as you switch
        # tabs, rather than leaving it implicit), what it's ranked close to,
        # optionally what it's ranked far from (the "double distance"/
        # contrastive score -- see TomoBoxer.compute_mahalanobis), which
        # embedding space, and leave-one-out -- all one coherent
        # sentence-like control instead of splitting it across rows.
        controls_row = QtWidgets.QHBoxLayout()
        controls_row.addWidget(QtWidgets.QLabel("Rank:"))
        self.compute_distance_target_label = QtWidgets.QLabel()
        self.compute_distance_target_label.setStyleSheet("font-weight: bold;")
        controls_row.addWidget(self.compute_distance_target_label)
        controls_row.addWidget(QtWidgets.QLabel("close to"))
        self.compute_distance_reference_combo = QtWidgets.QComboBox()
        self.compute_distance_reference_combo.setToolTip(
            "The set whose embedding distribution the current tab's particles "
            "are compared to (e.g. Confirmed) -- can be the current tab itself, "
            "to spot outliers within it.")
        self.compute_distance_reference_combo.currentIndexChanged.connect(
            self._update_compute_distance_state)
        controls_row.addWidget(self.compute_distance_reference_combo)
        controls_row.addWidget(QtWidgets.QLabel("far from"))
        self.compute_distance_negative_combo = QtWidgets.QComboBox()
        self.compute_distance_negative_combo.addItem("(none)", None)
        self.compute_distance_negative_combo.setToolTip(
            "Optional: also rank away from this set's own embedding distribution "
            "(e.g. Recycle Bin) -- the rank becomes distance-to-\"close to\" minus "
            "distance-to-\"far from\", so particles near the first and away from "
            "the second sort first. Leave at \"(none)\" for a plain single-distance "
            "rank, same as before.")
        self.compute_distance_negative_combo.currentIndexChanged.connect(
            self._update_compute_distance_state)
        controls_row.addWidget(self.compute_distance_negative_combo)
        # Populated from whatever embedding spaces are actually attached
        # (see _refresh_compute_distance_controls) rather than hardcoded --
        # several methods can be attached at once and each contributes its
        # full embedding and its UMAP as separately rankable spaces.
        self.compute_distance_space_combo = QtWidgets.QComboBox()
        self.compute_distance_space_combo.setToolTip(
            "Which embedding to measure distances in. Every attached method "
            "contributes its full embedding and, when one was given, its UMAP.")
        controls_row.addWidget(self.compute_distance_space_combo)
        self.compute_distance_loo_check = QtWidgets.QCheckBox("Leave-One-Out")
        controls_row.addWidget(self.compute_distance_loo_check)
        loo_help_button = QtWidgets.QToolButton()
        loo_help_button.setText("?")
        loo_help_button.setToolTip("What does leave-one-out mean, and how slow is it?")
        loo_help_button.clicked.connect(self._show_leave_one_out_help)
        controls_row.addWidget(loo_help_button)
        controls_row.addStretch(1)
        group_layout.addLayout(controls_row)

        self.compute_distance_button = QtWidgets.QPushButton("▶  Rank by Embedding Distance")
        # Saved so _update_compute_distance_state can restore it after
        # temporarily swapping in a "why is this disabled" tooltip (for a
        # same-set conflict) -- see there.
        self._compute_distance_button_tip = (
            "Ranks the current tab's particles by their Mahalanobis distance to "
            "the \"close to\" set's embedding distribution, then lays them out "
            "closest-first. Set a \"far from\" set too for a contrastive rank: "
            "distance-to-\"close to\" minus distance-to-\"far from\", favoring "
            "particles near the first and away from the second. \"Sort by Rank\" "
            "(below) re-displays this same result later without recomputing it.")
        self.compute_distance_button.setToolTip(self._compute_distance_button_tip)
        self.compute_distance_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.compute_distance_button.setStyleSheet("""
            QPushButton {
                background-color: #2e7d32;
                color: white;
                font-weight: bold;
                font-size: 13px;
                padding: 8px 14px;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #388e3c; }
            QPushButton:pressed { background-color: #1b5e20; }
            QPushButton:disabled { background-color: #9e9e9e; }
        """)
        self.compute_distance_button.clicked.connect(self._run_compute_distance)
        group_layout.addWidget(self.compute_distance_button)

        # Menu-only equivalent of the button above (TomoBoxer's Set menu),
        # kept in sync with compute_distance_group's own enabled state by
        # _update_compute_distance_state -- there is deliberately no
        # separate toolbar button for it, since the big green one already
        # serves that role.
        self.rank_action = QtGui.QAction("Rank by Embedding Distance", self)
        self.rank_action.setToolTip(self.compute_distance_button.toolTip())
        self.rank_action.triggered.connect(self._run_compute_distance)

        self.compute_distance_group.setEnabled(False)  # no embeddings loaded yet
        # Added to the layout further down, right below the tab grid --
        # see the trailing layout.addWidget(self.compute_distance_group)
        # call -- so it sits next to Sort by Original ID/Sort by Rank
        # rather than above the tabs entirely: all three act on "whatever
        # tab you're currently looking at," so they read as one group.

        self.tabs = QtWidgets.QTabWidget()
        # setTabBar must happen before any tabs are added (Qt requirement).
        # setMovable(True) plus _ConstrainedTabBar is what lets normal sets
        # be dragged into any order while "All"/Recycle Bin/Confirmed/"+"
        # stay pinned -- see _is_tab_locked and _on_tab_moved below.
        self.tabs.setTabBar(_ConstrainedTabBar())
        self.tabs.tabBar().locked_check = self._is_tab_locked
        self.tabs.tabBar().drag_bounds = self._tab_drag_bounds
        self.tabs.setMovable(True)
        self.tabs.tabBar().tabMoved.connect(self._on_tab_moved)
        layout.addWidget(self.tabs)

        self.grid = ParticleGrid()
        self.tabs.addTab(self.grid, "All")
        self._connect_grid(self.grid)

        self._class_grids = {}     # classid -> ParticleGrid (docked, in a tab)
        self._class_names = {}     # classid -> name currently shown
        self._detached = {}        # classid -> DetachedGridWindow
        self._levels_state = None  # (lo, hi) currently applied, so a newly created grid gets it too
        self._pending_entries = {}  # classid -> [(uid, img), ...] not yet applied to a
                                     # not-currently-shown docked tab's grid (see sync_particles)
        self._pending_all_entries = None  # same idea, but for the "All" tab specifically
                                     # (see sync_particles/_on_tab_changed) -- None means
                                     # "All" is already current, nothing deferred
        self._visibility_checkboxes = {}  # classid -> QCheckBox, embedded in that tab
        self._color_swatches = {}  # classid -> QToolButton (color swatch), embedded in that tab
        self._pinning_fixed_tabs = False  # reentrancy guard, see _pin_fixed_tabs
        self._selected_uids = set()  # the one authoritative selection -- see _commit_selection
        self._uid_scores = {}  # uid -> (score, has_embedding), refreshed by every sync_particles
                                # call -- read by the one-shot "Sort by Rank" button
                                # (apply_score_order) so it doesn't need fresh data from TomoBoxer

        self._plus_placeholder = QtWidgets.QWidget()
        self.tabs.addTab(self._plus_placeholder, "+")
        self.tabs.setTabToolTip(self.tabs.indexOf(self._plus_placeholder), "Create a new set")
        self._last_tab_index = 0

        # Menu-only equivalents of the "+" tab and double-clicking a tab
        # (TomoBoxer's Set menu) -- both gestures already exist and work
        # fine, these just make them reachable without knowing that.
        self.new_set_action = QtGui.QAction("New Set...", self)
        self.new_set_action.setToolTip("Create a new set.")
        self.new_set_action.triggered.connect(self._prompt_new_set)

        self.rename_set_action = QtGui.QAction("Rename Set...", self)
        self.rename_set_action.setToolTip("Rename the current tab's set.")
        self.rename_set_action.triggered.connect(self._prompt_rename_current_set)

        # The Rank panel operates on "whatever tab is currently on screen,"
        # same as every action in the toolbar just below it -- keeping them
        # adjacent (rather than the Rank panel sitting above the tabs
        # entirely, as it used to) makes that relationship visible instead
        # of implied.
        layout.addWidget(self.compute_distance_group)

        # A real QToolBar (not a bare row of QPushButtons) so these pick up
        # the exact same flat "chip" look as every other toolbar in the
        # app, rather than looking like a different, unstyled widget glued
        # on underneath. Grouped with separators by what they actually do:
        # grid/window layout, then re-ordering, then the two genuinely
        # destructive actions (marked in red -- see widgets.mark_destructive
        # -- so they read as different in kind from the harmless ones next
        # to them, not just alphabetically nearby).
        self.gallery_toolbar = QtWidgets.QToolBar()
        self.gallery_toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)

        self.organize_action = QtGui.QAction("Organize Grid", self)
        self.organize_action.setToolTip("Compact the grid, removing gaps left by deleted particles.")
        self.organize_action.triggered.connect(lambda: self._active_grid().reorganize())
        self.gallery_toolbar.addAction(self.organize_action)

        self.detach_action = QtGui.QAction("Detach Tab", self)
        self.detach_action.setToolTip("Pop the current tab's grid into its own floating window.")
        self.detach_action.setEnabled(False)  # starts on "All", which can't be detached
        self.detach_action.triggered.connect(self._detach_current_tab)
        self.gallery_toolbar.addAction(self.detach_action)

        self.gallery_toolbar.addSeparator()

        # One-shot re-layout actions, not modes: each click snapshots the
        # current order and stops there -- neither one keeps re-sorting as
        # more particles are added or moved afterward (see apply_id_order/
        # apply_score_order). "Original ID" restores plain uid order,
        # undoing whatever "By Rank" last did (including its dimming); "By
        # Rank" reads whatever was last computed via the "Rank by Embedding
        # Distance" button above, it doesn't compute anything itself.
        self.sort_by_id_action = QtGui.QAction("Sort by Original ID", self)
        self.sort_by_id_action.setToolTip(
            "One-time re-layout of this set's particles by their original ID.")
        self.sort_by_id_action.setEnabled(False)  # starts on "All", not a real set
        self.sort_by_id_action.triggered.connect(self._sort_current_by_id)
        self.gallery_toolbar.addAction(self.sort_by_id_action)

        self.sort_by_rank_action = QtGui.QAction("Sort by Rank", self)
        self.sort_by_rank_action.setToolTip(
            "One-time re-layout of this set's particles by their rank -- the "
            "distance last computed by \"Rank by Embedding Distance\" above "
            "(closest first; particles never ranked come first of all). "
            "Doesn't compute anything itself, only re-displays the last result.")
        self.sort_by_rank_action.setEnabled(False)  # starts on "All", not a real set
        self.sort_by_rank_action.triggered.connect(self._sort_current_by_score)
        self.gallery_toolbar.addAction(self.sort_by_rank_action)

        self.gallery_toolbar.addSeparator()

        self.delete_set_action = QtGui.QAction("Delete Set", self)
        self.delete_set_action.setToolTip("Delete this set, sending every particle in it to the Recycle Bin.")
        self.delete_set_action.setEnabled(False)
        self.delete_set_action.triggered.connect(self._delete_current_set)
        self.gallery_toolbar.addAction(self.delete_set_action)
        mark_destructive(self.gallery_toolbar.widgetForAction(self.delete_set_action))

        self.empty_recycle_action = QtGui.QAction("Empty Recycle Bin", self)
        self.empty_recycle_action.setToolTip(
            "Clear everything currently in the Recycle Bin -- archived to a "
            "hidden backup file first (not loaded again), and still undoable "
            "with Ctrl+Z right after.")
        self.empty_recycle_action.setVisible(False)  # only shown while viewing the Recycle Bin tab
        self.empty_recycle_action.triggered.connect(self._empty_recycle_bin)
        self.gallery_toolbar.addAction(self.empty_recycle_action)
        mark_destructive(self.gallery_toolbar.widgetForAction(self.empty_recycle_action))

        layout.addWidget(self.gallery_toolbar)

        # Old attribute names kept as aliases to the actions' own toolbar
        # widgets (still real, clickable QToolButtons) rather than removed
        # outright -- nothing outside this method should need them (the
        # actions above are the real thing now), but this keeps anything
        # that still reaches in by the old button names working unchanged.
        self.organize_button = self.gallery_toolbar.widgetForAction(self.organize_action)
        self.detach_button = self.gallery_toolbar.widgetForAction(self.detach_action)
        self.delete_set_button = self.gallery_toolbar.widgetForAction(self.delete_set_action)
        self.sort_by_id_button = self.gallery_toolbar.widgetForAction(self.sort_by_id_action)
        self.sort_by_score_button = self.gallery_toolbar.widgetForAction(self.sort_by_rank_action)
        self.empty_recycle_button = self.gallery_toolbar.widgetForAction(self.empty_recycle_action)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.tabs.tabBar().tabBarDoubleClicked.connect(self._on_tab_double_clicked)

    def _target_or_none(self):
        return self._target() if self._target is not None else None

    def _active_classid(self):
        w = self.tabs.currentWidget()
        for cid, grid in self._class_grids.items():
            if grid is w:
                return cid
        return None

    def active_view_classid(self):
        """Classid of whichever tab the gallery is actually showing right
        now -- a normal set *or* a fixed category (Recycle Bin/Confirmed)
        -- or None for "All". Unlike TomoBoxer.currentset (which only
        tracks the active *picking* class and never changes for "All" or
        the fixed categories) or class_activated (which only fires for
        normal classes), this always reflects literally what's on screen;
        used by Show Active Set Only / Show All Sets to key off the tab
        you're actually looking at rather than the picking class."""
        return self._active_classid()

    def _on_tab_changed(self, index):
        widget = self.tabs.widget(index)
        if widget is self._plus_placeholder:
            self._prompt_new_set()
            return
        self._last_tab_index = index

        cid = self._active_classid()
        is_fixed = cid in categories.FIXED_CLASS_IDS if cid is not None else False
        self.delete_set_action.setEnabled(cid is not None and not is_fixed)
        self.detach_action.setEnabled(cid is not None)  # "All" can't be detached
        self.empty_recycle_action.setVisible(cid == categories.RECYCLE_BIN_ID)
        self.sort_by_id_action.setEnabled(cid is not None)
        self.sort_by_rank_action.setEnabled(cid is not None)
        self._update_compute_distance_state()
        if cid is not None and not is_fixed:
            self.class_activated.emit(cid)
        # Unlike class_activated (only normal/pickable classes), this
        # fires for *every* real tab -- Recycle Bin, Confirmed, and "All"
        # (cid None) included -- since Show Active Set Only needs to follow
        # whatever the gallery is showing regardless of whether that tab
        # could ever become the active picking class.
        self.active_tab_changed.emit(cid)

        if widget is self.grid:
            if self._pending_all_entries is not None:
                self.grid.sync_particles_grouped(self._pending_all_entries, self._all_visible_class_order())
                self.grid.apply_selection(self._selected_uids)
                self._pending_all_entries = None
            return

        for classid, grid in self._class_grids.items():
            if grid is widget:
                pending = self._pending_entries.pop(classid, None)
                if pending is not None:
                    grid.sync_particles(pending)
                    grid.apply_selection(self._selected_uids)
                break

    # -- sets management (New/Rename/Delete/visibility/Save/Empty Bin) ------
    def _prompt_new_set(self):
        target = self._target_or_none()
        self.tabs.setCurrentIndex(self._last_tab_index)  # never linger on "+"
        if target is None:
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "New Set", "Enter a name for the new set:")
        if not ok:
            return
        name = str(name).strip()
        if not name:
            return
        if name in target.sets.values():
            QtWidgets.QMessageBox.information(self, "New Set", "A set with that name already exists.")
            return
        cid = target.new_set(name)
        self.rebuild_tabs(target.sets)
        grid = self._class_grids.get(cid)
        if grid is not None:
            self.tabs.setCurrentWidget(grid)

    def _prompt_rename_current_set(self):
        self._on_tab_double_clicked(self.tabs.currentIndex())

    def _on_tab_double_clicked(self, index):
        widget = self.tabs.widget(index)
        if widget is self.grid or widget is self._plus_placeholder:
            return  # "All" and "+" aren't renamable
        target = self._target_or_none()
        classid = None
        for cid, grid in self._class_grids.items():
            if grid is widget:
                classid = cid
                break
        if target is None or classid is None:
            return
        if classid in categories.FIXED_CLASS_IDS:
            QtWidgets.QMessageBox.information(
                self, "Rename Set", "Recycle Bin and Confirmed are fixed categories and can't be renamed.")
            return
        oldname = target.sets[classid]
        name, ok = QtWidgets.QInputDialog.getText(self, "Rename Set", "Enter a new name:", text=oldname)
        if not ok:
            return
        name = str(name).strip()
        if not name or name == oldname:
            return
        if name in target.sets.values():
            QtWidgets.QMessageBox.information(self, "Rename Set", "A set with that name already exists.")
            return
        target.rename_set(f"{classid}::{oldname}", name)
        self.rebuild_tabs(target.sets)

    def _delete_current_set(self):
        target = self._target_or_none()
        cid = self._active_classid()
        if target is None or cid is None or cid in categories.FIXED_CLASS_IDS:
            return
        name = target.sets.get(cid)
        reply = QtWidgets.QMessageBox.question(
            self, "Delete Set",
            f'Are you sure you want to delete "{name}"? Every particle in it will be sent to the Recycle Bin.',
            QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            # Switch away from the tab being removed *before* removing it --
            # otherwise Qt shifts the current tab onto whatever ends up in
            # its place, which (when deleting the last real class) is the
            # "+" placeholder, immediately firing an unwanted "create a new
            # set" prompt as a side effect of deleting one.
            self.tabs.setCurrentWidget(self.grid)
            target.delete_set(f"{cid}::{name}")
            self.rebuild_tabs(target.sets)

    def _refresh_compute_distance_controls(self):
        """Keeps the reference-set combos in sync with whatever sets exist
        (called from rebuild_tabs, so on every set add/rename/delete),
        preserving the current selection where possible, and refreshes the
        UMAP item's presence/availability. Actual enabled-state (whether
        embeddings are loaded, and whether leave-one-out applies) is
        _update_compute_distance_state's job, called at the end of this."""
        target = self._target_or_none()
        if target is None:
            return
        combo = self.compute_distance_reference_combo
        previous = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for cid in categories.category_order(target.sets):
            combo.addItem(target.sets[cid], cid)
        idx = combo.findData(previous) if previous is not None else -1
        if idx < 0:
            idx = combo.findData(categories.CONFIRMED_ID)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        combo.blockSignals(False)

        # Same set list as the reference combo, plus a leading "(none)"
        # (data=None) that keeps the plain, single-distance rank -- the
        # default, and the only option before this control existed.
        negative_combo = self.compute_distance_negative_combo
        previous_negative = negative_combo.currentData()
        negative_combo.blockSignals(True)
        negative_combo.clear()
        negative_combo.addItem("(none)", None)
        for cid in categories.category_order(target.sets):
            negative_combo.addItem(target.sets[cid], cid)
        idx = negative_combo.findData(previous_negative) if previous_negative is not None else 0
        negative_combo.setCurrentIndex(idx if idx >= 0 else 0)
        negative_combo.blockSignals(False)

        # One entry per attached space, so several methods can be compared
        # from the same dropdown ("CryoSIAM - full (1024-d)", "TomoTwin -
        # UMAP (2-d)", ...). Rebuilt wholesale rather than patched item by
        # item, since attaching a method adds two at once; the current
        # choice is preserved when it's still available.
        space_combo = self.compute_distance_space_combo
        previous_space = space_combo.currentData()
        space_combo.blockSignals(True)
        space_combo.clear()
        for name, entry in target.embedding_spaces.items():
            space_combo.addItem(entry["label"], name)
        idx = space_combo.findData(previous_space) if previous_space else -1
        space_combo.setCurrentIndex(idx if idx >= 0 else 0)
        space_combo.blockSignals(False)

        self._update_compute_distance_state()

    def _update_compute_distance_state(self):
        target = self._target_or_none()
        cid = self._active_classid()
        self.compute_distance_target_label.setText(
            target.sets[cid] if target is not None and cid is not None else "—")
        has_embeddings = target is not None and bool(target.embedding_spaces)
        reference_cid = self.compute_distance_reference_combo.currentData()
        negative_cid = self.compute_distance_negative_combo.currentData()
        conflict = negative_cid is not None and negative_cid == reference_cid

        # No embeddings at all isn't fixable from anything in this group
        # (you'd need to import some elsewhere first), so graying out the
        # whole thing -- combos included -- is fine there. A same-set
        # conflict is different: it's fixable right here, by changing
        # either combo, so disabling the *group* (which contains those
        # very combos) would trap the user with no way to undo their own
        # choice. Only the actual "go" button (and its menu equivalent)
        # get disabled for a conflict, with a tooltip explaining why;
        # every combo stays live regardless.
        group_enabled = has_embeddings and cid is not None
        self.compute_distance_group.setEnabled(group_enabled)
        self.compute_distance_group.setToolTip(
            "" if has_embeddings else
            "No embeddings are loaded -- import a particle list with an embeddings "
            "bundle attached first (File > Import Particle List with Embeddings...).")

        can_run = group_enabled and not conflict
        self.compute_distance_button.setEnabled(can_run)
        self.rank_action.setEnabled(can_run)
        self.compute_distance_button.setToolTip(
            "\"close to\" and \"far from\" can't be the same set." if conflict
            else self._compute_distance_button_tip)

        # Leave-one-out is relevant (and only ever applied) to whichever
        # term -- "close to", "far from", or both -- actually equals the
        # current tab; see TomoBoxer.compute_mahalanobis.
        same = group_enabled and (reference_cid == cid or negative_cid == cid)
        self.compute_distance_loo_check.setEnabled(same)
        if not same:
            self.compute_distance_loo_check.setChecked(False)
        self.compute_distance_loo_check.setToolTip(
            "" if same else "Only available when \"close to\" and/or \"far from\" is the current tab itself.")

    def _show_leave_one_out_help(self):
        QtWidgets.QMessageBox.information(self, "Leave-One-Out", _LEAVE_ONE_OUT_HELP)

    def _run_compute_distance(self):
        target = self._target_or_none()
        cid = self._active_classid()
        if target is None or cid is None:
            return
        target.run_compute_distance(
            target_classid=cid,
            reference_classid=self.compute_distance_reference_combo.currentData(),
            space=self.compute_distance_space_combo.currentData(),
            leave_one_out=(self.compute_distance_loo_check.isEnabled()
                            and self.compute_distance_loo_check.isChecked()),
            negative_classid=self.compute_distance_negative_combo.currentData())

    def _empty_recycle_bin(self):
        target = self._target_or_none()
        if target is None:
            return
        count = sum(1 for b in target.boxes if b[5] == categories.RECYCLE_BIN_ID)
        if count == 0:
            return
        reply = QtWidgets.QMessageBox.question(
            self, "Empty Recycle Bin",
            f"Clear {count} particle(s) from the Recycle Bin? They'll be archived to a "
            "hidden backup file in the working directory first (not loaded again), and "
            "this is still undoable with Ctrl+Z right after.",
            QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            target.empty_recycle_bin()

    def _sort_current_by_id(self):
        """One-shot: re-lay-out the active tab by original ID. See
        ParticleGrid.apply_id_order -- this is a single snapshot, not an
        ongoing mode, and never touches which particles exist."""
        grid = self._class_grids.get(self._active_classid())
        if grid is None:
            return
        grid.apply_id_order()

    def _sort_current_by_score(self):
        """One-shot: re-lay-out the active tab by last-computed
        Mahalanobis score. See ParticleGrid.apply_score_order -- reads
        whatever's cached in self._uid_scores (kept current by every
        sync_particles call), doesn't compute anything itself."""
        self.sort_active_tab_by_score()

    def sort_active_tab_by_score(self):
        """Same one-shot action as the "Sort by Rank" button, callable
        directly -- used by TomoBoxer right after a Mahalanobis
        computation finishes, so the freshly-computed order is visible
        immediately without an extra manual click. A no-op if the
        computation's target set isn't the tab currently on screen (the
        caller is expected to have already switched to it -- see
        activate_class_tab)."""
        grid = self._class_grids.get(self._active_classid())
        if grid is None:
            return
        grid.apply_score_order(self._uid_scores)

    _LOCKED_VISIBILITY_TIP = 'Turn off "Show Active Set Only" to change visibility.'
    _NORMAL_VISIBILITY_TIP = "Show/hide this set's particles in the tomogram views."

    def _make_visibility_checkbox(self, classid):
        cb = QtWidgets.QCheckBox()
        target = self._target_or_none()
        locked = target is not None and target.is_solo_active()
        cb.setEnabled(not locked)
        cb.setToolTip(self._LOCKED_VISIBILITY_TIP if locked else self._NORMAL_VISIBILITY_TIP)
        cb.setChecked(target is not None and classid in target.sets_visible)
        cb.toggled.connect(lambda checked, cid=classid: self._set_visibility(cid, checked))
        return cb

    def _set_visibility(self, classid, checked):
        target = self._target_or_none()
        if target is None:
            return
        if checked:
            target.show_set(f"{classid}::")
        else:
            target.hide_set(f"{classid}::")

    def refresh_visibility_checkboxes(self):
        """Resync every checkbox's checked state from the model -- needed
        whenever sets_visible changes by some means other than a checkbox
        click itself (Show Active Set Only turning on/off, Show All
        Channels)."""
        target = self._target_or_none()
        if target is None:
            return
        for cid, cb in self._visibility_checkboxes.items():
            cb.blockSignals(True)
            cb.setChecked(cid in target.sets_visible)
            cb.blockSignals(False)
        for cid, win in self._detached.items():
            cb = win.visibility_checkbox
            cb.blockSignals(True)
            cb.setChecked(cid in target.sets_visible)
            cb.blockSignals(False)

    def set_visibility_controls_locked(self, locked):
        """Gray out every visibility checkbox while Show Active Set Only is
        active -- clicking one wouldn't do anything useful right now since
        it'd just be overridden on the next tab switch, and the tooltip
        explains why on hover (Qt shows tooltips on disabled widgets)."""
        tip = self._LOCKED_VISIBILITY_TIP if locked else self._NORMAL_VISIBILITY_TIP
        for cb in self._visibility_checkboxes.values():
            cb.setEnabled(not locked)
            cb.setToolTip(tip)
        for win in self._detached.values():
            win.visibility_checkbox.setEnabled(not locked)
            win.visibility_checkbox.setToolTip(tip)

    # -- per-tab color swatch -------------------------------------------
    def _make_color_swatch(self, classid):
        btn = QtWidgets.QToolButton()
        btn.setFixedSize(14, 14)
        btn.setAutoRaise(True)
        btn.setCursor(QtCore.Qt.PointingHandCursor)
        btn.setToolTip("This set's color -- click to change it.")
        btn.clicked.connect(lambda checked=False, cid=classid, b=btn: self._open_color_menu(cid, b))
        self._refresh_color_swatch(classid, btn)
        return btn

    def _refresh_color_swatch(self, classid, btn=None):
        btn = btn if btn is not None else self._color_swatches.get(classid)
        if btn is None:
            return
        color = categories.class_color(classid)
        btn.setStyleSheet(
            f"QToolButton {{ background-color: {color.name()}; "
            "border: 1px solid palette(mid); border-radius: 3px; }")

    def _refresh_all_color_swatches(self):
        for classid in self._color_swatches:
            self._refresh_color_swatch(classid)
        # A detached window's swatch isn't in _color_swatches (it lives in
        # its own top-level window, not the tab bar) -- refresh it too, so
        # e.g. Toggle Colors doesn't leave a detached class's swatch stale.
        for classid, win in self._detached.items():
            self._refresh_color_swatch(classid, win.color_swatch)

    @staticmethod
    def _revert_label(classid):
        if classid == categories.RECYCLE_BIN_ID:
            return "Revert to Red"
        if classid == categories.CONFIRMED_ID:
            return "Revert to Green"
        return "Reset to Automatic"

    def _build_color_menu(self, classid):
        menu = QtWidgets.QMenu(self)
        menu.addAction("Choose Color...", lambda: self._choose_color(classid))
        menu.addAction(self._revert_label(classid), lambda: self._reset_color(classid))
        return menu

    def _open_color_menu(self, classid, anchor):
        menu = self._build_color_menu(classid)
        menu.exec(anchor.mapToGlobal(anchor.rect().bottomLeft()))

    def _choose_color(self, classid):
        target = self._target_or_none()
        if target is None:
            return
        color = QtWidgets.QColorDialog.getColor(categories.class_color(classid), self, "Choose Color")
        if color.isValid():
            # Triggers rebuild_tabs() downstream (via update_all), which
            # already refreshes every swatch -- no separate call needed here.
            target.set_class_color(classid, color)

    def _reset_color(self, classid):
        target = self._target_or_none()
        if target is None:
            return
        target.reset_class_color(classid)

    def _connect_grid(self, grid):
        grid.mx_image_selected.connect(self.mx_image_selected)
        grid.selection_changed.connect(self._commit_selection)

    # -- selection (single source of truth, shared by every grid) -------
    # Each ParticleGrid used to own its selection independently, so which
    # uids looked "selected" depended on which grid instance happened to
    # handle the gesture -- Ctrl+click in the tomogram window always wrote
    # into "All"'s own grid specifically, invisible in whichever tab you
    # were actually viewing, and a particle moved to a different class
    # silently lost its highlight (the old grid dropped it with no
    # notification, the new one never knew to apply it). _selected_uids
    # here is the one place that's actually authoritative; every grid is
    # a "dumb" renderer of it via apply_selection().
    def selected_uids(self):
        return list(self._selected_uids)

    def toggle_uid(self, uid):
        """Toggle one uid in/out of the selection -- used by TomoBoxer for
        Ctrl+click in the tomogram views, so it affects the real selection
        regardless of which tab happens to be docked/visible right now."""
        sel = set(self._selected_uids)
        if uid in sel:
            sel.discard(uid)
        else:
            sel.add(uid)
        self._commit_selection(sel)

    def replace_selection(self, uids):
        """Replace the selection outright -- used by TomoBoxer's undo/redo
        restore-highlight and shape-based select tool in the tomogram
        views."""
        self._commit_selection(uids)

    def clear_selection(self):
        self._commit_selection([])

    def _commit_selection(self, uids):
        """The one place the selection actually changes, regardless of
        which grid (or the tomogram window) requested it: updates the
        shared set, pushes the result out to every grid this panel
        manages so all of them render it consistently, then re-emits
        selection_changed for TomoBoxer."""
        self._selected_uids = set(uids or [])
        for g in self._all_grids():
            g.apply_selection(self._selected_uids)
        self.selection_changed.emit(list(self._selected_uids))

    def _active_grid(self):
        w = self.tabs.currentWidget()
        return w if isinstance(w, ParticleGrid) else self.grid

    def force_refresh(self):
        """Force a full relayout of whatever's currently on screen, not
        just the reconciliation sync_particles normally does -- used by
        TomoBoxer.refresh_all (Edit > Refresh, F5) as an escape hatch if
        rendering ever visibly gets stuck. sync_particles (called right
        before this, as part of refresh_all's own update_all) already
        re-renders every pixmap; this additionally repositions every
        item, in case layout itself is what's stuck."""
        self._active_grid()._relayout_all()

    def active_uids_from(self, uid):
        """uid and everything visually after it in whichever grid is
        currently on screen -- see ParticleGrid.uids_from_slot. A click
        can only ever happen in the grid you're actually looking at, so
        that's always the right one to ask."""
        return self._active_grid().uids_from_slot(uid)

    def set_mouse_mode(self, mode):
        self.grid.set_mouse_mode(mode)
        for g in self._class_grids.values():
            g.set_mouse_mode(mode)

    def _all_grids(self):
        yield self.grid
        yield from self._class_grids.values()
        for win in self._detached.values():
            yield win.grid

    def set_levels(self, lo, hi):
        self._levels_state = (lo, hi)
        for g in self._all_grids():
            g.set_levels(lo, hi)

    def clear_levels(self):
        self._levels_state = None
        for g in self._all_grids():
            g.clear_levels()

    # -- tabs / classes -----------------------------------------------------
    def rebuild_tabs(self, sets: dict):
        """sets: {classid: name}. Adds/removes/renames class tabs to match.
        A brand new tab always lands at the end of the normal-sets group
        (just before "+"); an existing one's position is never touched
        here -- the user may have dragged it, and only _pin_fixed_tabs
        (fixed categories) and explicit drags (normal sets) ever move a
        tab once it exists. Detached windows are left alone (they aren't
        tabs), just their title is refreshed on rename."""
        wanted = dict(sets)
        creation_order = categories.category_order(wanted)

        for classid in list(self._class_grids.keys()):
            if classid not in wanted:
                grid = self._class_grids.pop(classid)
                self._class_names.pop(classid, None)
                self._pending_entries.pop(classid, None)
                self._visibility_checkboxes.pop(classid, None)
                self._color_swatches.pop(classid, None)
                idx = self.tabs.indexOf(grid)
                if idx >= 0:
                    # Switch away *before* removing it if it's the tab
                    # currently on screen -- otherwise Qt shifts the
                    # current tab onto whatever ends up in its place
                    # (possibly "+", which immediately fires an unwanted
                    # "create a new set" prompt as a side effect of
                    # deleting one). _delete_current_set already does this
                    # itself before calling delete_set, but doing it here
                    # too means anything that removes a class -- now or
                    # later -- can't reintroduce this by skipping that
                    # dance.
                    if self.tabs.currentWidget() is grid:
                        self.tabs.setCurrentWidget(self.grid)
                    self.tabs.removeTab(idx)

        for classid in list(self._detached.keys()):
            if classid not in wanted:
                win = self._detached.pop(classid)
                win.close()

        for classid in creation_order:
            name = wanted[classid]
            if classid in self._detached:
                win = self._detached[classid]
                win.setWindowTitle(name)
                self._class_names[classid] = name
                continue
            if classid in self._class_grids:
                if self._class_names.get(classid) != name:
                    self._class_names[classid] = name
                    # Normal sets get their number-prefixed label repainted
                    # by _renumber_tabs() below; fixed categories aren't
                    # numbered, so they need setting directly here.
                    if classid in categories.FIXED_CLASS_IDS:
                        idx = self.tabs.indexOf(self._class_grids[classid])
                        if idx >= 0:
                            self.tabs.setTabText(idx, name)
                continue
            grid = ParticleGrid()
            self._connect_grid(grid)
            if self._levels_state is not None:
                grid.set_levels(*self._levels_state)
            self._class_grids[classid] = grid
            self._class_names[classid] = name
            # Insert before "+" (which addTab would otherwise land after,
            # since it's always the last tab) -- this alone is what puts a
            # brand new set at the end of the normal-sets group.
            self.tabs.insertTab(self.tabs.indexOf(self._plus_placeholder), grid, name)
            idx = self.tabs.indexOf(grid)
            cb = self._make_visibility_checkbox(classid)
            self._visibility_checkboxes[classid] = cb
            self.tabs.tabBar().setTabButton(idx, QtWidgets.QTabBar.LeftSide, cb)
            swatch = self._make_color_swatch(classid)
            self._color_swatches[classid] = swatch
            self.tabs.tabBar().setTabButton(idx, QtWidgets.QTabBar.RightSide, swatch)

        self._pin_fixed_tabs()
        self._renumber_tabs()
        # A class's color can change for reasons this method doesn't see
        # directly (Toggle Colors' rotation, or a swatch on some other
        # already-existing tab) -- refreshing every swatch here, on every
        # call, keeps them always correct rather than tracking each
        # possible trigger individually. Cheap: just a handful of classes.
        self._refresh_all_color_swatches()
        self._refresh_compute_distance_controls()

    def _is_tab_locked(self, index):
        """"All", "+", and the fixed categories can't be dragged -- only
        normal sets can (see _ConstrainedTabBar)."""
        widget = self.tabs.widget(index)
        if widget is self.grid or widget is self._plus_placeholder:
            return True
        for cid, grid in self._class_grids.items():
            if grid is widget:
                return cid in categories.FIXED_CLASS_IDS
        return False

    def _tab_drag_bounds(self):
        """Pixel x-range (tab-bar-local) a normal tab may be dragged
        within -- right after "All" and whichever fixed categories exist,
        up to just before "+". Computed fresh at the start of each drag
        (see _ConstrainedTabBar.mousePressEvent), not cached, so it
        reflects the pinned tabs' actual current widths."""
        bar = self.tabs.tabBar()
        fixed_count = 1 + sum(1 for cid in categories.FIXED_CLASS_IDS if cid in self._class_grids)
        lo = bar.tabRect(fixed_count - 1).right() + 1
        plus_idx = self.tabs.indexOf(self._plus_placeholder)
        hi = bar.tabRect(plus_idx).left() - 1
        return (lo, hi)

    def _pin_fixed_tabs(self):
        """Keep "All" at position 0, Recycle Bin/Confirmed (whichever
        exist) pinned right after it in that order, and "+" pinned last.
        Normal sets are deliberately left alone here -- their relative
        order is up to the user (drag to rearrange) now, not recomputed
        on every call. Needed both after rebuild_tabs adds/removes tabs
        and after reattaching a detached tab, since insertTab/addTab
        there just appends.

        Reentrancy-guarded: moveTab() below emits tabMoved synchronously,
        which _on_tab_moved handles by calling straight back into this
        method -- without the guard, that nested call would repeat the
        very same corrections the outer call hasn't finished making yet,
        recursing arbitrarily deep instead of the two or three moveTab
        calls this actually ever needs."""
        if self._pinning_fixed_tabs:
            return
        self._pinning_fixed_tabs = True
        try:
            bar = self.tabs.tabBar()
            all_idx = self.tabs.indexOf(self.grid)
            if all_idx >= 0 and all_idx != 0:
                bar.moveTab(all_idx, 0)
            pos = 1
            for cid in categories.FIXED_CLASS_IDS:
                grid = self._class_grids.get(cid)
                if grid is None:
                    continue
                cur = self.tabs.indexOf(grid)
                if cur >= 0 and cur != pos:
                    bar.moveTab(cur, pos)
                pos += 1
            plus_idx = self.tabs.indexOf(self._plus_placeholder)
            last = self.tabs.count() - 1
            if plus_idx >= 0 and plus_idx != last:
                bar.moveTab(plus_idx, last)
        finally:
            self._pinning_fixed_tabs = False

    def _on_tab_moved(self, from_index, to_index):
        """Re-pin "All"/Recycle Bin/Confirmed/"+" after every single step
        of a drag, not just check whether the dragged tab's own landing
        spot looks in-range: a mouseMoveEvent clamp (_ConstrainedTabBar)
        stops most out-of-bounds drags before they happen, but Qt's own
        per-pixel drag-reorder can still resolve one large/fast mouse move
        as several internal swaps at once, and a narrow "is to_index
        within [fixed_count, plus_index-1]" check can pass even though a
        normal tab ended up wedged *between* two fixed categories (e.g.
        Recycle Bin and Confirmed swapped past each other, each only
        partially). _pin_fixed_tabs() moving a fixed tab back to its own
        slot naturally shoves whatever normal tab was squatting there into
        the first free normal slot instead (moveTab's own semantics), so
        unconditionally re-pinning after every reorder step self-heals any
        scramble, not just the single-boundary-overshoot case."""
        self._pin_fixed_tabs()
        self._renumber_tabs()

    def _current_class_order(self):
        """Fixed categories, then normal sets, in whatever order their
        tabs actually sit in right now -- the live source of truth for
        display order, since drag-reordering only ever moves the real
        QTabWidget's tabs and nothing else needs to track it separately."""
        order = []
        for i in range(self.tabs.count()):
            widget = self.tabs.widget(i)
            if widget is self.grid or widget is self._plus_placeholder:
                continue
            for cid, grid in self._class_grids.items():
                if grid is widget:
                    order.append(cid)
                    break
        return order

    def _all_visible_class_order(self):
        """Every class that can have particles shown in the "All" tab --
        fixed categories and normal sets in their current docked tab
        order, plus any *detached* class (fixed or normal) appended at
        the end. Unlike _current_class_order() alone, this doesn't drop a
        detached class: its tab isn't in the bar to read an order from,
        but its particles are still just as visible in the tomogram
        views and still need their own group in "All" -- omitting it here
        left its uids sitting in this grid's _items with no _slot_of_uid
        entry at all (sync_particles_grouped only assigns one to classids
        it's told about), crashing the very next relayout."""
        docked = self._current_class_order()
        seen = set(docked)
        detached_only = [cid for cid in self._detached if cid not in seen]
        detached_only.sort(key=lambda cid: (cid not in categories.FIXED_CLASS_IDS, self._class_names.get(cid, "")))
        return docked + detached_only

    def normal_class_order(self):
        """Normal (non-fixed) classids in current tab order, skipping any
        that are currently detached -- a detached set has no tab position
        to switch to, so number-key selection skips it entirely (see
        TomoBoxer.select_set_by_number)."""
        return [cid for cid in self._current_class_order() if cid not in categories.FIXED_CLASS_IDS]

    def _renumber_tabs(self):
        """Label each normal set's tab with its current position (1-9) --
        exactly what the number-key shortcuts switch to, so the mapping
        is always visible and always matches whatever's on screen,
        dragged or not."""
        for position, cid in enumerate(self.normal_class_order(), start=1):
            grid = self._class_grids.get(cid)
            idx = self.tabs.indexOf(grid) if grid is not None else -1
            if idx >= 0:
                name = self._class_names.get(cid, "")
                label = f"{position}  {name}" if position <= 9 else name
                self.tabs.setTabText(idx, label)

    def activate_class_tab(self, classid):
        """Switch to a class's own tab -- used by number-key shortcuts so
        the gallery stays in sync with whichever class just became active
        for picking, the same as clicking the tab directly would."""
        grid = self._class_grids.get(classid)
        if grid is not None:
            self.tabs.setCurrentWidget(grid)

    def _detach_current_tab(self):
        classid = self._active_classid()
        if classid is None:
            return  # "All"/"+" can't be detached
        grid = self._class_grids.pop(classid)
        name = self._class_names.get(classid, str(classid))
        # Popped (not reused) here: removeTab below deletes a tab's corner
        # widget as a side effect of removing the tab, so this Python
        # reference could already be pointing at a destroyed C++ object by
        # the time DetachedGridWindow might otherwise try to reuse it. A
        # fresh one is made just below instead (same as _reattach does).
        self._visibility_checkboxes.pop(classid, None)
        self._color_swatches.pop(classid, None)
        # Switch away from the tab being removed *first* -- otherwise, when
        # it's the last real class tab, Qt shifts the current tab onto
        # whatever ends up in its place, which is the "+" placeholder,
        # immediately (and confusingly) firing the "create a new set"
        # prompt as a side effect of detaching one (same issue as deleting
        # the last real set -- see _delete_current_set).
        self.tabs.setCurrentWidget(self.grid)
        idx = self.tabs.indexOf(grid)
        if idx >= 0:
            self.tabs.removeTab(idx)
        self._renumber_tabs()  # close the numbering gap left behind
        win = DetachedGridWindow(
            classid, grid, name,
            self._make_visibility_checkbox(classid), self._make_color_swatch(classid))
        win.closed.connect(self._reattach)
        self._detached[classid] = win
        win.show()

    def _reattach(self, classid):
        win = self._detached.pop(classid, None)
        if win is None:
            return
        grid = win.grid
        name = self._class_names.get(classid, str(classid))
        self._class_grids[classid] = grid
        grid.setVisible(True)
        self.tabs.addTab(grid, name)
        idx = self.tabs.indexOf(grid)
        cb = self._make_visibility_checkbox(classid)
        self._visibility_checkboxes[classid] = cb
        self.tabs.tabBar().setTabButton(idx, QtWidgets.QTabBar.LeftSide, cb)
        swatch = self._make_color_swatch(classid)
        self._color_swatches[classid] = swatch
        self.tabs.tabBar().setTabButton(idx, QtWidgets.QTabBar.RightSide, swatch)
        self._pin_fixed_tabs()  # a reattached tab lands at the end of the normal group
        self._renumber_tabs()
        self.tabs.setCurrentWidget(grid)

    # -- particle sync --------------------------------------------------------
    def sync_particles(self, entries, scores=None):
        """entries: iterable of (uid, img, classid). scores: {uid:
        (score, has_embedding)} as built by TomoBoxer.update_boximgs --
        cached (self._uid_scores) for use by any tab currently showing
        the "Sort by Rank" order, and by the toggle itself when it's
        switched on later without a fresh sync happening."""
        entries = list(entries)
        self._uid_scores = scores if scores is not None else {}
        current_widget = self.tabs.currentWidget()
        if current_widget is self.grid:
            self.grid.sync_particles_grouped(entries, self._all_visible_class_order())
            self.grid.apply_selection(self._selected_uids)
            self._pending_all_entries = None
        else:
            # "All" isn't the tab on screen right now -- defer its full
            # grouped rebuild (sync_particles_grouped fully recomputes every
            # thumbnail's contrast and every item's layout on every call, by
            # design -- see its own docstring) until the user actually
            # switches to it, exactly like the per-class tabs below already
            # do. At tens of thousands of particles this was the single
            # biggest per-click cost regardless of which tab was actually
            # being looked at.
            self._pending_all_entries = entries
        by_class = {}
        for uid, img, cid in entries:
            by_class.setdefault(cid, []).append((uid, img))
        for classid, grid in self._class_grids.items():
            class_entries = by_class.get(classid, [])
            if grid is current_widget:
                grid.sync_particles(class_entries)
                grid.apply_selection(self._selected_uids)
                self._pending_entries.pop(classid, None)
            else:
                # Not the tab on screen right now -- its thumbnails aren't
                # visible either way, so defer the (potentially expensive,
                # many-thumbnail) sync until it's actually switched to (see
                # _on_tab_changed) instead of eagerly maintaining every
                # class's full grid on every single particle change,
                # including ones nobody is currently looking at. This is
                # what made opening a tomogram with many pre-existing
                # particles across several classes slow: it was building
                # every tab's entire thumbnail grid up front, most of which
                # wouldn't be looked at before the next edit rebuilt them
                # all again anyway.
                self._pending_entries[classid] = class_entries
        for classid, win in self._detached.items():
            # Detached windows are independent, possibly-visible-right-now
            # top-level windows, not tab pages Qt can tell us are hidden --
            # always keep them eagerly in sync.
            win.grid.sync_particles(by_class.get(classid, []))
            win.grid.apply_selection(self._selected_uids)

    def closeEvent(self, event):
        target = self._target() if self._target is not None else None
        if target is not None and not getattr(target, "_closing", False):
            event.ignore()
            target.request_close_all()
            return
        for win in list(self._detached.values()):
            win.close()
        event.accept()

