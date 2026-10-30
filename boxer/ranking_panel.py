"""The "Rank by Embedding Distance" controls of the particle list: which
set is ranked (the tab on screen), close to which set, far from which, in
which embedding space, with or without leave-one-out, and -- in a
multi-tomogram session -- which tomograms' sets are pooled as the
references. The window does the ranking (TomoBoxer.run_compute_distance);
this panel only gathers the choice and keeps itself enabled or not."""
from __future__ import annotations

import weakref

from PySide6 import QtCore, QtGui, QtWidgets

from . import categories
from .widgets import make_action

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


class RankingPanel(QtWidgets.QWidget):
    def __init__(self, target, active_classid, parent=None):
        """`active_classid` is the particle list's own: the class id of the
        tab on screen, or None on a view tab."""
        super().__init__(parent)
        self._target = weakref.ref(target) if target is not None else None
        self._active_classid = active_classid
        # Deliberately the loudest thing in this window: computing/sorting
        # by Mahalanobis distance used to live as a File menu item (then a
        # popup dialog triggered from a button), which made it easy to miss
        # or fiddly to reach. Now it's its own group, inline, right here --
        # the reference set/embedding space/leave-one-out controls plus the
        # big green "go" button all live together with nothing modal in
        # between, so the whole feature reads and edits in place.
        group_layout = QtWidgets.QVBoxLayout(self)
        group_layout.setContentsMargins(0, 0, 0, 0)
        group_layout.setSpacing(3)

        # One row for everything: which set gets ranked (the current tab,
        # kept live-updated by update_state as you switch
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
            self.update_state)
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
            self.update_state)
        controls_row.addWidget(self.compute_distance_negative_combo)
        # Populated from whatever embedding spaces are actually attached
        # (see refresh) rather than hardcoded --
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

        # Multi-tomogram sessions only: which tomograms' particles make up
        # the "close to" / "far from" references. The same set name is
        # looked up in each -- Confirmed pools every chosen tomogram's
        # Confirmed, Recycle Bin every Recycle Bin. "This tomogram" is the
        # default, and is exactly what ranking did before sessions existed.
        self.compute_distance_scope_row = QtWidgets.QWidget()
        scope_layout = QtWidgets.QHBoxLayout(self.compute_distance_scope_row)
        scope_layout.setContentsMargins(0, 0, 0, 0)
        scope_layout.addWidget(QtWidgets.QLabel("References from:"))
        self.compute_distance_scope_combo = QtWidgets.QComboBox()
        self.compute_distance_scope_combo.addItem("this tomogram", "this")
        self.compute_distance_scope_combo.addItem("all tomograms in the session", "all")
        self.compute_distance_scope_combo.addItem("chosen tomograms...", "choose")
        self.compute_distance_scope_combo.setToolTip(
            "Which tomograms' particles form the \"close to\" and \"far from\" references. "
            "Each chosen tomogram contributes its own set of the same name, read from its "
            "working directory, pooled with this tomogram's.")
        self.compute_distance_scope_combo.activated.connect(self._on_scope_activated)
        scope_layout.addWidget(self.compute_distance_scope_combo)
        scope_layout.addStretch(1)
        group_layout.addWidget(self.compute_distance_scope_row)
        in_session = target is not None and getattr(target, "session", None) is not None
        self.compute_distance_scope_row.setVisible(in_session)
        if in_session:
            self._restore_scope(target.session_controller)

        self.compute_distance_button = QtWidgets.QPushButton("▶  Rank by Embedding Distance")
        # Saved so update_state can restore it after
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
        # kept in sync with this panel's own enabled state by
        # update_state -- there is deliberately no
        # separate toolbar button for it, since the big green one already
        # serves that role.
        self.rank_action = make_action(self, "Rank by Embedding Distance",
                                       self.compute_distance_button.toolTip(), self._run_compute_distance)

        self.setEnabled(False)  # no embeddings loaded yet
        # The particle list adds this panel to its layout right below the
        # tab grid (ParticleListPanel.__init__), so it sits next to Sort by Original ID/Sort by Rank
        # rather than above the tabs entirely: all three act on "whatever
        # tab you're currently looking at," so they read as one group.

    def refresh(self):
        """Keeps the reference-set combos in sync with whatever sets exist
        (called from rebuild_tabs, so on every set add/rename/delete),
        preserving the current selection where possible, and refreshes the
        UMAP item's presence/availability. Actual enabled-state (whether
        embeddings are loaded, and whether leave-one-out applies) is
        update_state's job, called at the end of this."""
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
        for name in target.ordered_space_names():
            space_combo.addItem(target.embedding_spaces[name]["label"], name)
        idx = space_combo.findData(previous_space) if previous_space else -1
        space_combo.setCurrentIndex(idx if idx >= 0 else 0)
        space_combo.blockSignals(False)

        self.update_state()

    def update_state(self):
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
        self.setEnabled(group_enabled)
        self.setToolTip(
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
            negative_classid=self.compute_distance_negative_combo.currentData(),
            reference_tomograms=self.reference_tomograms())

    # -- reference scope (multi-tomogram sessions) ------------------------------
    def reference_tomograms(self):
        """None for "this tomogram"; otherwise every tomogram whose
        references are pooled, this one included."""
        target = self._target_or_none()
        if target is None or getattr(target, "session", None) is None:
            return None
        scope, chosen = target.session_controller.reference_scope
        if scope == "all":
            return list(target.session.names)
        if scope == "choose":
            return [target.session_name] + [t for t in chosen if t != target.session_name]
        return None

    def _restore_scope(self, controller):
        """The scope is a session-wide choice, kept on the controller, so it
        survives switching tomograms rather than resetting with each one."""
        scope, chosen = controller.reference_scope
        combo = self.compute_distance_scope_combo
        combo.setCurrentIndex(max(combo.findData(scope), 0))
        self._label_chosen(chosen)

    def _label_chosen(self, chosen):
        combo = self.compute_distance_scope_combo
        idx = combo.findData("choose")
        combo.setItemText(idx, f"chosen tomograms ({len(chosen)})..." if chosen
                          else "chosen tomograms...")

    def _on_scope_activated(self, index):
        target = self._target_or_none()
        if target is None or getattr(target, "session", None) is None:
            return
        controller = target.session_controller
        scope = self.compute_distance_scope_combo.itemData(index)
        previous_scope, chosen = controller.reference_scope
        if scope == "choose":
            # here, not at module level: session_window imports main_window, which imports us
            from .session_window import ChooseTomogramsDialog
            dlg = ChooseTomogramsDialog(
                [t for t in target.session.names if t != target.session_name], chosen, self)
            if dlg.exec() == QtWidgets.QDialog.Accepted and dlg.chosen():
                chosen = dlg.chosen()
            elif not chosen:
                # nothing chosen, ever: fall back to what was selected before
                self.compute_distance_scope_combo.setCurrentIndex(
                    max(self.compute_distance_scope_combo.findData(previous_scope), 0))
                return
        controller.reference_scope = (scope, chosen)
        self._label_chosen(chosen)


    def _target_or_none(self):
        return self._target() if self._target is not None else None
