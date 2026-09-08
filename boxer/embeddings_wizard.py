"""File > Import Particle List with Embeddings... -- pick an embedding
method, then its files, and hand them to
TomoBoxer.import_particle_list_with_embeddings, which does the actual
validation and import.

This dialog deliberately does no validation itself -- it's just the
method choice and the file pickers. All of the "do these actually match"
checks (particle counts, and the cross-checks against whatever
coordinates the files record themselves) live in one place either the GUI
or the command-line flags can reach, so the two paths can never drift
into checking different things.
"""
from __future__ import annotations

from PySide6 import QtWidgets

from . import embeddings_io

# Per method: label, embeddings-file filter, umap filter, and whether a
# separate particle list is required (CryoSIAM's .h5 carries no
# coordinates; TomoTwin's .temb carries its own).
_METHODS = {
    "cryosiam": {
        "label": "CryoSIAM",
        "embeddings": "Embeddings (*.h5 *.hdf5);;All files (*)",
        "umap": "UMAP CSV (*.csv);;All files (*)",
        "list_required": True,
        "note": ("CryoSIAM's .h5 carries no coordinates, so a particle list is "
                  "required — the two are matched by row order. If a UMAP CSV is "
                  "also given, its own recorded coordinates (centroid-0/1/2) are "
                  "cross-checked against the particle list."),
    },
    "tomotwin": {
        "label": "TomoTwin",
        "embeddings": "TomoTwin embeddings (*.temb);;All files (*)",
        "umap": "TomoTwin UMAP (*.tumap);;All files (*)",
        "list_required": False,
        "note": ("TomoTwin's .temb carries its own X/Y/Z, so a particle list is "
                  "optional — give one only if you want it cross-checked against "
                  "the embeddings. Coordinates in the .tumap are checked the same "
                  "way."),
    },
}


class EmbeddingsImportDialog(QtWidgets.QDialog):
    """attach=False imports a particle list as a new set with embeddings
    attached; attach=True adds another method's embeddings to particles
    that are already imported, so one set can be ranked by several
    methods and compared. The only difference is whether a particle list
    is involved at all -- in attach mode the rows are already fixed by
    the first import, so there's nothing to pick."""

    def __init__(self, parent=None, attach=False):
        super().__init__(parent)
        self._attach = attach
        self.setWindowTitle(
            "Attach Embeddings to Current Particles" if attach
            else "Import Particle List with Embeddings")
        self.setMinimumWidth(460)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(
            "Add another method's embeddings to the particles already imported, "
            "so the same picks can be ranked by each and compared. The files must "
            "describe those same particles, in the same row order."
            if attach else
            "Import a particle list as a new set, with its embeddings attached "
            "for ranking. Files are matched by row order."))

        form = QtWidgets.QFormLayout()

        self.method_combo = QtWidgets.QComboBox()
        for key in embeddings_io.METHODS:
            self.method_combo.addItem(_METHODS[key]["label"], key)
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        form.addRow("Method:", self.method_combo)

        self.coord_edit = QtWidgets.QLineEdit()
        self.coord_row = self._file_row(self.coord_edit, "Text files (*.txt);;All files (*)")
        self.coord_label = QtWidgets.QLabel("Particle list:")
        if not attach:
            form.addRow(self.coord_label, self.coord_row)

        self.emb_edit = QtWidgets.QLineEdit()
        self.emb_row = self._file_row(self.emb_edit, _METHODS["cryosiam"]["embeddings"])
        self.emb_label = QtWidgets.QLabel("Embeddings:")
        form.addRow(self.emb_label, self.emb_row)

        self.umap_edit = QtWidgets.QLineEdit()
        self.umap_row = self._file_row(self.umap_edit, _METHODS["cryosiam"]["umap"])
        form.addRow("UMAP (optional):", self.umap_row)
        layout.addLayout(form)

        self.note = QtWidgets.QLabel()
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color: palette(mid);")
        layout.addWidget(self.note)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._on_method_changed()

    def _current_method(self):
        return self.method_combo.currentData()

    def _on_method_changed(self):
        spec = _METHODS[self._current_method()]
        self.note.setText(
            "Rows are matched to the particles already imported, and the file's own "
            "coordinates (where it records them) are checked against theirs."
            if self._attach else spec["note"])
        self.coord_label.setText(
            "Particle list:" if spec["list_required"] else "Particle list (optional):")
        # Re-point each browse button at this method's own file filter.
        self.emb_row.filter = spec["embeddings"]
        self.umap_row.filter = spec["umap"]

    def _file_row(self, edit, file_filter):
        row = QtWidgets.QWidget()
        row.filter = file_filter
        h = QtWidgets.QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit)
        browse = QtWidgets.QPushButton("Browse...")
        browse.clicked.connect(lambda _=False, e=edit, r=row: self._browse(e, r.filter))
        h.addWidget(browse)
        return row

    def _browse(self, edit, file_filter):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select file", "", file_filter)
        if path:
            edit.setText(path)

    def values(self):
        """(method, coord_path, embeddings_path, umap_path) -- each path
        None if left blank. coord_path is always None in attach mode,
        where the rows are already fixed by the original import."""
        return (
            self._current_method(),
            None if self._attach else (self.coord_edit.text().strip() or None),
            self.emb_edit.text().strip() or None,
            self.umap_edit.text().strip() or None,
        )

    def list_required(self):
        return (not self._attach) and _METHODS[self._current_method()]["list_required"]
