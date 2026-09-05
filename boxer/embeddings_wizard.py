"""File > Import Particle List with Embeddings... -- lets the user pick a
coordinate/particle-list text file plus its matched embeddings (an HDF5
file, and optionally a companion UMAP-reduced CSV), then hands the three
paths to TomoBoxer.import_particle_list_with_embeddings, which does the
actual validation and import.

This dialog deliberately does no validation itself -- it's just the file
pickers. All of the "do these actually match" checks (particle counts,
and the spatial cross-check against the UMAP file's own recorded
coordinates) live in one place either the GUI or the --embedded_particle_list
CLI flag can reach, so the two paths can never drift into checking
different things.
"""
from __future__ import annotations

from PySide6 import QtWidgets


class EmbeddingsImportDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Particle List with Embeddings")
        self.setMinimumWidth(420)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(
            "Import a plain-text \"x y z\" particle list as a new set, "
            "matched to its embeddings by line/row order."))

        form = QtWidgets.QFormLayout()
        self.coord_edit = QtWidgets.QLineEdit()
        form.addRow("Particle list:", self._file_row(self.coord_edit, "Text files (*.txt);;All files (*)"))

        self.h5_edit = QtWidgets.QLineEdit()
        form.addRow("Embeddings (.h5):", self._file_row(self.h5_edit, "HDF5 files (*.h5 *.hdf5);;All files (*)"))

        self.umap_edit = QtWidgets.QLineEdit()
        form.addRow("UMAP CSV (optional):", self._file_row(self.umap_edit, "CSV files (*.csv);;All files (*)"))
        layout.addLayout(form)

        note = QtWidgets.QLabel(
            "The particle list and the .h5 file must have exactly the same number "
            "of particles, in the same order. If a UMAP CSV is also given, its own "
            "recorded coordinates (centroid-0/1/2) are cross-checked against the "
            "particle list's -- a mismatch on either check refuses the import rather "
            "than silently attaching the wrong embedding to a particle.")
        note.setWordWrap(True)
        note.setStyleSheet("color: palette(mid);")
        layout.addWidget(note)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _file_row(self, edit, file_filter):
        row = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit)
        browse = QtWidgets.QPushButton("Browse...")
        browse.clicked.connect(lambda: self._browse(edit, file_filter))
        h.addWidget(browse)
        return row

    def _browse(self, edit, file_filter):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select file", "", file_filter)
        if path:
            edit.setText(path)

    def values(self):
        """(coord_path, h5_path, umap_path) -- each None if left blank."""
        return (
            self.coord_edit.text().strip() or None,
            self.h5_edit.text().strip() or None,
            self.umap_edit.text().strip() or None,
        )
