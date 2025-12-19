from __future__ import annotations

import sys
from PySide6.QtWidgets import QApplication, QMainWindow

from qt_app.ttt_widget import TicTacToeWidget


def main():
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setWindowTitle("Крестики-нолики")
    win.setCentralWidget(TicTacToeWidget())
    win.resize(1100, 600)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
