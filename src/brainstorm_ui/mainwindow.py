from PyQt6.QtWidgets import QMainWindow


class AppMainWindow(QMainWindow):
    def __init__(self) -> None:
        QMainWindow.__init__(self)
        
        self.resize(600, 400)


__all__ = ['AppMainWindow']
