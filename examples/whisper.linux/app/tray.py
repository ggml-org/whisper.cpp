"""System tray icon and menu for whisper.linux."""

import subprocess
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from .config import (
    Config, AppState, AVAILABLE_MODELS, DEFAULT_VOICE_COMMANDS, _REPO_ROOT,
    _list_models, _list_audio_devices, log,
)

if TYPE_CHECKING:
    from .app import WhisperLinuxApp


def _create_icon(color, size=64):
    """Create a microphone tray icon programmatically via QPainter."""
    from PyQt5.QtCore import Qt, QRect, QPoint
    from PyQt5.QtGui import QPixmap, QPainter, QColor, QBrush, QPen, QIcon

    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    p = QPainter(pixmap)
    p.setRenderHint(QPainter.Antialiasing)

    c = QColor(color)
    p.setBrush(QBrush(c))
    p.setPen(QPen(c.darker(130), 2))

    # Mic body (rounded rect)
    bw, bh = size // 3, size // 2
    bx = (size - bw) // 2
    by = size // 8
    p.drawRoundedRect(bx, by, bw, bh, bw // 3, bw // 3)

    # Arc below mic
    p.setBrush(Qt.NoBrush)
    p.setPen(QPen(c, max(2, size // 16)))
    arc_w = int(bw * 1.6)
    arc_h = size // 4
    arc_x = (size - arc_w) // 2
    arc_y = by + bh - arc_h // 2
    p.drawArc(arc_x, arc_y, arc_w, arc_h, 0, -180 * 16)

    # Stem
    mid_x = size // 2
    stem_top = arc_y + arc_h
    stem_bot = stem_top + size // 6
    p.drawLine(mid_x, stem_top, mid_x, stem_bot)

    # Base
    base_w = size // 3
    p.drawLine(mid_x - base_w // 2, stem_bot, mid_x + base_w // 2, stem_bot)

    p.end()
    return QIcon(pixmap)


class _CallHelper:
    """Helper to marshal calls from background threads to Qt main thread."""

    def __init__(self):
        from PyQt5.QtCore import QObject, pyqtSignal

        class _Obj(QObject):
            sig = pyqtSignal(object)

        self._obj = _Obj()
        self._obj.sig.connect(self._run)

    @staticmethod
    def _run(func):
        func()

    def call(self, func):
        self._obj.sig.emit(func)


class TrayIcon:
    """System tray icon with right-click menu."""

    COLORS = {
        AppState.IDLE: "#888888",
        AppState.RECORDING: "#e53935",
        AppState.PROCESSING: "#ffb300",
        AppState.LISTENING: "#2196F3",
        AppState.DICTATING: "#4CAF50",
    }

    # Minimum expected sizes (bytes) for downloaded models
    _MODEL_MIN_SIZES = {
        "tiny": 70_000_000,
        "base": 130_000_000,
        "small": 450_000_000,
        "medium": 1_400_000_000,
        "large-v1": 2_900_000_000,
        "large-v2": 2_900_000_000,
        "large-v3": 2_900_000_000,
        "large-v3-turbo": 1_500_000_000,
    }

    def __init__(self, app_ref: "WhisperLinuxApp"):
        from PyQt5.QtWidgets import QSystemTrayIcon

        self._app_ref = app_ref
        self._icons = {state: _create_icon(color) for state, color in self.COLORS.items()}
        self._downloading = set()
        self._kept_actions = []
        self._call_helper = _CallHelper()

        self.tray = QSystemTrayIcon()
        self.tray.setIcon(self._icons[AppState.IDLE])
        self.tray.setToolTip("whisper.linux \u2014 Voice Typing")
        self.tray.activated.connect(self._on_activated)

        self._build_menu()
        self.tray.show()

    # -- Menu construction --

    def _build_menu(self):
        from PyQt5.QtWidgets import QMenu, QAction, QActionGroup

        self.menu = QMenu()

        self.action_toggle = QAction("Start Recording", self.menu)
        self.action_toggle.triggered.connect(lambda: self._app_ref.toggle())
        self.menu.addAction(self.action_toggle)

        self.menu.addSeparator()

        # Language
        lang_menu = self.menu.addMenu("Language")
        self._lang_group = QActionGroup(lang_menu)
        self._lang_group.setExclusive(True)
        for code, label in [("ru", "Russian"), ("en", "English"), ("auto", "Auto-detect")]:
            a = QAction(label, lang_menu, checkable=True)
            a.setData(code)
            if code == self._app_ref.config.language:
                a.setChecked(True)
            a.triggered.connect(self._on_language_changed)
            self._lang_group.addAction(a)
            lang_menu.addAction(a)

        # Model
        self._model_menu = self.menu.addMenu("Model")
        self._model_group = QActionGroup(self._model_menu)
        self._model_group.setExclusive(True)
        self._rebuild_model_menu()

        # Settings
        self._build_settings_menu()

        self.menu.addSeparator()

        quit_action = QAction("Quit", self.menu)
        quit_action.triggered.connect(self._app_ref.quit)
        self.menu.addAction(quit_action)

        self.tray.setContextMenu(self.menu)

    def _rebuild_model_menu(self):
        from PyQt5.QtWidgets import QAction

        menu = self._model_menu
        menu.clear()
        self._kept_actions.clear()
        for a in self._model_group.actions():
            self._model_group.removeAction(a)

        current_model = Path(self._app_ref.config.model).name if self._app_ref.config.model else ""
        downloaded = _list_models(self._app_ref.config.model_search_dirs)
        downloaded_names = {name for name, _ in downloaded}

        if not downloaded and self._app_ref.config.model:
            name = Path(self._app_ref.config.model).stem.replace("ggml-", "")
            downloaded = [(name, self._app_ref.config.model)]
            downloaded_names = {name}

        for name, path in downloaded:
            size_mb = Path(path).stat().st_size / (1024 * 1024)
            label = f"{name}  ({size_mb:.0f} MB)"
            a = QAction(label, menu, checkable=True)
            a.setData(path)
            if Path(path).name == current_model:
                a.setChecked(True)
            a.triggered.connect(self._on_model_changed)
            self._model_group.addAction(a)
            menu.addAction(a)

        available = [(n, s) for n, s in AVAILABLE_MODELS if n not in downloaded_names]
        if available and downloaded:
            menu.addSeparator()

        for name, size in available:
            if name in self._downloading:
                a = QAction(f"{name}  {size}  \u23f3", menu)
                a.setEnabled(False)
            else:
                a = QAction(f"{name}  {size}  \u2193 Download", menu)
                a.setData(name)
                a.triggered.connect(lambda checked, n=name: self._on_download_model(n))
            menu.addAction(a)
            self._kept_actions.append(a)

    def _rebuild_wake_model_menu(self):
        from PyQt5.QtWidgets import QAction

        menu = self._wake_model_menu
        menu.clear()
        for a in self._wake_model_group.actions():
            self._wake_model_group.removeAction(a)

        config = self._app_ref.config
        same_label = "Same as main model"
        if config.model and Path(config.model).is_file():
            main_mb = Path(config.model).stat().st_size / (1024 * 1024)
            same_label = f"Same as main model  ({main_mb:.0f} MB)"
        a = QAction(same_label, menu, checkable=True)
        a.setData("")
        if not config.wake_model:
            a.setChecked(True)
        a.triggered.connect(self._on_wake_model_changed)
        self._wake_model_group.addAction(a)
        menu.addAction(a)
        for name, path in _list_models(config.model_search_dirs):
            size_mb = Path(path).stat().st_size / (1024 * 1024)
            label = f"{name}  ({size_mb:.0f} MB)"
            a = QAction(label, menu, checkable=True)
            a.setData(path)
            if path == config.wake_model:
                a.setChecked(True)
            a.triggered.connect(self._on_wake_model_changed)
            self._wake_model_group.addAction(a)
            menu.addAction(a)

    def _build_settings_menu(self):
        from PyQt5.QtWidgets import QAction, QActionGroup

        settings_menu = self.menu.addMenu("Settings")
        config = self._app_ref.config

        # Input mode
        input_menu = settings_menu.addMenu("Input mode")
        self._input_mode_group = QActionGroup(input_menu)
        self._input_mode_group.setExclusive(True)
        for val, label in [("hotkey", "Hotkey (push-to-talk)"), ("listen", "Listening (wake word)")]:
            a = QAction(label, input_menu, checkable=True)
            a.setData(val)
            if val == config.input_mode:
                a.setChecked(True)
            a.triggered.connect(self._on_input_mode_changed)
            self._input_mode_group.addAction(a)
            input_menu.addAction(a)

        # Output mode
        output_menu = settings_menu.addMenu("Output mode")
        self._output_mode_group = QActionGroup(output_menu)
        self._output_mode_group.setExclusive(True)
        for val, label in [("batch", "Batch (all at once)"), ("stream", "Streaming (per segment)")]:
            a = QAction(label, output_menu, checkable=True)
            a.setData(val)
            if val == config.output_mode:
                a.setChecked(True)
            a.triggered.connect(self._on_output_mode_changed)
            self._output_mode_group.addAction(a)
            output_menu.addAction(a)

        settings_menu.addSeparator()

        # Threads
        threads_menu = settings_menu.addMenu("Threads")
        self._threads_group = QActionGroup(threads_menu)
        self._threads_group.setExclusive(True)
        for n in [1, 2, 4, 8, 16]:
            a = QAction(str(n), threads_menu, checkable=True)
            a.setData(n)
            if n == config.threads:
                a.setChecked(True)
            a.triggered.connect(self._on_threads_changed)
            self._threads_group.addAction(a)
            threads_menu.addAction(a)

        # GPU
        gpu_menu = settings_menu.addMenu("GPU")
        self._gpu_group = QActionGroup(gpu_menu)
        self._gpu_group.setExclusive(True)
        gpu_devices = getattr(config, '_gpu_devices', [])
        if gpu_devices:
            for name, idx in gpu_devices:
                label = f"{idx}: {name}"
                a = QAction(label, gpu_menu, checkable=True)
                a.setData(idx)
                if idx == config.gpu_device:
                    a.setChecked(True)
                a.triggered.connect(self._on_gpu_changed)
                self._gpu_group.addAction(a)
                gpu_menu.addAction(a)
        else:
            a = QAction(f"Device {config.gpu_device}", gpu_menu, checkable=True)
            a.setData(config.gpu_device)
            a.setChecked(True)
            self._gpu_group.addAction(a)
            gpu_menu.addAction(a)

        # Audio
        audio_menu = settings_menu.addMenu("Audio")
        self._audio_group = QActionGroup(audio_menu)
        self._audio_group.setExclusive(True)
        audio_devices = _list_audio_devices()
        for label, device_id in audio_devices:
            a = QAction(label, audio_menu, checkable=True)
            a.setData(device_id)
            if device_id == config.audio_device:
                a.setChecked(True)
            a.triggered.connect(self._on_audio_changed)
            self._audio_group.addAction(a)
            audio_menu.addAction(a)

        # Paste mode
        paste_menu = settings_menu.addMenu("Paste mode")
        self._paste_group = QActionGroup(paste_menu)
        self._paste_group.setExclusive(True)
        for keys, label in [("shift+Insert", "Shift+Insert (universal)"),
                            ("ctrl+v", "Ctrl+V (regular apps)"),
                            ("ctrl+shift+v", "Ctrl+Shift+V (terminals)")]:
            a = QAction(label, paste_menu, checkable=True)
            a.setData(keys)
            if keys == config.paste_keys:
                a.setChecked(True)
            a.triggered.connect(self._on_paste_keys_changed)
            self._paste_group.addAction(a)
            paste_menu.addAction(a)

        settings_menu.addSeparator()

        # Models directory
        models_dir = config.models_dir or "auto"
        self._models_dir_action = QAction(f"Models: {models_dir}", settings_menu)
        self._models_dir_action.triggered.connect(self._on_models_dir_change)
        settings_menu.addAction(self._models_dir_action)

        settings_menu.addSeparator()

        # Wake word
        self._wake_word_action = QAction(f"Wake word: {config.wake_word}", settings_menu)
        self._wake_word_action.triggered.connect(self._on_wake_word_change)
        settings_menu.addAction(self._wake_word_action)

        # Wake model
        self._wake_model_menu = settings_menu.addMenu("Wake model")
        self._wake_model_group = QActionGroup(self._wake_model_menu)
        self._wake_model_group.setExclusive(True)
        self._rebuild_wake_model_menu()

        settings_menu.addSeparator()

        # Silence timeout (inline spinbox)
        from PyQt5.QtWidgets import QWidgetAction, QWidget, QHBoxLayout, QLabel, QDoubleSpinBox
        silence_widget = QWidget()
        silence_layout = QHBoxLayout(silence_widget)
        silence_layout.setContentsMargins(8, 2, 8, 2)
        silence_label = QLabel("Silence timeout:")
        self._silence_spin = QDoubleSpinBox()
        self._silence_spin.setRange(0.5, 300.0)
        self._silence_spin.setSingleStep(0.5)
        self._silence_spin.setDecimals(1)
        self._silence_spin.setSuffix("s")
        self._silence_spin.setValue(config.silence_timeout)
        self._silence_spin.valueChanged.connect(self._on_silence_timeout_spin)
        silence_layout.addWidget(silence_label)
        silence_layout.addWidget(self._silence_spin)
        self._silence_widget_action = QWidgetAction(settings_menu)
        self._silence_widget_action.setDefaultWidget(silence_widget)
        settings_menu.addAction(self._silence_widget_action)

        # End signal toggle
        self._end_signal_action = QAction("End signal (beep)", settings_menu, checkable=True)
        self._end_signal_action.setChecked(config.end_signal)
        self._end_signal_action.triggered.connect(self._on_end_signal_toggled)
        settings_menu.addAction(self._end_signal_action)

        # Voice commands toggle
        self._voice_cmd_action = QAction("Voice commands", settings_menu, checkable=True)
        self._voice_cmd_action.setChecked(config.voice_commands)
        self._voice_cmd_action.triggered.connect(self._on_voice_commands_toggled)
        settings_menu.addAction(self._voice_cmd_action)

        # Voice commands editor
        self._edit_cmds_action = QAction("Edit voice commands...", settings_menu)
        self._edit_cmds_action.triggered.connect(self._on_edit_voice_commands)
        settings_menu.addAction(self._edit_cmds_action)

    # -- Event handlers --

    def _on_activated(self, reason):
        from PyQt5.QtWidgets import QSystemTrayIcon
        if reason == QSystemTrayIcon.Trigger:
            self._app_ref.toggle()

    def _on_language_changed(self):
        action = self._lang_group.checkedAction()
        if action:
            self._app_ref.config.language = action.data()
            self._app_ref.config.save()
            log.info("Language changed to: %s", action.data())

    def _on_model_changed(self):
        action = self._model_group.checkedAction()
        if action:
            self._app_ref.config.model = action.data()
            self._app_ref.config.save()
            self._rebuild_wake_model_menu()
            log.info("Model changed to: %s", action.data())

    def _on_download_model(self, name):
        if name in self._downloading:
            return
        self._downloading.add(name)
        self._rebuild_model_menu()
        log.info("Download requested: model '%s'", name)
        self.notify("whisper.linux", f"Downloading {name}...")
        t = threading.Thread(target=self._do_download_model, args=(name,), daemon=True)
        t.start()

    def _marshal_call(self, func):
        """Schedule a callable on the Qt main thread (thread-safe)."""
        self._call_helper.call(func)

    def _do_download_model(self, name):
        script = _REPO_ROOT / "models" / "download-ggml-model.sh"
        if not script.is_file():
            self._downloading.discard(name)
            log.error("Download script not found: %s", script)
            self._marshal_call(
                lambda: self.notify("Error", "download-ggml-model.sh not found"))
            return
        models_dir = self._app_ref.config.models_dir
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        model_path = Path(models_dir) / f"ggml-{name}.bin"

        # Remove incomplete downloads so the script doesn't skip them
        min_size = self._MODEL_MIN_SIZES.get(name, 0)
        if model_path.is_file() and min_size and model_path.stat().st_size < min_size:
            log.warning("Removing incomplete model %s (%d bytes, expected >= %d)",
                        model_path, model_path.stat().st_size, min_size)
            model_path.unlink()

        log.info("Download started: model '%s' \u2192 %s", name, models_dir)
        try:
            result = subprocess.run(
                ["bash", str(script), name, models_dir],
                capture_output=True, text=True, timeout=3600,
            )
            if result.returncode != 0:
                log.error("Download script failed (rc=%d): %s",
                          result.returncode, result.stderr[:200])
            if model_path.is_file():
                size = model_path.stat().st_size
                if min_size and size < min_size:
                    log.error("Downloaded model too small (%d bytes, expected >= %d), removing",
                              size, min_size)
                    model_path.unlink()
                    self._downloading.discard(name)
                    msg = f"Download incomplete ({size // 1_000_000}MB), please retry"
                    self._marshal_call(lambda: self.notify("Error", msg))
                    return
                self._downloading.discard(name)
                size_mb = size / (1024 * 1024)
                log.info("Download complete: model '%s' (%d MB)", name, size_mb)
                msg = f"Model '{name}' ready ({size_mb:.0f} MB)"
                self._marshal_call(lambda: (
                    self._rebuild_model_menu(),
                    self._rebuild_wake_model_menu(),
                    self.notify("whisper.linux", msg),
                ))
            else:
                self._downloading.discard(name)
                err = result.stderr[:100]
                log.error("Download failed: model file not found after script: %s", model_path)
                self._marshal_call(
                    lambda: self.notify("Error", f"Download failed: {err}"))
        except subprocess.TimeoutExpired:
            self._downloading.discard(name)
            log.error("Download timed out for model '%s' (1 hour limit)", name)
            self._marshal_call(
                lambda: self.notify("Error", f"Download timed out for '{name}' (1 hour limit)"))
        except Exception as e:
            self._downloading.discard(name)
            log.error("Download failed for model '%s': %s", name, e)
            err_str = str(e)
            self._marshal_call(
                lambda: self.notify("Error", f"Download failed: {err_str}"))

    def _on_threads_changed(self):
        action = self._threads_group.checkedAction()
        if action:
            self._app_ref.config.threads = action.data()
            self._app_ref.config.save()
            log.info("Threads changed to: %d", action.data())

    def _on_gpu_changed(self):
        action = self._gpu_group.checkedAction()
        if action:
            self._app_ref.config.gpu_device = action.data()
            self._app_ref.config.save()
            log.info("GPU device changed to: %d", action.data())

    def _on_audio_changed(self):
        action = self._audio_group.checkedAction()
        if action:
            self._app_ref.config.audio_device = action.data()
            self._app_ref.config.save()
            log.info("Audio device changed to: %s", action.data())

    def _on_paste_keys_changed(self):
        action = self._paste_group.checkedAction()
        if action:
            self._app_ref.config.paste_keys = action.data()
            self._app_ref.config.save()
            log.info("Paste mode changed to: %s", action.data())

    def _on_models_dir_change(self):
        from PyQt5.QtWidgets import QFileDialog
        current = self._app_ref.config.models_dir or str(Path.home())
        new_dir = QFileDialog.getExistingDirectory(None, "Models directory", current)
        if not new_dir:
            return
        self._app_ref.config.models_dir = new_dir
        self._app_ref.config.save()
        self._models_dir_action.setText(f"Models: {new_dir}")
        self._rebuild_model_menu()
        self._rebuild_wake_model_menu()
        log.info("Models dir changed to: %s", new_dir)

    def _on_input_mode_changed(self):
        action = self._input_mode_group.checkedAction()
        if action:
            old_val = self._app_ref.config.input_mode
            new_val = action.data()
            if old_val != new_val and self._app_ref.state != AppState.IDLE:
                self._app_ref._force_idle()
            self._app_ref.config.input_mode = new_val
            self._app_ref.config.save()
            log.info("Input mode changed to: %s", new_val)
            if new_val == "listen" and self._app_ref.state == AppState.IDLE:
                self._app_ref.toggle()

    def _on_output_mode_changed(self):
        action = self._output_mode_group.checkedAction()
        if action:
            old_val = self._app_ref.config.output_mode
            new_val = action.data()
            if old_val != new_val and self._app_ref.state != AppState.IDLE:
                self._app_ref._force_idle()
            self._app_ref.config.output_mode = new_val
            self._app_ref.config.save()
            log.info("Output mode changed to: %s", new_val)

    def _on_wake_word_change(self):
        from PyQt5.QtWidgets import QInputDialog
        current = self._app_ref.config.wake_word
        text, ok = QInputDialog.getText(None, "Wake Word", "Enter wake word:", text=current)
        if ok and text.strip():
            self._app_ref.config.wake_word = text.strip()
            self._app_ref.config.save()
            self._wake_word_action.setText(f"Wake word: {text.strip()}")
            log.info("Wake word changed to: %s", text.strip())

    def _on_wake_model_changed(self):
        action = self._wake_model_group.checkedAction()
        if action:
            self._app_ref.config.wake_model = action.data()
            self._app_ref.config.save()
            log.info("Wake model changed to: %s", action.data() or "(same as main)")

    def _on_silence_timeout_spin(self, value):
        self._app_ref.config.silence_timeout = value
        self._app_ref.config.save()
        log.info("Silence timeout changed to: %.1fs", value)

    def _on_end_signal_toggled(self):
        self._app_ref.config.end_signal = self._end_signal_action.isChecked()
        self._app_ref.config.save()
        log.info("End signal: %s", self._app_ref.config.end_signal)

    def _on_voice_commands_toggled(self):
        self._app_ref.config.voice_commands = self._voice_cmd_action.isChecked()
        self._app_ref.config.save()
        log.info("Voice commands: %s", self._app_ref.config.voice_commands)

    def _on_edit_voice_commands(self):
        from PyQt5.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QTableWidget,
            QTableWidgetItem, QPushButton, QHeaderView, QAbstractItemView,
        )
        from PyQt5.QtCore import Qt

        config = self._app_ref.config
        cmds = dict(config.voice_commands_map)

        dlg = QDialog()
        dlg.setWindowTitle("Voice Commands")
        dlg.setMinimumSize(450, 400)
        layout = QVBoxLayout(dlg)

        table = QTableWidget(len(cmds), 2)
        table.setHorizontalHeaderLabels(["Word", "Action"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        for row, (word, action) in enumerate(cmds.items()):
            table.setItem(row, 0, QTableWidgetItem(word))
            table.setItem(row, 1, QTableWidgetItem(action))
        layout.addWidget(table)

        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add")
        remove_btn = QPushButton("Remove")
        reset_btn = QPushButton("Reset defaults")
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(remove_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(reset_btn)
        layout.addLayout(btn_layout)

        ok_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        ok_layout.addStretch()
        ok_layout.addWidget(ok_btn)
        ok_layout.addWidget(cancel_btn)
        layout.addLayout(ok_layout)

        def add_row():
            row = table.rowCount()
            table.insertRow(row)
            table.setItem(row, 0, QTableWidgetItem(""))
            table.setItem(row, 1, QTableWidgetItem("key:Return"))
            table.editItem(table.item(row, 0))

        def remove_row():
            rows = sorted({idx.row() for idx in table.selectedIndexes()}, reverse=True)
            for row in rows:
                table.removeRow(row)

        def reset_defaults():
            table.setRowCount(0)
            for word, action in DEFAULT_VOICE_COMMANDS.items():
                row = table.rowCount()
                table.insertRow(row)
                table.setItem(row, 0, QTableWidgetItem(word))
                table.setItem(row, 1, QTableWidgetItem(action))

        add_btn.clicked.connect(add_row)
        remove_btn.clicked.connect(remove_row)
        reset_btn.clicked.connect(reset_defaults)
        ok_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)

        if dlg.exec_() == QDialog.Accepted:
            new_cmds = {}
            for row in range(table.rowCount()):
                w = (table.item(row, 0).text() or "").strip().lower()
                a = (table.item(row, 1).text() or "").strip()
                if w and a:
                    new_cmds[w] = a
            config.voice_commands_map = new_cmds
            config.save()
            # Update the live VoiceCommands instance
            self._app_ref._voice_commands._commands = new_cmds
            log.info("Voice commands updated: %d entries", len(new_cmds))

    # -- State & notifications --

    def set_state(self, state: AppState):
        self.tray.setIcon(self._icons[state])
        if state == AppState.IDLE:
            self.action_toggle.setText("Start Recording")
            self.tray.setToolTip("whisper.linux \u2014 Idle")
        elif state == AppState.RECORDING:
            self.action_toggle.setText("Stop Recording")
            self.tray.setToolTip("whisper.linux \u2014 Recording...")
        elif state == AppState.PROCESSING:
            self.action_toggle.setText("Processing...")
            self.tray.setToolTip("whisper.linux \u2014 Transcribing...")
        elif state == AppState.LISTENING:
            self.action_toggle.setText("Stop Listening")
            self.tray.setToolTip("whisper.linux \u2014 Listening (say wake word)...")
        elif state == AppState.DICTATING:
            self.action_toggle.setText("Stop Dictating")
            self.tray.setToolTip("whisper.linux \u2014 Dictating...")

    def notify(self, title: str, message: str):
        from PyQt5.QtWidgets import QSystemTrayIcon
        self.tray.showMessage(title, message, QSystemTrayIcon.Information, 3000)
