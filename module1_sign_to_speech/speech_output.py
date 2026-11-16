import base64
import queue
import subprocess
import threading


class SpeechOutput:
    def __init__(self):
        self._texts = queue.Queue()
        self._state_lock = threading.Lock()
        self._pending_count = 0
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    @property
    def is_busy(self):
        with self._state_lock:
            return self._pending_count > 0

    def speak(self, text):
        if text.strip():
            with self._state_lock:
                self._pending_count += 1
            self._texts.put(text)

    def close(self):
        self._texts.put(None)
        self._worker.join()

    def _run(self):
        while True:
            text = self._texts.get()
            try:
                if text is None:
                    return

                text_base64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
                script = (
                    "$ErrorActionPreference = 'Stop'; "
                    "Add-Type -AssemblyName System.Speech; "
                    f"$text = [Text.Encoding]::UTF8.GetString("
                    f"[Convert]::FromBase64String('{text_base64}')); "
                    "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    "try { $speaker.Speak($text) } finally { $speaker.Dispose() }"
                )
                command = base64.b64encode(script.encode("utf-16le")).decode("ascii")
                result = subprocess.run(
                    [
                        "powershell.exe",
                        "-NoProfile",
                        "-NonInteractive",
                        "-EncodedCommand",
                        command,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode:
                    details = result.stderr.strip() or result.stdout.strip()
                    print(f"Voice output failed: {details}")
            except OSError as error:
                print(f"Voice output could not start: {error}")
            finally:
                self._texts.task_done()
                if text is not None:
                    with self._state_lock:
                        self._pending_count -= 1
