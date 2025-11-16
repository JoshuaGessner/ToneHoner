import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import time
import json
import urllib.request
import urllib.error

import uvicorn

# Import the FastAPI app without running the server
try:
    from main import app  # FastAPI instance
except Exception as e:
    raise RuntimeError(f"Failed to import server app: {e}")


class ServerController:
    def __init__(self):
        self.server = None
        self.thread = None
        self.running = False
        self.host = "0.0.0.0"
        self.port = 8000

    def start(self, host: str, port: int, fp16: bool, debug_rtf: bool, metrics_interval: int):
        if self.running:
            return
        # Apply settings to environment for the server process
        os.environ["TONEHONER_FP16"] = "1" if fp16 else "0"
        os.environ["TONEHONER_DEBUG_RTF"] = "1" if debug_rtf else "0"
        os.environ["TONEHONER_METRICS_INTERVAL"] = str(max(1, int(metrics_interval)))

        self.host = host
        self.port = int(port)

        # Disable uvicorn's default logging config to avoid PyInstaller issues
        # Use a simple log config that doesn't rely on formatters
        config = uvicorn.Config(
            app, 
            host=self.host, 
            port=self.port, 
            log_level="info",
            log_config=None  # Disable default logging config
        )
        self.server = uvicorn.Server(config)
        # Avoid installing signal handlers in a background thread (prevents Ctrl+C noise)
        try:
            self.server.install_signal_handlers = False
        except Exception:
            pass

        def _run():
            try:
                self.running = True
                self.server.run()
            finally:
                self.running = False

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()

    def stop(self):
        if not self.running or self.server is None:
            return
        try:
            self.server.should_exit = True
            # Give it a moment to shut down gracefully
            for _ in range(50):
                if not self.running:
                    break
                time.sleep(0.1)
        except Exception:
            pass
        finally:
            self.server = None
            self.thread = None
            self.running = False

    def reset_model(self):
        # Call the control endpoint to reload model and warm-up
        try:
            base = self._client_base_url()
            url = f"{base}/control/reset"
            req = urllib.request.Request(url, method="POST")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    def fetch_metrics(self):
        try:
            base = self._client_base_url()
            url = f"{base}/metrics"
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

    def _client_base_url(self) -> str:
        host = (self.host or "").strip()
        if host in ("0.0.0.0", "::", ""):
            host = "127.0.0.1"
        return f"http://{host}:{self.port}"


class ServerGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ToneHoner Server Console")
        self.root.geometry("720x520")

        self.controller = ServerController()
        self._save_job = None

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tabs
        self.tab_server = ttk.Frame(self.notebook, padding=12)
        self.tab_metrics = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(self.tab_server, text="Server")
        self.notebook.add(self.tab_metrics, text="Metrics")

        self._build_server_tab()
        self._build_metrics_tab()

        # Load persisted settings (if any)
        self._load_settings()
        # Attach change listeners to auto-save with debounce
        self._attach_traces()

        # Periodic UI updates
        self._tick_status()
        self._tick_metrics()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_server_tab(self):
        # Host/Port
        ttk.Label(self.tab_server, text="Host").grid(row=0, column=0, sticky=tk.W)
        self.var_host = tk.StringVar(value="0.0.0.0")
        ttk.Entry(self.tab_server, textvariable=self.var_host, width=20).grid(row=0, column=1, sticky=tk.W)

        ttk.Label(self.tab_server, text="Port").grid(row=1, column=0, sticky=tk.W)
        self.var_port = tk.IntVar(value=8000)
        ttk.Entry(self.tab_server, textvariable=self.var_port, width=10).grid(row=1, column=1, sticky=tk.W)

        # Settings
        self.var_fp16 = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.tab_server, text="Enable FP16 (GPU)", variable=self.var_fp16).grid(row=2, column=0, columnspan=2, sticky=tk.W)

        self.var_debug_rtf = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.tab_server, text="Debug per-frame RTF logs", variable=self.var_debug_rtf).grid(row=3, column=0, columnspan=2, sticky=tk.W)

        ttk.Label(self.tab_server, text="Metrics interval (s)").grid(row=4, column=0, sticky=tk.W)
        self.var_metrics_interval = tk.IntVar(value=5)
        ttk.Spinbox(self.tab_server, from_=1, to=60, textvariable=self.var_metrics_interval, width=6).grid(row=4, column=1, sticky=tk.W)

        # Controls
        self.btn_start = ttk.Button(self.tab_server, text="Start", command=self._on_start)
        self.btn_stop = ttk.Button(self.tab_server, text="Stop", command=self._on_stop, state=tk.DISABLED)
        self.btn_reset = ttk.Button(self.tab_server, text="Reset Model", command=self._on_reset, state=tk.DISABLED)
        self.btn_start.grid(row=5, column=0, sticky=tk.W, pady=(12,4))
        self.btn_stop.grid(row=5, column=1, sticky=tk.W, pady=(12,4))
        self.btn_reset.grid(row=5, column=2, sticky=tk.W, pady=(12,4))

        # Status
        ttk.Separator(self.tab_server).grid(row=6, column=0, columnspan=3, sticky="ew", pady=8)
        ttk.Label(self.tab_server, text="Status", font=("Segoe UI", 10, "bold")).grid(row=7, column=0, sticky=tk.W)
        self.var_status = tk.StringVar(value="Stopped")
        ttk.Label(self.tab_server, textvariable=self.var_status).grid(row=7, column=1, columnspan=2, sticky=tk.W)

        for i in range(3):
            self.tab_server.columnconfigure(i, weight=0)

    def _settings_path(self) -> str:
        def _user_config_dir() -> str:
            # Windows
            appdata = os.environ.get("APPDATA")
            if os.name == "nt":
                if appdata and os.path.isdir(appdata):
                    return os.path.join(appdata, "ToneHoner")
                # Fallback to Roaming under user profile
                return os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "ToneHoner")
            # macOS
            if sys.platform == "darwin":
                return os.path.join(os.path.expanduser("~"), "Library", "Application Support", "ToneHoner")
            # Linux/Unix (XDG)
            xdg = os.environ.get("XDG_CONFIG_HOME")
            if xdg:
                return os.path.join(xdg, "ToneHoner")
            return os.path.join(os.path.expanduser("~"), ".config", "ToneHoner")

        cfg_dir = _user_config_dir()
        try:
            os.makedirs(cfg_dir, exist_ok=True)
        except Exception:
            # As a last resort, fall back to current directory
            cfg_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(cfg_dir, "server_settings.json")

    def _load_settings(self):
        path = self._settings_path()
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.var_host.set(str(data.get("host", self.var_host.get())))
                self.var_port.set(int(data.get("port", self.var_port.get())))
                self.var_fp16.set(bool(data.get("fp16", self.var_fp16.get())))
                self.var_debug_rtf.set(bool(data.get("debug_rtf", self.var_debug_rtf.get())))
                self.var_metrics_interval.set(int(data.get("metrics_interval", self.var_metrics_interval.get())))
        except Exception as e:
            messagebox.showwarning("Settings", f"Failed to load settings: {e}")

    def _save_settings(self):
        data = {
            "host": self.var_host.get().strip(),
            "port": int(self.var_port.get()),
            "fp16": bool(self.var_fp16.get()),
            "debug_rtf": bool(self.var_debug_rtf.get()),
            "metrics_interval": int(self.var_metrics_interval.get()),
        }
        try:
            with open(self._settings_path(), "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            # Non-fatal; just log to status label
            try:
                self.var_status.set(f"Settings save failed: {e}")
            except Exception:
                pass

    def _save_settings_debounced(self, delay_ms: int = 400):
        if self._save_job is not None:
            try:
                self.root.after_cancel(self._save_job)
            except Exception:
                pass
            self._save_job = None
        self._save_job = self.root.after(delay_ms, self._save_settings)

    def _attach_traces(self):
        def _cb(*_):
            self._save_settings_debounced()
        for var in (self.var_host, self.var_port, self.var_fp16, self.var_debug_rtf, self.var_metrics_interval):
            try:
                var.trace_add("write", _cb)
            except Exception:
                # For older Tk versions
                var.trace("w", _cb)

    def _build_metrics_tab(self):
        labels = [
            ("frames_processed", "Frames Processed"),
            ("framed_frames", "Framed Frames"),
            ("avg_proc_ms", "Avg Proc (ms)"),
            ("last_proc_ms", "Last Proc (ms)"),
            ("avg_rtf", "Avg RTF"),
            ("queue_backlog", "Queue Backlog"),
            ("sample_rate", "Sample Rate"),
        ]
        self.metric_vars = {}
        r = 0
        for key, title in labels:
            ttk.Label(self.tab_metrics, text=title).grid(row=r, column=0, sticky=tk.W, pady=(3,3))
            var = tk.StringVar(value="-")
            self.metric_vars[key] = var
            ttk.Label(self.tab_metrics, textvariable=var).grid(row=r, column=1, sticky=tk.W, pady=(3,3))
            r += 1

        ttk.Separator(self.tab_metrics).grid(row=r, column=0, columnspan=2, sticky="ew", pady=8)
        r += 1
        self.var_metrics_error = tk.StringVar(value="")
        ttk.Label(self.tab_metrics, textvariable=self.var_metrics_error, foreground="red").grid(row=r, column=0, columnspan=2, sticky=tk.W)

        self.tab_metrics.columnconfigure(1, weight=1)

    def _on_start(self):
        try:
            self.controller.start(
                host=self.var_host.get().strip(),
                port=int(self.var_port.get()),
                fp16=bool(self.var_fp16.get()),
                debug_rtf=bool(self.var_debug_rtf.get()),
                metrics_interval=int(self.var_metrics_interval.get()),
            )
            self._save_settings()
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.btn_reset.config(state=tk.NORMAL)
            self.var_status.set(f"Running on {self.var_host.get()}:{self.var_port.get()}")
        except Exception as e:
            messagebox.showerror("Start Failed", f"Could not start server: {e}")

    def _on_stop(self):
        try:
            self.controller.stop()
            self._save_settings()
            self.btn_start.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)
            self.btn_reset.config(state=tk.DISABLED)
            self.var_status.set("Stopped")
        except Exception as e:
            messagebox.showwarning("Stop Warning", f"Stop encountered an issue: {e}")

    def _on_reset(self):
        if not self.controller.running:
            return
        res = self.controller.reset_model()
        if isinstance(res, dict) and res.get("status") == "reloaded":
            messagebox.showinfo("Reset", "Model reloaded and warmed up.")
        else:
            message = json.dumps(res) if isinstance(res, dict) else str(res)
            messagebox.showerror("Reset Failed", f"Reset failed: {message}")

    def _tick_status(self):
        if self.controller.running:
            self.var_status.set(f"Running on {self.controller.host}:{self.controller.port}")
        else:
            self.var_status.set("Stopped")
        self.root.after(500, self._tick_status)

    def _tick_metrics(self):
        if self.controller.running:
            data = self.controller.fetch_metrics()
            if data is None:
                self.var_metrics_error.set("Unable to fetch metrics (server starting or unreachable)")
            else:
                self.var_metrics_error.set("")
                for k, var in self.metric_vars.items():
                    val = data.get(k, "-")
                    if isinstance(val, float):
                        if k.endswith("_ms"):
                            var.set(f"{val:.2f}")
                        else:
                            var.set(f"{val:.3f}")
                    else:
                        var.set(str(val))
        else:
            self.var_metrics_error.set("")
            for var in self.metric_vars.values():
                var.set("-")
        self.root.after(1000, self._tick_metrics)

    def _on_close(self):
        try:
            self.controller.stop()
        except Exception:
            pass
        try:
            self._save_settings()
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    gui = ServerGUI(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        try:
            gui.controller.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
