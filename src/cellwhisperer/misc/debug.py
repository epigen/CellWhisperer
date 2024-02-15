import debugpy
import logging
import subprocess


def start_debugger(wait_for_client=True):
    for port in range(5678, 5689):
        try:
            debugpy.listen(("0.0.0.0", port))
            print(f"Debugger listening on port {port}")
            if wait_for_client:
                debugpy.wait_for_client()
            break
        except (subprocess.CalledProcessError, RuntimeError, OSError) as e:
            pass
    else:
        logging.warning("No free port found for debugger")
