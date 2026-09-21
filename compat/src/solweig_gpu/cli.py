"""Legacy executable delegating to the CPU package parser and dispatcher."""

def main():
    from ._guard import reject_upstream
    reject_upstream()
    from solweig_light.cli import main as cpu_main
    return cpu_main()
