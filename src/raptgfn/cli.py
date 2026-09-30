"""RaptGFN command-line entry point.

Dispatch training and generation to their Hydra-enabled command modules.
"""

import subprocess
import sys


def main():
    """Dispatch the command-line request.
    
    Usage:
        raptgfn train [Hydra overrides...]
        raptgfn generate [Hydra overrides...]
    """
    if len(sys.argv) < 2:
        print("Usage:")
        print("  raptgfn train [Hydra overrides...]")
        print("  raptgfn generate [Hydra overrides...]")
        print("\nExamples:")
        print("  raptgfn train")
        print("  raptgfn train task=ht_selex_template")
        print("  raptgfn generate checkpoint_path=outputs/model.pth num_samples=1000")
        sys.exit(1)

    command = sys.argv[1]

    # Pass arguments after the command name to Hydra.
    # Run each Hydra-decorated entry point as a module.
    # Use a subprocess to preserve command-line initialization.
    if command == "train":
        # Launch training in a subprocess.
        cmd = [sys.executable, "-m", "raptgfn.commands.train"] + sys.argv[2:]
        sys.exit(subprocess.call(cmd))
    elif command == "generate":
        # Launch generation in a subprocess.
        cmd = [sys.executable, "-m", "raptgfn.commands.generate"] + sys.argv[2:]
        sys.exit(subprocess.call(cmd))
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, generate")
        sys.exit(1)


if __name__ == "__main__":
    main()
