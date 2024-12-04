import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path


def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Create the plot
    plt.figure(figsize=(15, 6))

    # Process each file and add to plot
    for data_file in args.data_files:
        # Load data
        with open(data_file) as f:
            data = json.load(f)

        # Calculate values
        ttfts = [data[entry]["ttft"] for entry in data]
        prompt_lens = [data[entry]["prompt_len"] for entry in data]

        # Print statistics
        filename = Path(data_file).stem
        avg_ttft = np.mean(ttfts)
        avg_prompt_len = np.mean(prompt_lens)
        print(f"\n{filename}:")
        print(f"  Average TTFT: {avg_ttft:.4f} seconds")
        print(f"  Average Prompt Length: {avg_prompt_len:.1f} characters")

        # Add trend line
        z = np.polyfit(prompt_lens, ttfts, 1)
        p = np.poly1d(z)

        # Get x range for smooth line
        x_range = np.linspace(min(prompt_lens), max(prompt_lens), 100)
        plt.plot(x_range, p(x_range), label=filename, alpha=0.8)

    plt.xlabel("Prompt Length (characters)")
    plt.ylabel("Time to First Token (seconds)")
    plt.title("Prompt Length vs Time to First Token - Trend Lines")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Save plot
    output_path = os.path.join(args.output_dir, "ttft_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nPlot saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze TTFT and prompt length from multiple datasets"
    )
    parser.add_argument(
        "--data-files", nargs="+", required=True, help="Paths to JSON data files"
    )
    parser.add_argument(
        "--output-dir", default="plots", help="Directory to save plots (default: plots)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
