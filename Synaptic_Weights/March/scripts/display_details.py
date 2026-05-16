from csv import DictReader
from pathlib import Path

def main():
	# Paths to your models
	base_dir = Path(__file__).resolve().parent
	folder = base_dir / "run_2026-05-15_23-37-49"
	file = folder / "details_log.csv"

	if not file.exists():
		raise FileNotFoundError(f"Could not find details log at: {file}")

	# Load and display the CSV file
	with file.open(newline="", encoding="utf-8") as handle:
		reader = DictReader(handle)
		rows = list(reader)
		columns = reader.fieldnames or []

	if not columns:
		raise ValueError(f"No columns found in CSV: {file}")

	preview_rows = rows[:5]
	table = [columns] + [[row.get(column, "") for column in columns] for row in preview_rows]
	widths = [max(len(str(cell)) for cell in column_values) for column_values in zip(*table)]

	print("Details log preview:\n")
	for row_index, row in enumerate(table):
		print(" ".join(str(cell).rjust(width) for cell, width in zip(row, widths)))
		if row_index == 0:
			print(" ".join("-" * width for width in widths))


if __name__ == "__main__":
	main()

