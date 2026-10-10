python3 - <<'PY'
import csv

# Extract protein-to-LOC mappings from GFF and deduplicate repeated CDS records.
protein_to_loc = {}
with open("genes.checked.gff") as f:
    for line in f:
        if line.startswith("#"):
            continue
        fields = line.rstrip("\r\n").split("\t")
        if len(fields) != 9 or fields[2] != "CDS":
            continue
        attrs = dict(x.split("=", 1) for x in fields[8].split(";") if "=" in x)
        protein, loc = attrs["protein_id"], attrs["gene"]
        if protein in protein_to_loc and protein_to_loc[protein] != loc:
            raise ValueError(f"Protein maps to multiple LOC IDs: {protein}")
        protein_to_loc[protein] = loc

# Save the two-column protein-to-LOC mapping table.
with open("Lotus_XP_to_LOC.tsv", "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t", lineterminator="\n")
    writer.writerow(["protein_id", "LOC_ID"])
    writer.writerows(sorted(protein_to_loc.items()))

# Add LOC_ID to the headerless BLAST output while retaining all 16 original columns.
columns = "qseqid sseqid pident length nident mismatch gapopen qstart qend sstart send evalue bitscore score qlen slen".split()
missing = set()
with open("Lotus_vs_Arabidopsis.blastp.tsv") as fin, open("Lotus_vs_Arabidopsis.with_LOC.tsv", "w", newline="") as fout:
    writer = csv.writer(fout, delimiter="\t", lineterminator="\n")
    writer.writerow(["LOC_ID"] + columns)
    for row in csv.reader(fin, delimiter="\t"):
        if not row:
            continue
        if len(row) != 16:
            raise ValueError(f"Expected 16 columns in the BLAST output: {row[:2]}")
        loc = protein_to_loc.get(row[0], "NA")
        if loc == "NA":
            missing.add(row[0])
        writer.writerow([loc] + row)

print(f"XP-to-LOC mappings: {len(protein_to_loc)}")
print(f"Unmatched protein IDs: {len(missing)}")
if missing:
    print("Examples of unmatched protein IDs:", sorted(missing)[:10])
PY
