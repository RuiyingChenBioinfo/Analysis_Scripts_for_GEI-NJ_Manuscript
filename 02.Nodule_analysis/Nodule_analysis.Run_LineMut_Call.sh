#!/usr/bin/env bash
set -eo pipefail

WORKDIR=/jdfsbjcas1/ST_BJ/P21Z28400N0234/litianle/9_LineMut_test
BAMDIR=${WORKDIR}/1010
LINEMUT=linemut_call
REF=${WORKDIR}/genome.fasta

SEC=sec17_UB_5ct_mixed
RG_BAM=${BAMDIR}/sec17_with_CB_RG_UB.bam
MAPPING=${WORKDIR}/S17_linemut_units_5celltypes_mixed.csv
OUTDIR=${WORKDIR}/LineMut_Output_sec17_UB_5ct_mixed

mkdir -p ${WORKDIR}/qsub_logs ${WORKDIR}/job_scripts ${WORKDIR}/.linemut_bin

cat > ${WORKDIR}/.linemut_bin/nproc <<'EOF'
#!/usr/bin/env bash
echo 16
EOF
chmod +x ${WORKDIR}/.linemut_bin/nproc

JOB=${WORKDIR}/job_scripts/run_linemut_${SEC}.sh

cat > ${JOB} <<EOF
#!/usr/bin/env bash
set -eo pipefail

cd ${WORKDIR}

source 0_Software/miniconda3/etc/profile.d/conda.sh
conda activate linemut

export PATH=${WORKDIR}/.linemut_bin:\$PATH
export SAMTOOLS=\$(which samtools)
export GATK=\$(which gatk)

echo "[${SEC}] start: \$(date)"
echo "RG_BAM: ${RG_BAM}"
echo "MAPPING: ${MAPPING}"
echo "OUTDIR: ${OUTDIR}"
echo "python: \$(which python)"
echo "python version: \$(python --version 2>&1)"
echo "samtools: \$(which samtools)"
echo "gatk: \$(which gatk || true)"
echo "nproc used by LineMut: \$(nproc)"

if [[ ! -s ${RG_BAM} ]]; then
    echo "ERROR: BAM not found: ${RG_BAM}"
    exit 1
fi
if [[ ! -s ${RG_BAM}.bai ]]; then
    echo "bai missing, indexing..."
    samtools index ${RG_BAM}
fi
if [[ ! -s ${MAPPING} ]]; then
    echo "ERROR: mapping file not found: ${MAPPING}"
    echo "Run this first: sed 's/Peripheral_tissues/mixed/g' S17_linemut_units_5celltypes.csv > S17_linemut_units_5celltypes_mixed.csv"
    exit 1
fi

echo "Read group:"
samtools view -H ${RG_BAM} | grep '^@RG' || true

echo "UB sanity check (first read, expect UB:Z:HEX_Cx_Cy):"
FIRST_UB=\$(samtools view ${RG_BAM} | head -n 1 | grep -oE 'UB:Z:[0-9A-F]+_[0-9]+_[0-9]+' || true)
echo "first read UB: \${FIRST_UB}"
if [[ -z "\${FIRST_UB}" ]]; then
    echo "ERROR: UB tag not found in first read!"
    exit 1
fi

echo "UB coverage check (first 100k reads, expect 0):"
NO_UB=\$(samtools view ${RG_BAM} | head -n 100000 | grep -vc 'UB:Z:' || true)
echo "reads without UB: \${NO_UB}"
if [[ "\${NO_UB}" -ne 0 ]]; then
    echo "ERROR: some reads missing UB tag"
    exit 1
fi

echo "BAM read count (expect 6227262):"
samtools view -c ${RG_BAM}

echo "Mapping celltype distribution:"
awk -F, '{print \$2}' ${MAPPING} | sort | uniq -c
echo "Mapping line count:"
wc -l ${MAPPING}

if [[ -d ${OUTDIR} ]]; then
    echo "ERROR: output directory already exists: ${OUTDIR}"
    exit 1
fi

echo "Launching LineMut: \$(date)"
if ${LINEMUT} \\
  -I ${RG_BAM} \\
  -R ${REF} \\
  -O ${OUTDIR} \\
  -m ${MAPPING} \\
  --cell-barcode-tag CB; then
    echo "[${SEC}] finished OK: \$(date)"
else
    rc=\$?
    echo "ERROR: LineMut failed with exit code \${rc}: \$(date)"
    exit \${rc}
fi
EOF

chmod +x ${JOB}

qsub -cwd \
  -l vf=200g,num_proc=16 \
  -P P21Z28400N0234_super \
  -binding linear:16 \
  -q st_supermem.q \
  -N LineMut_sec17_UB_5ct_mixed \
  -o ${WORKDIR}/qsub_logs \
  -e ${WORKDIR}/qsub_logs \
  ${JOB}
