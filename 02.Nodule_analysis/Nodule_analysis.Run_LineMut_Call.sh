#!/usr/bin/env bash
set -eo pipefail

WORKDIR=/9_LineMut_test
LINEMUT=0_Software/LineMut/linemut_call
REF=${WORKDIR}/genome.fasta

SEC=sec17_crop
RAW_BAM=${WORKDIR}/sec17_with_CB.bam
RG_BAM=${WORKDIR}/sec17_with_CB_RG.bam
MAPPING=${WORKDIR}/S17_crop_marker_based_mapping.linemut_units.csv
OUTDIR=${WORKDIR}/LineMut_Output_sec17_crop_marker_based

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

source miniconda3/etc/profile.d/conda.sh
conda activate linemut

export PATH=${WORKDIR}/.linemut_bin:\$PATH
export SAMTOOLS=\$(which samtools)
export GATK=\$(which gatk)

echo "[${SEC}] start: \$(date)"
echo "RAW_BAM: ${RAW_BAM}"
echo "RG_BAM: ${RG_BAM}"
echo "MAPPING: ${MAPPING}"
echo "OUTDIR: ${OUTDIR}"
echo "python: \$(which python)"
echo "python version: \$(python --version 2>&1)"
echo "samtools: \$(which samtools)"
echo "gatk: \$(which gatk || true)"
echo "nproc used by LineMut: \$(nproc)"

if [[ ! -s ${RG_BAM} ]]; then
    echo "RG BAM not found. Creating: ${RG_BAM}"

    gatk AddOrReplaceReadGroups \\
      -I ${RAW_BAM} \\
      -O ${RG_BAM} \\
      -RGID sec17 \\
      -RGLB sec17 \\
      -RGPL ILLUMINA \\
      -RGPU sec17 \\
      -RGSM sec17

    samtools index ${RG_BAM}
fi

if [[ ! -s ${RG_BAM}.bai ]]; then
    samtools index ${RG_BAM}
fi

echo "Read group:"
samtools view -H ${RG_BAM} | grep '^@RG'

echo "Mapping preview:"
head ${MAPPING}
echo "Mapping line count:"
wc -l ${MAPPING}

if [[ -d ${OUTDIR} ]]; then
    echo "ERROR: output directory already exists: ${OUTDIR}"
    exit 1
fi

${LINEMUT} \\
  -I ${RG_BAM} \\
  -R ${REF} \\
  -O ${OUTDIR} \\
  -m ${MAPPING} \\
  --cell-barcode-tag CB

echo "[${SEC}] finished: \$(date)"
EOF

chmod +x ${JOB}

qsub -cwd \
  -l vf=200g,num_proc=16 \
  -P Pxxx_super \
  -binding linear:16 \
  -q st_supermem.q \
  -N LineMut_sec17_crop \
  -o ${WORKDIR}/qsub_logs \
  -e ${WORKDIR}/qsub_logs \
  ${JOB}