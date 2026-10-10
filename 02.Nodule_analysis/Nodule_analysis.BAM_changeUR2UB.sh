samtools view -h -F 0x104 -q 255 sec17_with_CB_RG.bam | \
awk 'BEGIN{OFS="\t"; sep="\137"} /^@/{print; next} {ur=cx=cy=""; for(i=12;i<=NF;i++){if($i~/^UR:Z:/)ur=substr($i,6); if($i~/^Cx:i:/)cx=substr($i,6); if($i~/^Cy:i:/)cy=substr($i,6)} if(ur!=""&&cx!=""&&cy!="")print $0,"UB:Z:" ur sep cx sep cy; else print}' | \
samtools view -b -o sec17_with_CB_RG_UB.bam -

samtools view sec17_with_CB_RG_UB.bam | head -n 3 | grep -oE "UB:Z:[0-9A-F]+_[0-9]+_[0-9]+"
samtools view sec17_with_CB_RG_UB.bam | head -n 100000 | grep -vc "UB:Z:"   # Expect 0
samtools index sec17_with_CB_RG_UB.bam
samtools view -c sec17_with_CB_RG.bam  ## 7814540
samtools view -c sec17_with_CB_RG_UB.bam ## 6227262
samtools view -c -F 0x104 -q 255 sec17_with_CB_RG.bam ## 6227262