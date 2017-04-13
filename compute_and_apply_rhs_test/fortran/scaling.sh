#!/bin/bash
#SBATCH --job-name scaling
#SBATCH -p ec
#SBATCH --time=00:10:00
#SBATCH --account=FY150001
#SBATCH --nodes=1
#SBATCH --output=sout%j
#SBATCH --error=serr%j

#calc(){ awk "BEGIN { print "$*" }"  };

exec='./fs4omp'   #'./origomp'

export OMP_DISPLAY_ENV=true
export OMP_PROC_BIND=close
export OMP_SCHEDULE=static
export OMP_DYNAMIC=false

for (( th=1; th<=16; th=th+1 )); do

  export OMP_NUM_THREADS=${th}
  $exec > output

#  tm=$( awk '/raw/  {print $6}' output )
#for origomp
  tm=$( awk '/Raw/  {print $4}' output )
#  echo "Current time is ${tm}"

  if (( $th == 1 )); then
    firsttm=$tm
    speedup[$th]=1.0
  else
#    speedup[$th]=$( calc $firsttm / $tm )
    echo "${th}: first time is ${firsttm}, current time is ${tm}"
    speedup[$th]=$( echo $firsttm / $tm | bc -l )
  fi
done #th


#example with accuracy, scale is accuracy
#my_var=$(echo "scale=5; $temp_var/$temp_var2" | bc)
for (( th=1; th<=16; th=th+1 )); do
echo "${th} ${speedup[${th}]}"
done







