#!/bin/bash
#change these parameters for a new run; if Jobname is changed, make sure conf/parameters_jobname.txt exists
#do not use 0.01 but 1E-2 instead to avoid dots in file names

Jobname="1dTFIM" #job name and start of filenames of job and log files
pyscript="MPS_Program_S_vs_L.py"
Nvec=( 50, 70)
Jvec=( 1)
bvec=( 2E-1 9E-1 2E0)
eps_truncvec=( 1E-6)
Dmaxvec=( 10)
dt0vec=( 4E-1)
eps_dtvec=( 2E-2)
dt_redvec=( 8E-1)
eps_tolvec=( 1E-3)
phys_gapvec=( 1E-3)
bdim_startvec=( 2)

Jobnamestart="./jobs/"
Jobnamereplace="./jobs/referencejob.job"

output_dir=runtime_measurements
mkdir -p results log compiled_scripts ${output_dir}

for N in ${Nvec[@]}; do
  for J in ${Jvec[@]}; do
    for b in ${bvec[@]}; do
      for eps_trunc in ${eps_truncvec[@]}; do
        for Dmax in ${Dmaxvec[@]}; do
          for dt0 in ${dt0vec[@]}; do
            for eps_dt in ${eps_dtvec[@]}; do
              for dt_red in ${dt_redvec[@]}; do
                for eps_tol in ${eps_tolvec[@]}; do
                  for phys_gap in ${phys_gapvec[@]}; do
                    for bdim in ${bdim_startvec[@]}; do
                      prefix=${Jobname}_N${N}_J${J}_b${b}
                      center=epstrunc${eps_trunc}_Dmax${Dmax}_dt0${dt0}_epsdt${eps_dt}_dtred${dt_red}_epstol${eps_tol}_physgap${phys_gap}_bdim${bdim}
                      postfix=
                      Jobfile=${Jobnamestart}${prefix}_${center}_${postfix}.job

                      sed "s/JOBNAMEREPLACE/${Jobname}/g" ${Jobnamereplace} \
                      | sed "s/PYSCRIPTREPLACE/${pyscript}/g" \
                      | sed "s/NREPLACE/${N}/g" \
                      | sed "s/JREPLACE/${J}/g" \
                      | sed "s/BREPLACE/${b}/g" \
                      | sed "s/EPS_TRUNCREPLACE/${eps_trunc}/g" \
                      | sed "s/DMAXREPLACE/${Dmax}/g" \
                      | sed "s/DT0REPLACE/${dt0}/g" \
                      | sed "s/EPS_DTREPLACE/${eps_dt}/g" \
                      | sed "s/DT_REDREPLACE/${dt_red}/g" \
                      | sed "s/EPS_TOLREPLACE/${eps_tol}/g" \
                      | sed "s/PHYS_GAPREPLACE/${phys_gap}/g" \
                      | sed "s/BDIMREPLACE/${bdim}/g" \
                      | sed "s/OUTPUT_DIR/${output_dir}/g" \
                      > ${Jobfile}
                      chmod +x ${Jobfile}

                      sbatch ${Jobfile}	#run on cluster
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
