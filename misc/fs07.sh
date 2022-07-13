#!/bin/bash
#SBATCH -J fs07
#SBATCH -n 120

#SBATCH -t 336:00:00

#SBATCH --mail-user=fgarcia4@umd.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END


#needed for bash in deepthought2 (cf. https://www.glue.umd.edu/hpcc/help/jobs.html#basic)
. ~/.profile

###################################
# run with traditional star-formation 
###################################

#
# Run shell 
#
set -x
#NPE=320;                            # process number
NPE=120;                            # process number
export LANG=en_US

#set environment
module purge
module load intel


NRESTART=-1 #restart data file (use latest data if NRESTART < 0)
#NRESTART=12 #restart data file (use latest data if NRESTART < 0)
#NRESTART0=325 #restart data number in the original run  (use latest data if NRESTART < 0)


#directories
RAMSESDIR=/home/$USER/ramses/dwarf/ramses_dBH;       #ramses directory
BINDIR=/lustre/$USER/ramses/dwarf/bin;                 #bin directory
RAMSES=ramses_zoom0X3d
DIR=/lustre/$USER/ramses/dwarf/data/cluster_evolution/fs07_refine #output directory
ICS=music/haloD_l5_14_z9/level_*      #initial data directory
NMLBASE=namelist/first_galaxy/l5_14.nml               #base namelist
SED=SED/bc03_Chabrier                              #stellar SED
CHTAB=CHtab_files/cloudy_metal_HM12_z.bin          #cooling table
UVB=                                                #uv background file

#create and/or clean outuput directory
#rm -rf $DIR
mkdir -p $DIR
cd $DIR

#copy bin/data files (for a run starting from cosmological initial condition)
cp $BINDIR/$RAMSES ./
if [ ! -e ics ]; then
    mkdir -p ics
    cp -r /lustre/$USER/ramses/dwarf/data/$ICS ics/
    mkdir SED
    cp -r $RAMSESDIR/$SED SED/
    mkdir CHtab_files
    cp $RAMSESDIR/$CHTAB CHtab_files/
fi
if [ $NRESTART -lt 0 ]; then
    NRESTART=`ls -1d output*| wc -l` #use latest data 
fi

NUM=`ls -1 stdout*| wc -l`
STDOUT="stdout"$((NUM))

NMLFILE=namelist.nml
cp $RAMSESDIR/$NMLBASE $NMLFILE

#set-up initial levels
sed -i -e "s@^initfile(1).*@initfile(1)='ics/level_005'@g" $NMLFILE
sed -i -e "s@^initfile(2).*@initfile(2)='ics/level_006'@g" $NMLFILE
sed -i -e "s@^initfile(3).*@initfile(3)='ics/level_007'@g" $NMLFILE
sed -i -e "s@^initfile(4).*@initfile(4)='ics/level_008'@g" $NMLFILE
sed -i -e "s@^initfile(5).*@initfile(5)='ics/level_009'@g" $NMLFILE
sed -i -e "s@^initfile(6).*@initfile(6)='ics/level_010'@g" $NMLFILE
sed -i -e "s@^initfile(7).*@initfile(7)='ics/level_011'@g" $NMLFILE
sed -i -e "s@^initfile(8).*@initfile(8)='ics/level_012'@g" $NMLFILE
sed -i -e "s@^initfile(9).*@initfile(9)='ics/level_013'@g" $NMLFILE
sed -i -e "s@^initfile(10).*@initfile(10)='ics/level_014'@g" $NMLFILE

#simulation time [hour]
sed -i -e "s@^maxruntime.*@maxruntime=336@g" $NMLFILE

#restart output number
sed -i -e "s@^nrestart.*@nrestart=$NRESTART@g" $NMLFILE

#model parameters
sed -i -e "s@^foutput.*@foutput=1000000@g" $NMLFILE
#output interval in Myr, defualt 5.0, changed to 2.5 after output 173 (rerun) 156 (refine) i think
#1 Myrs after output 235 -FABG
sed -i -e "s@dtexpout_Myr.*@dtexpout_Myr=1@g" $NMLFILE 
sed -i -e "s@^nremap.*@nremap=1@g" $NMLFILE
sed -i -e "s@^fof_find.*@fof_find=.true.@g" $NMLFILE
sed -i -e "s@^fof_level.*@fof_level=15@g" $NMLFILE
sed -i -e "s@^npmin.*@npmin=100@g" $NMLFILE
sed -i -e "s@^nsubcycle.*@nsubcycle=1,1,1,1,1,1,1,1,1,2,2,2@g" $NMLFILE
sed -i -e "s@^nexpand.*@nexpand=2,2,2,2,2,2,4,4,1,1@g" $NMLFILE
sed -i -e "s@^jeans_refine.*@jeans_refine=9*0,2,4,6*8,3*4@g" $NMLFILE
sed -i -e "s@^m_refine.*@m_refine=20*8@g" $NMLFILE #refine all existing level KS modified FABG 2021-Sep-13
sed -i -e "s@^mass_cut_refine.*@mass_cut_refine=4.54747e-13@g" $NMLFILE
sed -i -e "s@^var_cut_refine.*@var_cut_refine=0.1@g" $NMLFILE
sed -i -e "s@^levelmin.*@levelmin=5@g" $NMLFILE
sed -i -e "s@^levelmax.*@levelmax=25@g" $NMLFILE
sed -i -e "s@^cic_levelmax.*@cic_levelmax=25@g" $NMLFILE
sed -i -e "s@^ngridtot.*@ngridtot=100000000@g" $NMLFILE # Increase ngridtot 2x - FABG 2022-Jun-26
sed -i -e "s@^nparttot.*@nparttot=100000000@g" $NMLFILE 
sed -i -e "s@^sed_dir.*@sed_dir='$SED'@g" $NMLFILE
sed -i -e "s@^cloudy_metal_file.*@cloudy_metal_file='$CHTAB'@g" $NMLFILE
sed -i -e "s@^uv_file.*@uv_file='$UVB'@g" $NMLFILE
sed -i -e "s@^eta_sn.*@eta_sn=0.1d0@g" $NMLFILE
sed -i -e "s@^Zpop3.*@Zpop3=1d-5@g" $NMLFILE
sed -i -e "s@^dpop3.*@dpop3=-1@g" $NMLFILE #use the same condition as for Pop II star formation
sed -i -e "s@^n_star.*@n_star=-1d0@g" $NMLFILE #use Jeans length to determine density threshold
sed -i -e "s@^pop2_njeans.*@pop2_njeans=4d0@g" $NMLFILE
sed -i -e "s@^Tpop2.*@Tpop2 = 1d3@g" $NMLFILE
sed -i -e "s@^pop2_model.*@pop2_model=4@g" $NMLFILE #SFC star formation
sed -i -e "s@^pop3_model.*@pop3_model=1@g" $NMLFILE #40+80 Msun binary Pop III
sed -i -e "s@^BH_model.*@BH_model=1@g" $NMLFILE #BHL accretion
sed -i -e "s@^sink.*@sink=.true.@g" $NMLFILE
sed -i -e "s@^rt_c_fraction.*@rt_c_fraction=30*0.01@g" $NMLFILE
sed -i -e "s@^rt_nsubcycle.*@rt_nsubcycle=10@g" $NMLFILE
sed -i -e "s@^group_egy.*@group_egy       = 12.44, 14.40, 19.90, 35.079@g" $NMLFILE
sed -i -e "s@^groupL0.*@groupL0       = 11.20, 13.60, 24.59, 54.42@g" $NMLFILE
sed -i -e "s@^groupL1.*@groupL1       = 13.60, 24.59, 54.42, 200.0@g" $NMLFILE
sed -i -e "s@^sec_ion.*@sec_ion = .false.@g" $NMLFILE
sed -i -e "s@^fout_sp.*@fout_sp = -1@g" f$NMLFILE
sed -i -e "s@^fout_hl.*@fout_hl = -1@g" $NMLFILE
sed -i -e "s@^fout_rhmx.*@fout_rhmx = -1@g" $NMLFILE
sed -i -e "s@^fout_phmn.*@fout_phmn = -1@g" $NMLFILE
sed -i -e "s@^nonlocal_depletion.*@nonlocal_depletion=.true.@g" $NMLFILE
sed -i -e "s@^pop2_stochastic_SN.*@pop2_stochastic_SN=.true.@g" $NMLFILE
sed -i -e "s@^modify_pressure_fix_cond.*@modify_pressure_fix_cond  = .true.@g" $NMLFILE
sed -i -e "s@^strict_Npart_refine.*@strict_Npart_refine = .true.@g" $NMLFILE #KS Modified FABG 2021-Sep-13
#SFC MODEL
sed -i -e "s@^SFC_dist_min_pc.*@SFC_dist_min_pc=1d2@g" $NMLFILE
sed -i -e "s@^SFC_nH_min.*@SFC_nH_min=-1d0@g" $NMLFILE
sed -i -e "s@^SFC_f_nH_min.*@SFC_f_nH_min=0.1d0@g" $NMLFILE
sed -i -e "s@^SFC_fate.*@SFC_fate='crop'@g" $NMLFILE
sed -i -e "s@^SFC_f_rhocrop.*@SFC_f_rhocrop=2d0@g" $NMLFILE
sed -i -e "s@^SFC_f_rhoSF.*@SFC_f_rhoSF=1d0@g" $NMLFILE
sed -i -e "s@^SFC_distribution.*@SFC_distribution='mass_weight'@g" $NMLFILE
sed -i -e "s@^SFC_flg_thin.*@SFC_flg_thin=.false.@g" $NMLFILE
#SF/SN/chem model
sed -i -e "s@^nH_postSF.*@nH_postSF=-1d2@g" $NMLFILE
sed -i -e "s@^nHmax_preSN.*@nHmax_preSN=1d2@g" $NMLFILE
sed -i -e "s@^flg_thin_star_cell.*@flg_thin_star_cell=.true.@g" $NMLFILE
sed -i -e "s@^tage_thin_star_cell.*@tage_thin_star_cell=1d1@g" $NMLFILE #cell containing stars with t < 10 Myr is assumed to be thin
sed -i -e "s@^H2_chem_cool.*@H2_chem_cool=.false.@g" $NMLFILE #H2 formation heating is neglected (should be ok for nH < 1e9 cm-3)
sed -i -e "s@^attn_after_chem.*@attn_after_chem=.true.@g" $NMLFILE #to accelerate computation


#######################################################
#      Set Pop II stellar mass to 10 Msun             #
#######################################################
sed -i -e "s@^m_star.*@m_star = 1e1@g" $NMLFILE #assume mass of PopII particle = 10 Msun
#######################################################

#######################################################
#      Set f_star = M_star/M_cloud to 0.7             #
#######################################################
sed -i -e "s@^SFC_f_star.*@SFC_f_star=7d-1@g" $NMLFILE #assume f_star = 0.7
#######################################################


#run ramses
date 2>&1 | tee -a $STDOUT
mpirun -n $NPE ./$RAMSES namelist.nml 2>&1 | tee -a $STDOUT
date 2>&1 | tee -a $STDOUT
