# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions

# Add locally built modules
# -callum

module use /mnt/storage/easybuild/modules/local
module load languages/intel/2017.01
module load languages/anaconda2/5.0.1
module use /mnt/storage/scratch/jp8463/modules/modulefiles
module load clang-ykt
module load tools/git

alias sunstat="squeue -u $USER"
alias sunsub="echo useSbatch"

