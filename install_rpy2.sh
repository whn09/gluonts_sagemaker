# update indices
apt-get update
# install two helper packages we need
apt-get install -y --no-install-recommends dirmngr gnupg apt-transport-https ca-certificates software-properties-common
# import the signing key (by Michael Rutter) for these repo
apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
# add the R 4.0 repo from CRAN -- adjust 'focal' to 'groovy' or 'bionic' as needed
# add-apt-repository "deb https://opentuna.cn/CRAN/bin/linux/ubuntu/ $(lsb_release -cs)-cran40/"
add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
apt-get update
apt-get install -y --no-install-recommends r-base r-base-dev
apt-get install -y libcurl4-openssl-dev
R -e 'install.packages(c("forecast", "nnfor"), repos="https://cloud.r-project.org", dependencies=TRUE)'
pip install 'rpy2>=2.9.*,<3.*'
