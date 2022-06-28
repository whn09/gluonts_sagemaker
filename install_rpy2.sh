# update indices
sudo apt update -qq
# install two helper packages we need
sudo apt install -y --no-install-recommends software-properties-common dirmngr apt-transport-https
# import the signing key (by Michael Rutter) for these repo
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
# add the R 4.0 repo from CRAN -- adjust 'focal' to 'groovy' or 'bionic' as needed
# sudo add-apt-repository "deb https://opentuna.cn/CRAN/bin/linux/ubuntu/ $(lsb_release -cs)-cran35/"
sudo add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran35/"
sudo apt update -qq
sudo apt install -y --no-install-recommends r-base r-base-dev
sudo apt install -y libcurl4-openssl-dev
sudo R -e 'install.packages(c("forecast", "nnfor"), repos="https://cloud.r-project.org", dependencies=TRUE)'
pip install 'rpy2>=2.9.*,<3.*'
