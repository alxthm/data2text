# Data2text

## Installation

### Python environment
Experiments are done with python 3.6.8 (default version on Mozart currently).
Create a virtualenv and install the `requirements.txt` with pip

```
pip install --upgrade pip
pip install -r requirements.txt
```

In order to be able to run and import code from the `src` directory, run:
```
pip install -e .
```

### Evaluation scripts
To install requirements for g2t evaluation with BLEU/METEOR (following [WebNLG 2020 challenge
instructions](https://github.com/WebNLG/GenerationEval/blob/master/install_dependencies.sh)):

```
cd src/eval/webnlg_g2t

git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .
wget https://storage.googleapis.com/bleurt-oss/bleurt-base-128.zip
unzip bleurt-base-128.zip
rm bleurt-base-128.zip 
cd ../
mv bleurt metrics

# INSTALL METEOR
wget https://www.cs.cmu.edu/~alavie/METEOR/download/meteor-1.5.tar.gz
tar -xvf meteor-1.5.tar.gz
mv meteor-1.5 metrics
rm meteor-1.5.tar.gz
```

Also (not mentioned on their readme, but necessary)
```
python -c 'import nltk; nltk.download("punkt")'
```

And for METEOR, make sure that Java JRE is installed on your machine

```
# on ubuntu
sudo apt-get install openjdk-8-jre

# on fedora, oracle linux, etc
yum install java-1.8.0-openjdk
```