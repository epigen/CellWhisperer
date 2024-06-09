BASEDIR=$(pwd)
export PYTHONPATH=$PYTHONPATH:$BASEDIR/metasra
export PYTHONPATH=$PYTHONPATH:$BASEDIR/metasra/map_sra_to_ontology
export PYTHONPATH=$PYTHONPATH:$BASEDIR/metasra/bktree

cd metasra/setup_map_sra_to_ontology
./setup.sh
