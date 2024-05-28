BASEDIR=$(pwd)
export PYTHONPATH=$PYTHONPATH:$BASEDIR/metasra
export PYTHONPATH=$PYTHONPATH:$BASEDIR/metasra/map_sra_to_ontology
export PYTHONPATH=$PYTHONPATH:$BASEDIR/metasra/bktree

echo $PYTHONPATH

python metasra/run_pipeline.py $1 $2 $3
