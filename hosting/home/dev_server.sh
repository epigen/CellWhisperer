git config --global --add safe.directory /opt/cellwhisperer/modules/cellxgene
make -C /opt/cellwhisperer/modules/cellxgene/client start-frontend

# cellxgene launch --debug -v -p 5005  --host 0.0.0.0 --max-category-items 500 --backed /dataset.h5ad  "http://cellwhisperer_clip:8910/api" & 
