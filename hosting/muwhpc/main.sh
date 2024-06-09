cd cellwhisperer
python src/model_server/main.py &
sleep 20
cellxgene launch --debug -v -p 5005  --host 0.0.0.0 --max-category-items 500 --backed ~/cellwhisperer/results/tabula_sapiens/cellwhisperer_clip_v1/cellxgene.h5ad  "http://localhost:8910/api" &
cellxgene launch --debug -v -p 5006  --host 0.0.0.0 --max-category-items 500 --backed ~/cellwhisperer/results/archs4_geo/cellwhisperer_clip_v1/cellxgene.h5ad  "http://localhost:8910/api" &
source activate nginx
nginx -c /msc/home/mschae83/cellwhisperer/hosting/muwhpc/nginx.conf
source activate ngrok
ngrok http 5100 --basic-auth "review:iclr2024-mlgenx" --domain upright-communal-panda.ngrok-free.app --log=stdout > ngrok.log

