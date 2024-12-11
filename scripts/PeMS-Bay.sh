
for name in gwnet HGCN STTNS T_GCN SANN DLinear DKFN AGCRN PDFormer GMAN\
STID ST_Norm D2STGNN AFDGCN STWave MegaCRN STGODE PGCN PMC_GCN STIDGCN\
DCRNN TESTAM WAVGCRN STGCN GAT_LSTM ASTRformer
do
for rate in 0.01 0.05 0.001 0.005 0.0001 0.0005
do
python -u main.py \
  --model_name $name\
  --data_name 'PeMS-Bay' \
  --lr $rate \
  --batch_size 32 \
  --resume_dir None\
  --output_dir None\
  --info 'None'\

done
done

for name in DGCN ASTGCN
do
for rate in 0.01 0.05 0.001 0.005 0.0001 0.0005
do
python -u main.py \
  --model_name $name\
  --exp_name 'deep_learning_interval'\
  --data_name 'PeMS-Bay' \
  --lr $rate \
  --batch_size 32 \
  --resume_dir None\
  --output_dir None\
  --info 'None'\

done
done




