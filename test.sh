echo 'OT FPL' &&
python test.py --config ./configs/test/test-sci1k-10.yaml --model save/rdn+btc-3rd_ot_fpl/epoch_186.pth --gpu 0 &&
echo 'OT' &&
python test.py --config ./configs/test/test-sci1k-10.yaml --model save/rdn+btc-3rd_ot/epoch_186.pth --gpu 0

true

  
# echo 'OT' &&
# bash scripts/test_SCI1K.sh save/rdn+btc-3rd_ot/epoch_186.pth 0 &&
# true
# python train.py --config configs/train/train-rdn+btc-3rd.yaml --tag _wo_res --gpu 0 --resume save/rdn+btc-3rd_wo_res/epoch_last.pth &&