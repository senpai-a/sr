python train.py -i "SR_training_datasets\T91" -o "T91Ex.p" -li -cgls -e
python train.py -i "SR_training_datasets\BSDS200" -o "B200li.p" -li -cgls
python train.py -i "SR_training_datasets\BSDS200" -o "B200Exli.p" -li -cgls -e
python train.py -i "SR_training_datasets\BSDS200" -o "B200cu.p" -cu -cgls
python train.py -i "SR_training_datasets\BSDS200" -o "B200Excu.p" -cu -cgls -e

python train.py -i "SR_training_datasets\BSDS200" -o "B200lils.p" -li -ls -l 0.01
python train.py -i "SR_training_datasets\BSDS200" -o "B200Exlils.p" -li -ls -e -l 0.01
python train.py -i "SR_training_datasets\BSDS200" -o "B200culs.p" -cu -ls -l 0.01
python train.py -i "SR_training_datasets\BSDS200" -o "B200Exculs.p" -cu -ls -e -l 0.01

python train.py -i "SR_training_datasets\G100" -o "G100Ex.p" -li -cgls -e

python test.py -f "B200li.p" -i "SR_testing_datasets\Set14" -o "B200cuSet14" -gt -cu
python test.py -f "B200lils.p" -i "SR_testing_datasets\Set14" -o "B200lils_Set14" -gt -li
python test.py -f "B200Exli.p" -i "SR_testing_datasets\Set14" -o "B200Exli_Set14" -gt -li -e
python test.py -f "B200Exlils.p" -i "SR_testing_datasets\Set14" -o "B200Exlils_Set14" -gt -li -e
 

python psnr.py -gt "SR_testing_datasets\Set5" -sr "results\bicubicSet5"
python bicubic.py -gt -i "SR_testing_datasets\Set5" -o "bicubicSet5"