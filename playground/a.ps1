python train.py -o "T91Ex.p"

python train.py -i "SR_training_datasets\G100" -o "G100Ex.p" -li -cgls -e

python test.py -f "g100" -i "SR_testing_datasets\Set5" -o "g100_Set5" -gt
python test.py -f "g100" -i "SR_testing_datasets\Set14" -o "g100_Set14" -gt
 
python test.py -f "B500cu.p" -i "SR_testing_datasets\Set14" -o "B500cu_Set14" -gt -cu
 
python test.py -f "B200Exli.p" -i "SR_testing_datasets\Set14" -o "B200Exli_Set14" -gt -li -e
python test.py -f "B200Excu.p" -i "SR_testing_datasets\Set14" -o "B200Excu_Set14" -gt -cu -e
 

python psnr.py -gt "SR_testing_datasets\Set5" -sr "results\bicubic_Set5"

python bicubic.py -gt -i "SR_testing_datasets\Set5" -o "bicubic_set5"
python bugslap.py -f "b200" -i "SR_testing_datasets\Set5" -o "test" -gt