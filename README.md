#cvpr anti-spoof workshop solution

##step1:  -->  train

### use 8 V100 to train
```
nohup sh dist_train.sh /home/dataSet/ > train.log &
```

------------------------ data root --------------------------------
```
#(base) [root@dl ]# ll /home/dataSet/
#total 4875908
#drwxrwxr-x.   92 1000 1000       4096 Jan 11 17:58 dev
#drwxrwxr-x.  167 1000 1000       4096 Feb 22 15:22 test
#drwxrwxr-x. 4021 1000 1000     196608 Jan 10 13:48 train
#---------------------------------------------------------
```

### show training logs
```
tailf train.log  
```

##step2:  -->  test

we can concatenate the dev.txt and test.txt, feed to the test procedure.
use this cmd
```
cat dev.txt test.txt > dev_test.txt
```
## run test
```
nohup sh test.sh ./datalist/dev_test.txt /home/dataSet dev_test_score.txt > test.log &
```

This code borrows heavily from https://github.com/snap-research/EfficientFormer