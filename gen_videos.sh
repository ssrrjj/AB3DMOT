# 0001 0002 0003 0004 0005 0006 0007 0008 0009 0010 0011 0012 0013 0014 0015 0016 0017 0018 0019 0020
for SEQ_NUM in 0000 0001 0002 0003 0004 0005 0006 0007 0008 0009 0010 0011 0012 0013 0014 0015 0016 0017 0018 0019 0020
    do

    echo "Starting $SEQ_NUM computation"
        python genvideo.py ./results/pointrcnn_thresh_world_eval_1/Car/trk_gt_vis/$SEQ_NUM ./videos/pointrcnn_thresh_world_eval_1/Car/gt$SEQ_NUM
    echo "Completed $SEQ_NUM computation"

    done
