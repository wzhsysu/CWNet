python main.py \
        mode=test \
        model.name=NAF_DA \
        model.ver=v1 \
        data.root=/PATH/TO/LSMI/galaxy_512/ \
        data.illum_aug=false \
        data.random_crop=false \
        criterion.key_pairs=[illum_loss] \
        camera=galaxy \
        load.ckpt_path=/PATH/TO/ckpt.pt \
        test.visualize_result=false