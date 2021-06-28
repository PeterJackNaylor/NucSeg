
// dataset paths
params.tnbc_path = "/data/dataset/nuc_segmentation/TNBC_NucleiSegmentation"
params.monuseg_path = "/data/dataset/nuc_segmentation/MoNucleiSegmentation"
params.monusac_path = "/data/dataset/nuc_segmentation/MoNuSAC_images_and_annotations"
params.consep_path = "/data/dataset/nuc_segmentation/consep/CoNSeP/"
params.cpm_path = "/data/dataset/nuc_segmentation/DataCPM"
params.pancrops_path = "/data/dataset/nuc_segmentation/pan_crops"

params.benchmark = 1

SIZE = 250 
DATASETS = Channel.from(["tnbc", file("src/python/datasets/tnbc.py"), file(params.tnbc_path)],
                        ["monuseg", file("src/python/datasets/monuseg.py"), file(params.monuseg_path)],
                        ["monusac", file("src/python/datasets/monusac.py"), file(params.monusac_path)],
                        ["consep", file("src/python/datasets/consep.py"), file(params.consep_path)],
                        ["cpm", file("src/python/datasets/cpm.py"), file(params.cpm_path)],
                        ["pancrops", file("src/python/datasets/pancrops.py"), file(params.pancrops_path)])

CWD=System.getProperty("user.dir")

process preparing_data {
    tag "${NAME}"
    containerOptions '-B /data:/data'

    input:
        set NAME, file(PY), file(PATH) from DATASETS
    output:
        file("Xy_${NAME}.npz") into XY

    script:
    """
    python $PY --path $PATH --size $SIZE
    """
}

split_data_into_tvt = file('src/python/datasets/train_valid_test_split.py')
LABELS = ["binary", "distance"]
process train_validation_test {
    input:
        file _ from XY .collect()
        each LABEL from LABELS
    output:
        set val("$LABEL"), "Xy_train.npz", "Xy_validation.npz" into TRAIN_VAL_SET
        set val("$LABEL"), "Xy_test.npz" into TEST_SET
    script:
    """
    python $split_data_into_tvt $LABEL
    """
}


pytraining = file("src/python/nn/training.py")
LR = [1e-1, 1e-2, 1e-3, 1e-4]
WD = [1e-1, 5e-3, 5e-5]

if (params.benchmark == 1){
    BACKBONES = ['Unet', 'FPN', 'Linknet', 'PSPNet']
    MODELS = ['vgg16', 'resnet50', 'densenet121', 'inceptionv3', 'efficientnetb4']
    ENCODER = ['imagenet', 'None']
    LOSS = ['CE', 'focal', 'mse']
} else {
    BACKBONES = ['Unet']
    MODELS = ['vgg16']
    ENCODER = ['imagenet']
    LOSS = ['CE']
}



process training {
    maxForks 1
    containerOptions '--nv'

    beforeScript "source ${CWD}/environment/GPU_LOCKS/set_gpu.sh ${CWD}"
    afterScript  "source ${CWD}/environment/GPU_LOCKS/free_gpu.sh ${CWD}"

    input:
        set type, file(train), file(validation) from TRAIN_VAL_SET
        each backbone from BACKBONES
        each model from MODELS
        each lr from LR
        each encoder from ENCODER 
        each loss from LOSS
    output:
        set val("PARAM: type=${type}; backbone=${backbone}; model=${model} ; lr={lr}; encoder=${encoder}; loss=${loss}"), \
            file('model.h5'), file('history.csv') into trained_models
    when:
        (type == 'distance' && loss == 'mse') || (type == 'binary' && loss != 'mse')
    script:
    """
    export SM_FRAMEWORK=tf.keras
    python $pytraining  --path_train $train \
                        --path_validation $validation \
                        --backbone $backbone \
                        --model $model \
                        --encoder $encoder \
                        --batch_size 32 \
                        --epochs 100 \
                        --learning_rate $lr \
                        --loss $loss
    """
}