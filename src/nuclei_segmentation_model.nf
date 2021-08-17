
// dataset paths
params.tnbc_path = "/data/dataset/nuc_segmentation/TNBC_NucleiSegmentation"
params.monuseg_path = "/data/dataset/nuc_segmentation/MoNucleiSegmentation"
params.monusac_path = "/data/dataset/nuc_segmentation/MoNuSAC_images_and_annotations"
params.consep_path = "/data/dataset/nuc_segmentation/consep/CoNSeP/"
params.cpm_path = "/data/dataset/nuc_segmentation/DataCPM"
params.pancrops_path = "/data/dataset/nuc_segmentation/pan_crops"

params.benchmark = 1

if (params.benchmark == 1){
    EPOCHS = 100
    MODELS = ['Unet', 'FPN', 'Linknet', 'PSPNet']
    BACKBONES = ['vgg16', 'resnet50', 'densenet121', 'inceptionv3', 'efficientnetb4']
    ENCODER = ['imagenet', 'None']
    LOSS = ['CE', 'focal', 'mse']
    LABELS = ["binary", "distance"]
    LR = [1e-1, 1e-2, 1e-3, 1e-4]
    WD = [1e-1, 5e-3, 5e-5]
    ALPHA = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    BETA= [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
} else {
    EPOCHS = 20
    MODELS = ['Unet']
    BACKBONES = ['vgg16']
    ENCODER = ['imagenet']
    LOSS = ['CE']
    LABELS = ["binary"]
    LR = [1e-2, 1e-3]
    WD = [5e-3]
    ALPHA = [1, 4]
    BETA= [0.2, 0.5]
}


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

process training {
    // to enable GPU
    containerOptions '--nv'

    beforeScript "source ${CWD}/environment/GPU_LOCKS/set_gpu.sh ${CWD}"
    afterScript  "source ${CWD}/environment/GPU_LOCKS/free_gpu.sh ${CWD}"

    input:
        set type, file(train), file(validation) from TRAIN_VAL_SET
        each backbone from BACKBONES
        each model from MODELS
        each lr from LR
        each wd from WD
        each encoder from ENCODER 
        each loss from LOSS
    output:
        set val("PARAM: type=${type}; backbone=${backbone}; model=${model}; lr=${lr}; wd=${wd}; encoder=${encoder}; loss=${loss}"), \
            val(type), file('model_weights.h5'), file('history.csv'), file('meta.pkl'), file(validation)  into TRAINED_MODELS
            
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
                        --epochs $EPOCHS \
                        --learning_rate $lr \
                        --weight_decay $wd \
                        --loss $loss
    """
}

pyvalidation = file("src/python/nn/validation.py")

process validation_with_ws {

    containerOptions '--nv'

    beforeScript "source ${CWD}/environment/GPU_LOCKS/set_gpu.sh ${CWD}"
    afterScript  "source ${CWD}/environment/GPU_LOCKS/free_gpu.sh ${CWD}"

    input:
        set file(param), type, file(weights), file(history), \
            file(meta), file(validation) from TRAINED_MODELS
        each alpha from ALPHA 
        each beta from BETA
    output:
        file('score.csv') into VALIDATION_SCORE
    when:
        (type == 'binary' && beta == 0.5) || (type == 'distance')
    script:
    """
    python $pyvalidation    --weights ${weights} \
                            --meta ${meta} \
                            --path ${validation} \
                            --alpha ${alpha} \
                            --beta ${beta} \
                            --param ${param} \
                            --aji

    """
    
}

VALIDATION_SCORE.collectFile(skip: 1, keepHeader: true)
                .set { ALL_VALIDATION }


pytest = file('src/python/nn/testing.py')
process test {

    publishDir "./output", mode: 'copy'

    containerOptions '--nv'
    beforeScript "source ${CWD}/environment/GPU_LOCKS/set_gpu.sh ${CWD}"
    afterScript  "source ${CWD}/environment/GPU_LOCKS/free_gpu.sh ${CWD}"

    input:
        file f from  ALL_VALIDATION
        set type, file(data) from TEST_SET
    output:
        set file('score.csv'), file('model_weights.h5'), file('meta.pkl'), file('final_score.csv'), file('samples')
    script:
    """
    python $pytest --path $data --scores $f --aji
    """
    
}