
// dataset paths
params.data_path = "/data/dataset/tissue_segmentation/data_tissue"

params.benchmark = 1

if (params.benchmark == 1){
    EPOCHS = 100
    MODELS = ['Unet', 'Linknet']
    BACKBONES = ['efficientnetb4']
    ENCODER = ['imagenet']
    LOSS = ['CE']
    LABELS = ["binary"]
    LR = [1e-2, 1e-3, 1e-4]
    WD = [5e-1, 5e-3, 5e-5]
} else {
    EPOCHS = 100
    MODELS = ['Unet']
    BACKBONES = ['resnet50']
    ENCODER = ['imagenet']
    LOSS = ['CE']
    LABELS = ["binary"]
    LR = [1e-2, 1e-3, 1e-4, 1e-5]
    WD = [5e-3]
}


SIZE = 224
prepare_data = file("src/python/datasets/tissue_segmentation.py")
data_path = file(params.data_path)

CWD=System.getProperty("user.dir")

process preparing_data {
    tag "tissueSeg"
    containerOptions '-B /data:/data'

    input:
        file PATH from data_path
    output:
        file("Xy_tissueSeg.npz") into XY

    script:
    """
    python $prepare_data --path $PATH --size $SIZE
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
                        --batch_size 16 \
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
    output:
        file('score.csv') into VALIDATION_SCORE
    script:
    """
    python $pyvalidation    --weights ${weights} \
                            --meta ${meta} \
                            --path ${validation} \
                            --param ${param} \
                            --no_aji

    """
    
}

VALIDATION_SCORE.collectFile(skip: 1, keepHeader: true)
                .set { ALL_VALIDATION }


pytest = file('src/python/nn/testing.py')
process test {

    publishDir "./output_tissue", mode: 'copy'

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
    python $pytest --path $data --scores $f --no_aji
    """
    
}