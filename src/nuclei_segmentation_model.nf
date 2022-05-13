
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
    MODELS = ['Unet', 'FPN', 'Linknet']
    BACKBONES = ['resnet50']
    ENCODERS = ['imagenet']
    LOSSS = ['CE']
    LABELS = ["binary"]
    LRS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    WDS = [1e-1, 5e-3, 5e-5]
    ALPHAS = [6, 12, 18, 24, 30]
    BETAS = [0, 0.2, 0.5, 0.7]
} else {
    EPOCHS = 20
    MODELS = ['Unet']
    BACKBONES = ['vgg16']
    ENCODERS = ['imagenet']
    LOSSS = ['CE']
    LABELS = ["binary"]
    LRS = [1e-2, 1e-3]
    WDS = [5e-3]
    ALPHAS = [1, 4]
    BETAS = [0.2, 0.5]
}


SIZE = 250 
DATASETS = Channel.from(["tnbc", file("src/python/datasets/tnbc.py"), file(params.tnbc_path)],
                        ["monuseg", file("src/python/datasets/monuseg.py"), file(params.monuseg_path)],
                        ["monusac", file("src/python/datasets/monusac.py"), file(params.monusac_path)],
                        ["consep", file("src/python/datasets/consep.py"), file(params.consep_path)],
                        ["cpm", file("src/python/datasets/cpm.py"), file(params.cpm_path)],
                        ["pancrops", file("src/python/datasets/pancrops.py"), file(params.pancrops_path)])

CWD=System.getProperty("user.dir")

PY_NORMALIZE = file("src/python/datasets/normalize.py")
params.target = "None"
TARGET = params.target


process preparing_data {
    tag "${NAME}"
    containerOptions '-B /data:/data'

    input:
        tuple val(NAME), path(PY), path(PATH)
    output:
        path("Xy_${NAME}.npz")

    script:
        """
        python $PY --path $PATH --size $SIZE --target $TARGET
        """
}

split_data_into_tvt = file('src/python/datasets/train_valid_test_split.py')

process train_validation_test {
    input:
        path XY
        each LAB
    output:
        tuple val("$LAB"), path("Xy_train.npz"), path("Xy_validation.npz")
        tuple val("$LAB"), path("Xy_test.npz")
    script:
        """
        python $split_data_into_tvt $LAB
        """
}

pytraining = file("src/python/nn/training.py")

process training {
    // to enable GPU
    containerOptions '--nv'

    beforeScript "source ${CWD}/environment/GPU_LOCKS/set_gpu.sh ${CWD}"
    afterScript  "source ${CWD}/environment/GPU_LOCKS/free_gpu.sh ${CWD}"

    input:
        tuple val(TYPE), path(TRAIN), path(VALIDATION)
        each BACKBONE
        each MODEL
        each LR
        each WD
        each ENCODER 
        each LOSS
    output:
        tuple val("PARAM: type=${TYPE}; backbone=${BACKBONE}; model=${MODEL}; lr=${LR}; wd=${WD}; encoder=${ENCODER}; loss=${LOSS}"), \
            val(TYPE), path('model_weights.h5'), path('history.csv'), path('meta.pkl'), path(validation)
            
    when:
        (TYPE == 'distance' && LOSS == 'mse') || (TYPE == 'binary' && LOSS != 'mse')
    script:
        """
        export SM_FRAMEWORK=tf.keras
        python $pytraining  --path_train $TRAIN \
                            --path_validation $VALIDATION \
                            --backbone $BACKBONE \
                            --model $MODEL \
                            --encoder $ENCODER \
                            --batch_size 32 \
                            --epochs $EPOCHS \
                            --learning_rate $LR \
                            --weight_decay $WD \
                            --loss $LOSS
        """
}

pyvalidation = file("src/python/nn/validation.py")



process validation_with_ws {

    containerOptions '--nv'

    beforeScript "source ${CWD}/environment/GPU_LOCKS/set_gpu.sh ${CWD}"
    afterScript  "source ${CWD}/environment/GPU_LOCKS/free_gpu.sh ${CWD}"

    input:
        tuple path(PARAM), val(TYPE), path(WEIGHTS), path(HISTORY), \
            path(META), path(VALIDATION), val(F1_SCORE)
        each ALPHA 
        each BETA
    output:
        path('score.csv')
    when:
        ( F1_SCORE > 0.6 ) && ((TYPE == 'binary' && BETA == 0.5) || (TYPE == 'distance'))
    script:
        """
        python $pyvalidation    --weights ${WEIGHTS} \
                                --meta ${META} \
                                --path ${VALIDATION} \
                                --alpha ${ALPHA} \
                                --beta ${BETA} \
                                --history ${HISTORY} \
                                --param ${PARAM} \
                                --aji

        """
    
}

pytest = file('src/python/nn/testing.py')
process test {

    publishDir "./outputs/nuclei_segmentation_model", mode: 'copy'

    containerOptions '--nv'
    beforeScript "source ${CWD}/environment/GPU_LOCKS/set_gpu.sh ${CWD}"
    afterScript  "source ${CWD}/environment/GPU_LOCKS/free_gpu.sh ${CWD}"

    input:
        path(f)
        tuple val(TYPE), path(DATA)
    output:
        tuple path('score.csv'), path('model_weights.h5'), path('meta.pkl'), path('final_score.csv'), path('samples')
    script:
        """
        python $pytest --path $DATA --scores $f --aji
        """
    
}




workflow {
    main:
        preparing_data(DATASETS)
        train_validation_test(preparing_data.out.collect(), LABELS)
        training(train_validation_test.out[0], BACKBONES, MODELS, LRS, WDS, ENCODERS, LOSSS)

        training.out .map{ it0, it1, it2, history, it4, it5 -> [it0, it1, it2, history, it4, it5, Channel.fromPath(history).splitCsv(header: ["c1","c2","c3","c4","c5","c6","c7","c8","val_score","c9","c10","c11"], skip:1).map { row -> Float.valueOf("${row.val_score}") } .max () .val]}
                .set{trained_models}
        validation_with_ws(trained_models, ALPHAS, BETAS)
        test(validation_with_ws.out.collectFile(skip: 1, keepHeader: true), train_validation_test.out[1])
}
